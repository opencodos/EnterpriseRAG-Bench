"""Arm 1: the study's BM25 paradigm over one tier, read by the study's reader.

Retrieves the top-5 documents for each question, renders them into a context block, and
asks the reader for an answer under the repository's own answer-generation prompt.
Writes two files, because this arm produces the control arm's input as well as its own
result:

* an **answers** file in the official format the scorer reads, and
* a **contexts** file -- the same questions, the same documents, and the context block
  this arm's reader was given, verbatim.

The second exists so the control arm (BM25's retrieval read by an Aethos-tier model)
differs from this one in the model and in nothing else. Reassembling a context there
from the chunks would make the two differ in the rendering as well, which is the one
thing a control may not do.

Usage:
    python -m arms.bm25 --tier-tree /data/tier-T0 --out-dir results/T0/bm25
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch
from tqdm import tqdm

from arms.common import (
    Chunk,
    GRANULARITIES,
    RETRIEVAL_GRANULARITY,
    TOP_K,
    Tier,
    documents_of,
    format_context_chunks,
    load_tier,
)
from arms.index_bm25 import DEFAULT_OPENSEARCH_URL, index_name_for
from arms.run import (
    FAILED_ANSWER_TEXT,
    bind_run_identity,
    load_core_questions,
    preflight_reader,
    reconcile_outputs,
    report_run,
    write_row,
)
from src.llm.factory import get_llm
from src.llm.interface import Message
from src.prompts.vector_search_answer_gen import ANSWER_GEN_PROMPT


def retrieve(client: OpenSearch, index_name: str, query: str, top_k: int) -> list[Chunk]:
    """The top-*k* chunks for *query*, best first."""
    response = client.search(
        index=index_name,
        body={
            "query": {"match": {"text": {"query": query}}},
            "size": top_k,
            "_source": ["dataset_doc_uuid", "chunk_index", "chunk_total", "title", "text"],
        },
    )
    return [
        Chunk(
            dsid=hit["_source"]["dataset_doc_uuid"],
            index=hit["_source"]["chunk_index"],
            total=hit["_source"]["chunk_total"],
            title=hit["_source"]["title"],
            text=hit["_source"]["text"],
        )
        for hit in response["hits"]["hits"]
    ]


def answer(question: str, context: str, quiet: bool) -> str:
    """One reader call over a rendered context block."""
    prompt = ANSWER_GEN_PROMPT.format(context_documents=context, question=question)
    llm = get_llm(tools=None, quiet=quiet)
    return "".join(
        chunk
        for chunk in llm.generate([Message(role="user", content=prompt)])
        if isinstance(chunk, str)
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval + reader over one tier.")
    parser.add_argument("--tier-tree", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--opensearch-url", default=DEFAULT_OPENSEARCH_URL)
    parser.add_argument("--index-name", default=None)
    parser.add_argument(
        "--granularity",
        default=RETRIEVAL_GRANULARITY,
        choices=GRANULARITIES,
        help=(
            f"What the top-k counts: whole documents or windowed chunks "
            f"(default: {RETRIEVAL_GRANULARITY}, the study's)"
        ),
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help=(
            "Questions in flight. Above 1 the recorded latency is a queueing time "
            "rather than the arm's, so hold it at whatever every other tier used."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test escape hatch")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if args.top_k != TOP_K:
        print(
            f"[warn] retrieval depth {args.top_k} is not the study's {TOP_K}; this run "
            f"is not comparable to the published curve"
        )

    if args.granularity != RETRIEVAL_GRANULARITY:
        print(
            f"[warn] granularity {args.granularity!r} is not the study's "
            f"{RETRIEVAL_GRANULARITY!r}; this run is not comparable to the published "
            f"curve. Under chunk-level retrieval the reader sees only the matching "
            f"window of a retrieved document, not the document."
        )

    tier: Tier = load_tier(args.tier_tree)
    index_name = args.index_name or index_name_for(tier, args.granularity)
    questions = load_core_questions(limit=args.limit)

    # Before anything is read back: resume keys off the question id alone, so an
    # output directory has to be pinned to the settings that wrote it or a second run
    # under different ones would report the first run's answers as its own.
    bind_run_identity(
        args.out_dir,
        {
            "arm": "bm25",
            "tier": tier.name,
            "manifest_sha256": tier.manifest_sha256,
            "index": index_name,
            "granularity": args.granularity,
            "top_k": args.top_k,
            "parallelism": args.parallelism,
        },
    )
    answers_path = args.out_dir / "answers.jsonl"
    contexts_path = args.out_dir / "contexts.jsonl"
    # Both files are one row per question, and the control arm refuses a contexts file
    # short of the question set -- so a run interrupted between the two writes is
    # trimmed back to what both hold before anything else happens.
    done = reconcile_outputs(answers_path, contexts_path)

    if not args.skip_preflight:
        preflight_reader()

    pending = [q for q in questions if q["question_id"] not in done]
    print(
        f"{tier.name} / bm25: {len(questions)} question(s), {len(done)} already answered, "
        f"{len(pending)} pending -> {args.out_dir}"
    )
    settings = {
        "index": index_name,
        "granularity": args.granularity,
        "top_k": args.top_k,
    }
    if not pending:
        report_run(
            answers_path,
            questions,
            arm="bm25",
            tier=tier,
            parallelism=args.parallelism,
            extra=settings,
        )
        return

    client = OpenSearch(hosts=[args.opensearch_url], use_ssl=False, verify_certs=False)
    if not client.indices.exists(index=index_name):
        raise SystemExit(
            f"index '{index_name}' does not exist; run `python -m arms.index_bm25 "
            f"--tier-tree {args.tier_tree}` first"
        )

    quiet = args.parallelism > 1
    lock = threading.Lock()

    def process(question: dict[str, Any]) -> None:
        started = time.perf_counter()
        chunks = retrieve(client, index_name, question["question"], args.top_k)
        context = format_context_chunks(chunks)
        document_ids = documents_of(chunks)
        # A question with no retrieval still gets a context block and an answer: the
        # study's not-found questions are supposed to end that way, and dropping the
        # row would take the question out of the denominator.
        try:
            text = answer(question["question"], context or "(no documents retrieved)", quiet)
            failure = None
        except Exception as exc:  # noqa: BLE001 -- the row records it either way
            text, failure = FAILED_ANSWER_TEXT, str(exc)
        latency = time.perf_counter() - started

        with lock:
            # Context first: a crash between the two writes then leaves a context row
            # with no answer, which the next run's reconcile trims. The other order
            # would leave an answered question whose context was never recorded.
            write_row(
                contexts_path,
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "document_ids": document_ids,
                    "context": context,
                },
            )
            write_row(
                answers_path,
                {
                    "question_id": question["question_id"],
                    "answer": text,
                    "document_ids": document_ids,
                    "latency_seconds": latency,
                    "chunks_retrieved": len(chunks),
                    "failure": failure,
                },
            )

    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {pool.submit(process, q): q["question_id"] for q in pending}
        with tqdm(total=len(pending), desc="Questions") as bar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((futures[future], exc))
                bar.update(1)

    for qid, exc in failures:
        print(f"  {qid} did not produce a row: {exc}")

    report_run(
        answers_path,
        questions,
        arm="bm25",
        tier=tier,
        parallelism=args.parallelism,
        extra=settings,
    )


if __name__ == "__main__":
    main()
