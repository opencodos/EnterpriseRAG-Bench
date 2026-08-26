"""Pool the candidates the trap and lure filter will judge.

The study's mining is method-blind: it does not ask one retriever what looks
adversarial, it pools two structurally different ones so that neither method's
blind spot decides what the bedrock contains. BM25 contributes its full-corpus
top-10 directly, and it also supplies the pool -- its top-200, or its top-1,000 for
the not-found questions, where the interesting candidates sit further down because
nothing actually answers the question -- that dense retrieval reranks in the next
stage.

Only the lexical half happens here, because it is the half that needs the whole
corpus in an index. Both halves of the pool come out of a single query per
question: the top-10 is a prefix of the top-200.

Requires the corpus indexed into OpenSearch:
    python -m src.scripts.answer_generation.index_document_bm25 --recreate

Usage:
    python -m ladder.pool [--index-name NAME] [--opensearch-url URL] [--resume]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ladder.common import OUT, load_questions, read_jsonl, write_jsonl

POOLS_PATH = OUT / "pools.jsonl"

# The pool the dense reranker reads. Deeper for the unanswerable questions: no
# document answers them, so the candidates worth judging are not near the top.
POOL_DEPTH = 200
POOL_DEPTH_NOT_FOUND = 1000
BM25_CONTRIBUTION = 10

NOT_FOUND = "info_not_found"


def _search(client: Any, index_name: str, question: dict[str, Any]) -> dict[str, Any]:
    depth = (
        POOL_DEPTH_NOT_FOUND if question["question_type"] == NOT_FOUND else POOL_DEPTH
    )
    response = client.search(
        index=index_name,
        body={
            "query": {"match": {"text": question["question"]}},
            "_source": ["dataset_doc_uuid"],
            "size": depth,
        },
        request_timeout=120,
    )
    hits = response["hits"]["hits"]
    ranked = [hit["_source"]["dataset_doc_uuid"] for hit in hits]
    scores = [hit["_score"] for hit in hits]
    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "pool_depth": depth,
        "pool": ranked,
        "bm25_scores": scores,
        "bm25_top10": ranked[:BM25_CONTRIBUTION],
    }


def build_pools(
    index_name: str, url: str, questions: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    from opensearchpy import OpenSearch

    client = OpenSearch(hosts=[url], http_compress=True, timeout=120)
    count = client.count(index=index_name)["count"]
    print(f"  {count:,} documents in {index_name}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pools = list(
            executor.map(
                lambda question: _search(client, index_name, question), questions
            )
        )
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool BM25 candidates for trap and lure mining."
    )
    parser.add_argument("--index-name", default="enterpriserag")
    parser.add_argument("--opensearch-url", default="http://localhost:9200")
    parser.add_argument("--output", type=Path, default=POOLS_PATH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume", action="store_true", help="Keep questions already pooled"
    )
    args = parser.parse_args()

    questions = load_questions()
    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        done = {row["question_id"]: row for row in read_jsonl(args.output)}
        print(f"  resuming: {len(done)} of {len(questions)} questions already pooled")

    pending = [
        question for question in questions if question["question_id"] not in done
    ]
    pools = (
        build_pools(args.index_name, args.opensearch_url, pending, args.workers)
        if pending
        else []
    )

    merged = {**done, **{pool["question_id"]: pool for pool in pools}}
    ordered = [
        merged[question["question_id"]]
        for question in questions
        if question["question_id"] in merged
    ]
    write_jsonl(args.output, ordered)

    shallow = [
        pool["question_id"]
        for pool in ordered
        if len(pool["pool"]) < pool["pool_depth"]
    ]
    print(f"  wrote {len(ordered)} pools to {args.output}")
    if shallow:
        print(
            f"  {len(shallow)} question(s) returned fewer hits than the pool depth, e.g. {shallow[:5]}"
        )


if __name__ == "__main__":
    main()
