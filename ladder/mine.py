"""Mine the traps and lures the bedrock carries beside its gold documents.

The second half of the study's method-blind procedure. Dense retrieval reranks the
BM25 pool and contributes ten candidates of its own; the union of those with BM25's
own top-10 is what an LLM filter then judges, one verdict per candidate:

  trap  the candidate concerns the same entity or topic as one of the question's
        gold documents while reporting the wrong version, date or decision. This
        is what stops a retriever from being right by being roughly on-topic.
  lure  for the unanswerable questions only, the five most similar candidates the
        filter confirms cannot answer the question, so that a not-found question
        cannot be solved by the mere absence of retrieved text.

Both are per-question judgements that become a corpus-wide set: a document mined
as a trap for one question may be gold for another, and the bedrock keeps one copy.
Those collisions are why the study's four components sum to more than the tier it
builds.

The filter is deliberately given the question's gold documents. A trap is defined
relative to what the right answer says, so a judge that cannot see the right answer
would be guessing at whether a version or a date is the wrong one.

Requires ladder/pools.jsonl (python -m ladder.pool) and an embedding + LLM key.

Usage:
    python -m ladder.mine [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.llm.factory import get_llm
from src.llm.interface import Message
from src.prompts.ladder_candidate_filter import (
    CANDIDATE_FILTER_PROMPT,
    LURE_FILTER_SYSTEM,
    TRAP_FILTER_SYSTEM,
)
from src.utils.json_extraction import extract_json_from_response

from ladder.common import (
    OUT,
    SOURCES_DIR,
    load_questions,
    load_uuid_index,
    read_document,
    read_jsonl,
    write_jsonl,
)
from ladder.pool import POOLS_PATH

TRAPS_PATH = OUT / "traps.jsonl"
LURES_PATH = OUT / "lures.jsonl"
VERDICTS_PATH = OUT / "verdicts.jsonl"
EMBEDDING_CACHE = OUT / "embedding_cache.jsonl"

EMBEDDING_MODEL = "text-embedding-3-large"
# Truncated well below the model's 3,072. These vectors only ever rerank a pool of
# at most a thousand documents that BM25 already selected -- they are not a corpus
# index, and the repository's own 3,072-dimension collection is a different job. At
# full width the cache for one run is ~5GB on disk and several times that resident,
# which is a real ceiling on resuming; Matryoshka truncation to 512 costs nothing
# measurable on a reranking task and makes the cache ordinary.
EMBEDDING_DIMENSIONS = 512
# The repository embeds a document as one vector; the same applies here, and a
# document longer than the model's window is cut at the last whole word.
EMBEDDING_CHAR_LIMIT = 30_000

DENSE_CONTRIBUTION = 10
LURES_PER_QUESTION = 5
NOT_FOUND = "info_not_found"

# Judged a few candidates at a time: one call per candidate wastes the shared
# question and gold context, and one call for all of them invites the judge to
# drift into ranking them against each other rather than judging each on its own.
FILTER_BATCH = 5

# How many times a judge that returns unparseable JSON is asked again before its
# batch is abandoned. A dropped batch costs a few candidates, never a wrong verdict.
FILTER_RETRIES = 3


def _judge_name() -> str:
    """The judge the seam will pick, for the run's own log and its provenance."""
    from src.llm.factory import LLM_MODEL_NAME, LLM_PROVIDER

    return f"{LLM_PROVIDER}/{LLM_MODEL_NAME or 'default'}"


def _document_body(dsid: str, index: dict[str, str], budget: int = 4000) -> str:
    document = read_document(SOURCES_DIR / index[dsid], SOURCES_DIR)
    if document is None:
        return ""
    text = document.text
    return text if len(text) <= budget else text[:budget].rsplit(" ", 1)[0] + " ..."


def _embed(client: Any, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=texts, dimensions=EMBEDDING_DIMENSIONS
    )
    return [item.embedding for item in response.data]


class Embeddings:
    """Embeds documents once and remembers them, since pools overlap across questions."""

    def __init__(self, client: Any, cache_path: Path) -> None:
        self._client = client
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        if cache_path.exists():
            for row in read_jsonl(cache_path):
                self._cache[row["key"]] = row["embedding"]
            print(f"  {len(self._cache):,} embeddings cached")

    def get(
        self, keyed_texts: dict[str, str], batch: int = 64
    ) -> dict[str, list[float]]:
        missing = [key for key in keyed_texts if key not in self._cache]
        for start in range(0, len(missing), batch):
            keys = missing[start : start + batch]
            vectors = _embed(
                self._client, [keyed_texts[key][:EMBEDDING_CHAR_LIMIT] for key in keys]
            )
            with self._cache_path.open("a", encoding="utf-8") as handle:
                for key, vector in zip(keys, vectors):
                    self._cache[key] = vector
                    handle.write(json.dumps({"key": key, "embedding": vector}) + "\n")
        return {key: self._cache[key] for key in keyed_texts}


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        + 1e-12
    )


def dense_rerank(
    embeddings: Embeddings,
    question: dict[str, Any],
    pool: list[str],
    index: dict[str, str],
) -> tuple[list[str], dict[str, float]]:
    """Rerank the BM25 pool by embedding similarity.

    Returns the study's ten contributed candidates and the similarity of every
    pooled document. The full scores are kept because lures are defined as the most
    *similar* candidates a filter clears, and a candidate BM25 contributed needs a
    similarity too or it would sort behind everything the dense half proposed.
    """
    bodies = {
        dsid: _document_body(dsid, index, EMBEDDING_CHAR_LIMIT)
        for dsid in pool
        if dsid in index
    }
    vectors = embeddings.get(
        {f"doc:{dsid}": body for dsid, body in bodies.items() if body}
    )
    query = embeddings.get({f"q:{question['question_id']}": question["question"]})
    query_vector = query[f"q:{question['question_id']}"]
    scored = [
        (dsid, _cosine(query_vector, vectors[f"doc:{dsid}"]))
        for dsid in bodies
        if f"doc:{dsid}" in vectors
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [dsid for dsid, _ in scored[:DENSE_CONTRIBUTION]], dict(scored)


def _judge(
    question: dict[str, Any], candidates: list[str], index: dict[str, str]
) -> list[dict[str, Any]]:
    unanswerable = question["question_type"] == NOT_FOUND
    gold_block = ""
    if not unanswerable:
        gold_bodies = [
            _document_body(dsid, index)
            for dsid in (question.get("expected_doc_ids") or [])
            if dsid in index
        ]
        gold_block = (
            "Gold documents (these answer the question):\n"
            + "\n\n---\n\n".join(gold_bodies)
            + "\n\n"
        )

    rendered = "\n\n---\n\n".join(
        f"[{position}] {_document_body(dsid, index)}"
        for position, dsid in enumerate(candidates)
    )
    messages = [
        Message(
            role="system",
            content=LURE_FILTER_SYSTEM if unanswerable else TRAP_FILTER_SYSTEM,
        ),
        Message(
            role="user",
            content=CANDIDATE_FILTER_PROMPT.format(
                question=question["question"],
                gold_block=gold_block,
                candidates=rendered,
            ),
        ),
    ]

    for attempt in range(FILTER_RETRIES):
        llm = get_llm(tools=None, quiet=True)
        response = ""
        for chunk in llm.generate(messages):
            if isinstance(chunk, str):
                response += chunk
        try:
            parsed = json.loads(extract_json_from_response(response))
        except (json.JSONDecodeError, ValueError):
            if attempt == FILTER_RETRIES - 1:
                return []
            continue
        return _verdicts_of(parsed, candidates)
    return []


def _verdicts_of(parsed: Any, candidates: list[str]) -> list[dict[str, Any]]:
    """Map a judge's positional verdicts back onto the candidates it was shown.

    A position the judge invented, repeated or dropped is discarded rather than
    guessed at: an unmatched verdict is a missing trap, and a misattributed one is
    a wrong document in the bedrock of every tier.
    """
    verdicts = []
    claimed: set[int] = set()
    for entry in parsed.get("verdicts", []) if isinstance(parsed, dict) else []:
        position = entry.get("candidate")
        if not isinstance(position, int) or not 0 <= position < len(candidates):
            continue
        if position in claimed:
            continue
        claimed.add(position)
        verdicts.append(
            {
                "dsid": candidates[position],
                "verdict": entry.get("verdict"),
                "why": entry.get("why"),
            }
        )
    return verdicts


def mine_question(
    embeddings: Embeddings,
    question: dict[str, Any],
    pool_row: dict[str, Any],
    index: dict[str, str],
) -> dict[str, Any]:
    own_gold = set(question.get("expected_doc_ids") or [])
    contributed, dense_score = dense_rerank(
        embeddings, question, pool_row["pool"], index
    )
    dense_rank = {dsid: position for position, dsid in enumerate(contributed)}

    candidates = [
        dsid for dsid in pool_row["bm25_top10"] + contributed if dsid not in own_gold
    ]
    candidates = list(dict.fromkeys(candidates))

    verdicts: list[dict[str, Any]] = []
    for start in range(0, len(candidates), FILTER_BATCH):
        verdicts.extend(
            _judge(question, candidates[start : start + FILTER_BATCH], index)
        )

    for verdict in verdicts:
        verdict["question_id"] = question["question_id"]
        verdict["question_type"] = question["question_type"]
        verdict["bm25_rank"] = (
            pool_row["pool"].index(verdict["dsid"])
            if verdict["dsid"] in pool_row["pool"]
            else None
        )
        verdict["dense_rank"] = dense_rank.get(verdict["dsid"])
        verdict["dense_score"] = dense_score.get(verdict["dsid"])
    return {"question_id": question["question_id"], "verdicts": verdicts}


def collect(
    verdict_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn per-question verdicts into the corpus-wide trap and lure sets.

    A document mined for several questions is kept once, credited to the question
    that ranked it highest -- the collisions the study removes as cross-category
    duplicates fall out of this rather than being counted separately.
    """
    traps: dict[str, dict[str, Any]] = {}
    lures: dict[str, dict[str, Any]] = {}

    for row in verdict_rows:
        verdicts = row["verdicts"]
        unanswerable = any(
            verdict["question_type"] == NOT_FOUND for verdict in verdicts
        )
        if unanswerable:
            confirmed = [
                verdict for verdict in verdicts if verdict["verdict"] == "cannot_answer"
            ]
            confirmed.sort(
                key=lambda verdict: verdict["dense_score"] or 0, reverse=True
            )
            for verdict in confirmed[:LURES_PER_QUESTION]:
                kept = lures.get(verdict["dsid"])
                if kept is None or (verdict["dense_score"] or 0) > (
                    kept["dense_score"] or 0
                ):
                    lures[verdict["dsid"]] = verdict
            continue
        for verdict in verdicts:
            if verdict["verdict"] != "trap":
                continue
            kept = traps.get(verdict["dsid"])
            if kept is None or (verdict["dense_score"] or 0) > (
                kept["dense_score"] or 0
            ):
                traps[verdict["dsid"]] = verdict

    return list(traps.values()), list(lures.values())


def _completed_verdicts() -> list[dict[str, Any]]:
    """Read back an appended verdict log, tolerating the tail a kill left behind.

    The log is appended to as questions finish, so an interrupted run can end on a
    half-written line. Dropping it costs one question's verdicts, which the resume
    then re-judges; refusing to parse the file would cost every question in it.
    """
    rows = []
    for line in VERDICTS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine traps and lures from the pooled candidates."
    )
    parser.add_argument("--pools", type=Path, default=POOLS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI()
    questions = {question["question_id"]: question for question in load_questions()}
    index = load_uuid_index()
    pools = read_jsonl(args.pools)
    if args.limit:
        pools = pools[: args.limit]

    done: dict[str, dict[str, Any]] = {}
    if args.resume and VERDICTS_PATH.exists():
        done = {row["question_id"]: row for row in _completed_verdicts()}
        print(f"  resuming: {len(done)} of {len(pools)} questions already judged")

    embeddings = Embeddings(client, EMBEDDING_CACHE)
    pending = [pool for pool in pools if pool["question_id"] not in done]
    print(f"  judging {len(pending)} question(s) with {_judge_name()}")

    # Appended as each question finishes rather than written at the end: this is a
    # run of hours against a paid judge, and a crash an hour in must cost the last
    # question rather than every one of them.
    judged: dict[str, dict[str, Any]] = {}
    append_lock = threading.Lock()

    def judge_one(pool: dict[str, Any]) -> None:
        row = mine_question(embeddings, questions[pool["question_id"]], pool, index)
        with append_lock:
            judged[row["question_id"]] = row
            with VERDICTS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(judged) % 25 == 0:
                print(f"  judged {len(judged)}/{len(pending)}")

    VERDICTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(judge_one, pending))

    # Rewritten whole so that a resumed run leaves one row per question rather than
    # the appended history of however many sittings it took.
    rows = list({**done, **judged}.values())
    write_jsonl(VERDICTS_PATH, rows)

    traps, lures = collect(rows)
    write_jsonl(TRAPS_PATH, traps)
    write_jsonl(LURES_PATH, lures)
    print(f"  {len(traps)} traps (study: 326), {len(lures)} lures (study: 99)")
    print(f"  wrote {TRAPS_PATH}, {LURES_PATH}, {VERDICTS_PATH}")


if __name__ == "__main__":
    main()
