"""Measure the corpus once, so every later stage reads numbers instead of guessing.

Two things are settled here. The first is the tokenizer question the tier table
cannot be checked without: the study publishes corpus tokens, chunk counts and
mean document tokens per tier but not the tokenizer that produced them, and the
choice moves the targets by percent. The full corpus is itself a published row --
511,959 documents, 600.8M tokens, 737,878 chunks, 1173.6 mean -- so a candidate
tokenizer either reproduces that row or does not.

None of them does. ``--compare-tokenizers`` measures the plausible candidates over
a sample and every one lands high: cl100k_base +2.9%, o200k_base +2.0%, and the
reader's own Qwen3 tokenizer +7.7%, against a published number that no rounding
explains. The study's tokenizer is therefore not recoverable from what it printed,
and the tier check cannot be an equality. What survives is the *shape*: the
published means fall 1472.7 -> 1173.6 from bedrock to full corpus, and that decline
is a property of the construction rather than of the tokenizer, so each tier's mean
divided by the full corpus's mean is comparable across tokenizers even when the
absolute is not. Those ratios are what build_manifests.py validates against, and
this module computes the targets. cl100k_base is adopted as the reporting
tokenizer -- not because it is closest, o200k_base is, but because it is what this
repository's own indexer chunks and embeds with, and internal consistency with the
harness that measures the arms is worth more than 1.6% on an absolute nobody can
reproduce anyway.

The second thing settled is composition: how many documents carry a noise marker,
how gold spreads across the nine source families, and how far 722 gold documents
sit from the published bedrock of 1,144.

Writes corpus_stats.jsonl -- one row per document, the token cache later stages
validate tiers from -- and census.json, the small committed summary.

Usage:
    python -m ladder.census [--sources-dir PATH] [--jobs N] [--reuse]
    python -m ladder.census --compare-tokenizers [--sample N]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

from ladder.common import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENCODING_NAME,
    OUT,
    SOURCES_DIR,
    UUID_INDEX_PATH,
    Document,
    chunk_count,
    corpus_paths,
    file_digest,
    gold_dsids,
    load_questions,
    load_uuid_index,
    not_found_questions,
    read_document,
    worker_count,
)

STATS_PATH = OUT / "corpus_stats.jsonl"
CENSUS_PATH = OUT / "census.json"

# Table 7 of the scaling study, at the four rungs this ladder builds and at the
# full corpus. The full-corpus row is what a tokenizer is calibrated against; the
# tier rows are what a built ladder is validated against.
PUBLISHED_TIERS = {
    "T0": {
        "documents": 1_144,
        "corpus_tokens": 1_700_000,
        "chunks": 2_018,
        "mean_doc_tokens": 1472.7,
    },
    "T3": {
        "documents": 2_254,
        "corpus_tokens": 3_000_000,
        "chunks": 3_606,
        "mean_doc_tokens": 1324.0,
    },
    "T8": {
        "documents": 6_980,
        "corpus_tokens": 8_500_000,
        "chunks": 10_419,
        "mean_doc_tokens": 1217.1,
    },
    "T13": {
        "documents": 21_614,
        "corpus_tokens": 25_600_000,
        "chunks": 31_475,
        "mean_doc_tokens": 1186.6,
    },
}
PUBLISHED_FULL_CORPUS = {
    "documents": 511_959,
    "corpus_tokens": 600_800_000,
    "chunks": 737_878,
    "mean_doc_tokens": 1173.6,
}

# The bedrock the study reports for its smallest tier, before and after the
# cross-category duplicates it removes.
PUBLISHED_BEDROCK = {
    "gold": 722,
    "traps": 326,
    "lures": 99,
    "organizational_pages": 2,
    "cross_category_duplicates": 5,
    "documents": 1144,
}

# Candidates for the study's unpublished tokenizer: this repository's own, the
# larger OpenAI vocabulary, and the tokenizer of the reader the study served.
TOKENIZER_CANDIDATES = ("cl100k_base", "o200k_base", "Qwen/Qwen3-Embedding-0.6B")

_ENCODERS: dict[str, Any] = {}


def _encoder(name: str) -> Any:
    if name not in _ENCODERS:
        if "/" in name:
            from transformers import AutoTokenizer

            _ENCODERS[name] = AutoTokenizer.from_pretrained(name)
        else:
            import tiktoken

            _ENCODERS[name] = tiktoken.get_encoding(name)
    return _ENCODERS[name]


def token_length(text: str, tokenizer: str = ENCODING_NAME) -> int:
    encoder = _encoder(tokenizer)
    if "/" in tokenizer:
        return len(encoder(text, add_special_tokens=False)["input_ids"])
    return len(encoder.encode(text, disallowed_special=()))


def _measure(
    paths: list[Path], sources_dir: Path, tokenizer: str
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        document = read_document(path, sources_dir)
        if document is not None:
            rows.append(_row(document, tokenizer))
    return rows


def _row(document: Document, tokenizer: str) -> dict[str, Any]:
    tokens = token_length(document.text, tokenizer)
    return {
        "dsid": document.dsid,
        "path": document.relative_path,
        "source": document.source,
        "noise": document.is_noise,
        "tokens": tokens,
        "chunks": chunk_count(tokens),
    }


def measure_corpus(
    sources_dir: Path, jobs: int, tokenizer: str
) -> list[dict[str, Any]]:
    paths = corpus_paths(sources_dir)
    print(f"  {len(paths):,} candidate files under {sources_dir}")
    if jobs == 1:
        return _measure(paths, sources_dir, tokenizer)

    import multiprocessing as mp

    batches = [paths[start::jobs] for start in range(jobs)]
    with mp.Pool(jobs) as pool:
        measured = pool.starmap(
            _measure, [(batch, sources_dir, tokenizer) for batch in batches]
        )
    rows = [row for batch in measured for row in batch]
    rows.sort(key=lambda row: row["path"])
    return rows


def compare_tokenizers(
    sources_dir: Path, sample_size: int, seed: int = 7
) -> dict[str, Any]:
    """Measure every candidate tokenizer over one sample, against the published mean.

    Reported as a ratio to the adopted tokenizer rather than as a sample mean,
    because a sample's mean is not the corpus's but the ratio between two
    tokenizers on the same documents is stable.
    """
    rng = random.Random(seed)
    paths = corpus_paths(sources_dir)
    documents = [
        doc
        for path in rng.sample(paths, min(sample_size, len(paths)))
        if (doc := read_document(path, sources_dir))
    ]
    texts = {
        "title+content": [doc.text for doc in documents],
        "content only": [doc.text.split("\n\n", 1)[1] for doc in documents],
    }
    base = sum(token_length(text, ENCODING_NAME) for text in texts["title+content"])
    published_mean = PUBLISHED_FULL_CORPUS["mean_doc_tokens"]

    variants = {}
    for tokenizer in TOKENIZER_CANDIDATES:
        for form, batch in texts.items():
            total = sum(token_length(text, tokenizer) for text in batch)
            ratio = total / base
            variants[f"{tokenizer} / {form}"] = {
                "ratio_to_adopted": round(ratio, 4),
                "sample_mean_doc_tokens": round(total / len(documents), 1),
            }
    return {
        "sample": len(documents),
        "seed": seed,
        "adopted": f"{ENCODING_NAME} / title+content",
        "published_full_corpus_mean": published_mean,
        "variants": variants,
        "conclusion": (
            "no candidate reproduces the published mean; the study's tokenizer is not "
            "recoverable from Table 7, so tier validation compares shape (per-tier mean "
            "divided by full-corpus mean) rather than absolute token counts"
        ),
    }


def tier_shape_targets(measured_full_mean: float) -> dict[str, Any]:
    """Per-tier mean-token targets, rescaled from the published ratios.

    A tier's mean divided by the full corpus's mean is the same number under any
    tokenizer that scales roughly uniformly, so this is the part of Table 7 that
    survives not knowing which tokenizer produced it.
    """
    published_full = PUBLISHED_FULL_CORPUS["mean_doc_tokens"]
    targets = {}
    for tier, row in PUBLISHED_TIERS.items():
        ratio = row["mean_doc_tokens"] / published_full
        targets[tier] = {
            "documents": row["documents"],
            "published_mean_doc_tokens": row["mean_doc_tokens"],
            "ratio_to_full_corpus": round(ratio, 4),
            "expected_mean_doc_tokens": round(measured_full_mean * ratio, 1),
        }
    return targets


def summarize(
    rows: list[dict[str, Any]], sources_dir: Path, tokenizer: str
) -> dict[str, Any]:
    questions = load_questions()
    gold = gold_dsids(questions)
    index = load_uuid_index()
    by_dsid = {row["dsid"]: row for row in rows}

    total_tokens = sum(row["tokens"] for row in rows)
    documents = len(rows)
    measured = {
        "documents": documents,
        "corpus_tokens": total_tokens,
        "chunks": sum(row["chunks"] for row in rows),
        "mean_doc_tokens": round(total_tokens / documents, 1),
    }

    noise = [row for row in rows if row["noise"]]
    gold_rows = [by_dsid[dsid] for dsid in gold if dsid in by_dsid]
    gold_tokens = sum(row["tokens"] for row in gold_rows)
    unindexed = [row["dsid"] for row in rows if row["dsid"] not in index]

    return {
        "tokenizer": tokenizer,
        "text": "title + '\\n\\n' + content, as the BM25 indexer joins them",
        "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "inputs": {
            "uuid_index_documents": len(index),
            "uuid_index_sha256": file_digest(UUID_INDEX_PATH),
            "questions": len(questions),
        },
        "full_corpus": {
            "measured": measured,
            "published": PUBLISHED_FULL_CORPUS,
            "delta_pct": {
                key: round(
                    100
                    * (measured[key] - PUBLISHED_FULL_CORPUS[key])
                    / PUBLISHED_FULL_CORPUS[key],
                    2,
                )
                for key in measured
            },
            "documents_absent_from_uuid_index": len(unindexed),
        },
        "tier_shape_targets": tier_shape_targets(measured["mean_doc_tokens"]),
        "by_source": dict(
            collections.Counter(row["source"] for row in rows).most_common()
        ),
        "noise": {
            "documents": len(noise),
            "share_pct": round(100 * len(noise) / documents, 4),
            "by_source": dict(
                collections.Counter(row["source"] for row in noise).most_common()
            ),
        },
        "gold": {
            "documents": len(gold),
            "resolved_in_corpus": len(gold_rows),
            "mean_doc_tokens": (
                round(gold_tokens / len(gold_rows), 1) if gold_rows else 0
            ),
            "by_source": dict(
                collections.Counter(
                    index[dsid].split("/")[0] for dsid in gold if dsid in index
                ).most_common()
            ),
            "noise_documents": sum(1 for row in gold_rows if row["noise"]),
        },
        "questions": {
            "total": len(questions),
            "by_type": dict(
                collections.Counter(q["question_type"] for q in questions).most_common()
            ),
            "not_found": len(not_found_questions(questions)),
            "without_expected_docs": sum(
                1 for q in questions if not q.get("expected_doc_ids")
            ),
        },
        "bedrock_target": dict(
            PUBLISHED_BEDROCK,
            non_gold_slots=PUBLISHED_BEDROCK["documents"] - PUBLISHED_BEDROCK["gold"],
            implied_mean_trap_lure_tokens=_implied_adversarial_mean(
                measured["mean_doc_tokens"],
                gold_tokens / len(gold_rows) if gold_rows else 0,
            ),
        ),
    }


def _implied_adversarial_mean(full_mean: float, gold_mean: float) -> float:
    """How long the mined traps and lures have to be for T0's mean to come out right.

    Falls straight out of the published bedrock: 722 gold documents of known length
    plus 422 mined ones make up a tier whose mean the study printed. A mining run
    that lands far from this number has selected the wrong kind of document, and
    that is checkable before a single answer is generated.
    """
    target = full_mean * (
        PUBLISHED_TIERS["T0"]["mean_doc_tokens"]
        / PUBLISHED_FULL_CORPUS["mean_doc_tokens"]
    )
    gold_count = PUBLISHED_BEDROCK["gold"]
    mined = PUBLISHED_BEDROCK["documents"] - gold_count
    return round(
        (target * PUBLISHED_BEDROCK["documents"] - gold_mean * gold_count) / mined, 1
    )


def report(census: dict[str, Any]) -> None:
    full = census["full_corpus"]
    print("\nFull corpus, measured against the study's published full-corpus row:")
    for key in ("documents", "corpus_tokens", "chunks", "mean_doc_tokens"):
        print(
            f"  {key:<16} measured {full['measured'][key]:>14,} "
            f"published {full['published'][key]:>14,} delta {full['delta_pct'][key]:>7.2f}%"
        )
    print("\nTier shape targets (published ratio rescaled to this tokenizer):")
    for tier, target in census["tier_shape_targets"].items():
        print(
            f"  {tier:<4} {target['documents']:>7,} docs  published mean {target['published_mean_doc_tokens']:>7.1f}"
            f"  ratio {target['ratio_to_full_corpus']:.4f}  expected mean {target['expected_mean_doc_tokens']:>7.1f}"
        )
    print(
        f"\nNoise documents: {census['noise']['documents']:,} ({census['noise']['share_pct']}% of corpus)"
    )
    print(
        f"Gold documents:  {census['gold']['documents']:,}, mean {census['gold']['mean_doc_tokens']} tokens"
    )
    print(f"  by source: {census['gold']['by_source']}")
    target = census["bedrock_target"]
    print(
        f"\nBedrock to hit: {target['documents']:,} documents "
        f"= {target['gold']} gold + {target['traps']} traps + {target['lures']} lures "
        f"+ {target['organizational_pages']} organizational pages "
        f"- {target['cross_category_duplicates']} cross-category duplicates"
    )
    print(f"  non-gold slots to mine: {target['non_gold_slots']}")
    print(
        f"  traps and lures must average ~{target['implied_mean_trap_lure_tokens']:,.0f} tokens for T0's mean to land"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the corpus the ladder is built from."
    )
    parser.add_argument("--sources-dir", type=Path, default=SOURCES_DIR)
    parser.add_argument("--jobs", type=int, default=worker_count())
    parser.add_argument("--tokenizer", default=ENCODING_NAME)
    parser.add_argument(
        "--reuse", action="store_true", help="Summarize an existing corpus_stats.jsonl"
    )
    parser.add_argument(
        "--compare-tokenizers",
        action="store_true",
        help="Only calibrate tokenizers, over a sample",
    )
    parser.add_argument("--sample", type=int, default=4000)
    args = parser.parse_args()

    if args.compare_tokenizers:
        comparison = compare_tokenizers(args.sources_dir, args.sample)
        print(json.dumps(comparison, indent=2))
        if CENSUS_PATH.exists():
            census = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
            census["tokenizer_calibration"] = comparison
            CENSUS_PATH.write_text(
                json.dumps(census, indent=2) + "\n", encoding="utf-8"
            )
            print(f"\n  recorded in {CENSUS_PATH}")
        return

    if args.reuse and STATS_PATH.exists():
        print(f"  reusing {STATS_PATH}")
        rows = [
            json.loads(line)
            for line in STATS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        print(f"  measuring with {args.jobs} workers under {args.tokenizer}")
        rows = measure_corpus(args.sources_dir, args.jobs, args.tokenizer)
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATS_PATH.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"  wrote {len(rows):,} rows to {STATS_PATH}")

    census = summarize(rows, args.sources_dir, args.tokenizer)
    if CENSUS_PATH.exists():
        previous = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
        if "tokenizer_calibration" in previous:
            census["tokenizer_calibration"] = previous["tokenizer_calibration"]
    CENSUS_PATH.write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
    report(census)
    print(f"\n  wrote {CENSUS_PATH}")


if __name__ == "__main__":
    main()
