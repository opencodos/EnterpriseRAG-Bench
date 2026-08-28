"""Index one tier for the study's BM25 arm, by document or by chunk.

This is a second BM25 index, not a change to the repository's own. The ladder build
depends on the shipped ``src.scripts.answer_generation.index_document_bm25`` exactly as
it is, since ``ladder.pool`` mines its trap and lure candidates from that index's
top-200 over the *full corpus*. This one is per-tier and is left free to vary.

**Granularity is a recorded setting, not a detail.** ``document`` is the study's, and
the default; ``chunk`` is kept because the first T0 measurement was taken with it and a
number is only interpretable next to the unit that produced it. The two live in
separate indices -- ``erb-docs-<tier>`` and ``erb-chunks-<tier>`` -- so a runner cannot
silently read one while reporting the other, and the runner records which it used.

Each tier is its own index. The search space is the experiment's independent variable,
so an arm that searched a wider index than the rung it reports would be measuring
nothing, and one index per tier makes that a property of the name rather than of a
filter every query has to remember to apply.

Usage:
    python -m arms.index_bm25 --tier-tree /data/tier-T0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opensearchpy import OpenSearch, helpers as os_helpers
from tqdm import tqdm

from arms.common import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GRANULARITIES,
    RETRIEVAL_GRANULARITY,
    Tier,
    document_chunks,
    load_tier,
    scaffold_chunks,
    whole_document,
)

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"

# The shipped indexer's settings, so the two differ in the unit indexed and in
# nothing else. A different analyzer would make the arms incomparable for a reason
# that has nothing to do with chunking.
INDEX_SETTINGS = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "analysis": {"analyzer": {"default": {"type": "standard"}}},
    },
    "mappings": {
        "properties": {
            "dataset_doc_uuid": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_total": {"type": "integer"},
            "title": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
        }
    },
}


def index_name_for(tier: Tier, granularity: str = RETRIEVAL_GRANULARITY) -> str:
    """One index per rung and unit, named after both.

    The unit is in the name because it is the thing most easily lost: two indices over
    the same rung that differ only in what a hit *is* would otherwise be
    interchangeable to every caller, and a cell measured against the wrong one would
    report a plausible number.
    """
    stem = "docs" if granularity == "document" else "chunks"
    return f"erb-{stem}-{tier.name.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index one tier's documents as chunks for the BM25 arm."
    )
    parser.add_argument(
        "--tier-tree",
        required=True,
        type=Path,
        help="A directory written by `python -m ladder.materialize`",
    )
    parser.add_argument("--opensearch-url", default=DEFAULT_OPENSEARCH_URL)
    parser.add_argument(
        "--granularity",
        default=RETRIEVAL_GRANULARITY,
        choices=GRANULARITIES,
        help=(
            f"What a hit is: a whole document or a windowed chunk "
            f"(default: {RETRIEVAL_GRANULARITY}, the study's)"
        ),
    )
    parser.add_argument(
        "--index-name", default=None, help="Defaults to erb-{docs,chunks}-<tier>"
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild an index that already exists",
    )
    args = parser.parse_args()

    started = time.time()
    tier = load_tier(args.tier_tree)
    index_name = args.index_name or index_name_for(tier, args.granularity)
    print(
        f"{tier.name}: {len(tier.dsids):,} documents "
        f"({tier.documents:,} with the organizational pages) -> {index_name}"
    )
    if args.granularity == "chunk":
        print(f"  indexing chunks at {CHUNK_SIZE}/{CHUNK_OVERLAP} tokens")
    else:
        print("  indexing whole documents, unwindowed")
    if args.granularity != RETRIEVAL_GRANULARITY:
        print(
            f"[warn] granularity {args.granularity!r} is not the study's "
            f"{RETRIEVAL_GRANULARITY!r}; this index is not comparable to the "
            f"published curve"
        )

    index = tier.uuid_index()
    client = OpenSearch(hosts=[args.opensearch_url], use_ssl=False, verify_certs=False)

    if client.indices.exists(index=index_name):
        if not args.recreate:
            raise SystemExit(
                f"index '{index_name}' already exists; pass --recreate to rebuild it. "
                f"Appending to it would leave a rung measured against a mixture of two "
                f"builds, which nothing downstream would show."
            )
        print(f"  dropping existing index '{index_name}'")
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=INDEX_SETTINGS)

    actions: list[dict[str, object]] = []
    unreadable: list[str] = []
    unit = "Chunking" if args.granularity == "chunk" else "Reading"
    for dsid in tqdm(tier.dsids, desc=unit, leave=False):
        path = tier.sources / index[dsid]
        try:
            doc = json.loads(path.read_bytes())
            chunks = (
                document_chunks(doc)
                if args.granularity == "chunk"
                else [whole_document(doc)]
            )
        except Exception:  # noqa: BLE001 -- any unreadable document is the same answer
            unreadable.append(dsid)
            continue
        for chunk in chunks:
            actions.append(
                {
                    "_index": index_name,
                    "_id": chunk.point_id,
                    "_source": {
                        "dataset_doc_uuid": chunk.dsid,
                        "chunk_index": chunk.index,
                        "chunk_total": chunk.total,
                        "title": chunk.title,
                        "text": chunk.text,
                    },
                }
            )

    # The bedrock's two organizational pages, which the manifest cannot name because
    # neither is a corpus document. The study counts them inside the tier and ten of
    # its questions have no other evidence, so an index without them searches 1,142
    # documents while reporting 1,144 -- and those ten score zero for a reason no
    # results file records.
    for chunk in scaffold_chunks(tier, args.granularity):
        actions.append(
            {
                "_index": index_name,
                "_id": chunk.point_id,
                "_source": {
                    "dataset_doc_uuid": chunk.dsid,
                    "chunk_index": chunk.index,
                    "chunk_total": chunk.total,
                    "title": chunk.title,
                    "text": chunk.text,
                },
            }
        )

    # A document the manifest names but this release cannot read would silently
    # shrink the search space, so it stops the build the way a missing one does.
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} manifest document(s) could not be read, starting "
            f"{unreadable[:3]}; the index would cover less than the tier"
        )

    print(f"  {len(actions):,} unit(s) to index")
    indexed = 0
    errors = 0
    for start in tqdm(range(0, len(actions), args.batch_size), desc="Batches"):
        success, failures = os_helpers.bulk(
            client, actions[start : start + args.batch_size], raise_on_error=False
        )
        indexed += success
        if failures:
            errors += len(failures)
            for failure in failures[:3]:
                print(f"  bulk error: {failure}")
    client.indices.refresh(index=index_name)

    if errors or indexed != len(actions):
        raise SystemExit(
            f"indexed {indexed:,} of {len(actions):,} unit(s) with {errors} error(s); "
            f"the index does not cover the tier"
        )

    print(
        f"\nDone. {tier.name}: {indexed:,} {args.granularity} unit(s) from "
        f"{tier.documents:,} documents "
        f"in {time.time() - started:.1f}s -> {index_name}"
    )


if __name__ == "__main__":
    main()
