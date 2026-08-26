"""Index one tier's documents as chunks, for the study's BM25 arm.

This is a second BM25 index, not a change to the repository's own. The shipped
``src.scripts.answer_generation.index_document_bm25`` indexes whole documents into one
``text`` field and the shipped runner retrieves whole documents -- and the ladder build
depends on exactly that behaviour, since ``ladder.pool`` mines its trap and lure
candidates from that index's document-level top-200. Re-pointing it at chunks would
change what phase 5 already committed. So the study's chunked retrieval gets its own
index, its own name and its own runner, and the document-level pair is left alone.

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

from arms.common import CHUNK_OVERLAP, CHUNK_SIZE, Tier, document_chunks, load_tier

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


def index_name_for(tier: Tier) -> str:
    """One index per rung, named after it."""
    return f"erb-chunks-{tier.name.lower()}"


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
    parser.add_argument("--index-name", default=None, help="Defaults to erb-chunks-<tier>")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild an index that already exists",
    )
    args = parser.parse_args()

    started = time.time()
    tier = load_tier(args.tier_tree)
    index_name = args.index_name or index_name_for(tier)
    print(
        f"{tier.name}: {len(tier.dsids):,} documents "
        f"({tier.documents:,} with the organizational pages) -> {index_name}"
    )
    print(f"  chunking at {CHUNK_SIZE}/{CHUNK_OVERLAP} tokens")

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
    for dsid in tqdm(tier.dsids, desc="Chunking", leave=False):
        path = tier.sources / index[dsid]
        try:
            doc = json.loads(path.read_bytes())
            chunks = document_chunks(doc)
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

    # A document the manifest names but this release cannot read would silently
    # shrink the search space, so it stops the build the way a missing one does.
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} manifest document(s) could not be chunked, starting "
            f"{unreadable[:3]}; the index would cover less than the tier"
        )

    print(f"  {len(actions):,} chunks to index")
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
            f"indexed {indexed:,} of {len(actions):,} chunks with {errors} error(s); "
            f"the index does not cover the tier"
        )

    print(
        f"\nDone. {tier.name}: {indexed:,} chunks from {len(tier.dsids):,} documents "
        f"in {time.time() - started:.1f}s -> {index_name}"
    )


if __name__ == "__main__":
    main()
