"""Copy one tier's documents into a standalone corpus tree an importer can be pointed at.

The consuming importer is handed a corpus tree and a manifest, and it walks the
whole tree reading every document's id to find the ones the manifest names. Pointing
it at `generated_data/sources` would work but is the wrong move twice over: it reads
511,958 documents to import 1,144 of them, and — the part that actually breaks — the
full corpus carries **four dsid collisions**, pairs of files that share one document
id, and an importer that finds a wanted id twice cannot know which file the manifest
meant and stops. One of the four is a gold document, so it is in the bedrock, so it
is in *every* tier: pointed at the full corpus, every tier import fails.

Materializing resolves each collision once, here, by taking the path the corpus's own
uuid_index.json records for that id — the same choice a third party rebuilding from
the public index would make — and writing exactly one file per manifest line. The
output tree has one document per id by construction, which is the property the import
needs and the full corpus does not have.

The two organizational pages are copied to the output root, deliberately outside
`sources/`, since they are not corpus documents and the importer must not try to read
them as such. Landing them in the deployment is a separate step the tier's README covers.

Usage:
    python -m ladder.materialize --tier T0 --out /tmp/tier-T0
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from ladder.common import (
    ORGANIZATIONAL_PAGES,
    OUT,
    REPO,
    SOURCES_DIR,
    UUID_INDEX_PATH,
    file_digest,
    load_uuid_index,
)


def _tier_dir(draft: bool) -> Path:
    return OUT / ("tiers_draft" if draft else "tiers")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize one tier as a corpus tree."
    )
    parser.add_argument("--tier", required=True, choices=("T0", "T3", "T8", "T13"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="overwrite a non-empty output"
    )
    parser.add_argument(
        "--skip-input-check",
        action="store_true",
        help="copy even if the corpus has drifted from the manifest it was built against",
    )
    args = parser.parse_args()

    tier_dir = _tier_dir(args.draft)
    manifest_path = tier_dir / f"{args.tier}.manifest"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} does not exist; build the tiers first")

    ladder = json.loads((tier_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded = ladder["manifest_sha256"][args.tier]
    actual = file_digest(manifest_path)
    if actual != recorded:
        raise SystemExit(
            f"{manifest_path} does not match the checksum manifest.json records\n"
            f"  file is  {actual}\n  expected {recorded}"
        )

    if not args.skip_input_check:
        index_digest = file_digest(UUID_INDEX_PATH)
        if index_digest != ladder["inputs"]["uuid_index_sha256"]:
            raise SystemExit(
                "the corpus has changed since this ladder was built\n"
                f"  uuid_index.json is {index_digest}\n"
                f"  ladder expects     {ladder['inputs']['uuid_index_sha256']}\n"
                "rebuild the ladder, or pass --skip-input-check to copy anyway"
            )

    dsids = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = load_uuid_index()

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to overwrite")
    sources_out = out / "sources"

    missing: list[str] = []
    mismatched: list[str] = []
    by_source: Counter[str] = Counter()
    for dsid in dsids:
        relative = index.get(dsid)
        if relative is None:
            missing.append(dsid)
            continue
        source = SOURCES_DIR / relative
        if not source.is_file():
            missing.append(dsid)
            continue
        destination = sources_out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        # Checked after the copy, against the copy: the point is that the tree the
        # importer will read carries the id the manifest named, not that the corpus
        # did when the ladder was built.
        try:
            written = json.loads(destination.read_bytes()).get("dataset_doc_uuid")
        except (json.JSONDecodeError, UnicodeDecodeError):
            written = None
        if written != dsid:
            mismatched.append(relative)
        by_source[relative.split("/")[0]] += 1

    if missing:
        raise SystemExit(
            f"{len(missing)} manifest document(s) are not in the corpus, starting {missing[:5]}; "
            f"a short tier is worse than no tier"
        )
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} copied file(s) carry a different id than the manifest named, "
            f"starting {mismatched[:5]}"
        )

    written_ids = {dsid for dsid in dsids}
    files = sum(1 for _ in sources_out.rglob("*.json"))
    if files != len(written_ids):
        raise SystemExit(
            f"wrote {files} files for {len(written_ids)} ids; the tree is not one file per id"
        )

    for relative in ORGANIZATIONAL_PAGES:
        shutil.copy2(REPO / relative, out / Path(relative).name)
    shutil.copy2(manifest_path, out / manifest_path.name)
    shutil.copy2(
        tier_dir / f"{args.tier}.provenance.json", out / f"{args.tier}.provenance.json"
    )

    print(f"materialized {args.tier}: {files:,} documents -> {sources_out}")
    for source, count in by_source.most_common():
        print(f"  {source:<14}{count:>7,}")
    print(f"  organizational pages -> {out} (outside sources/, land them separately)")
    print(f"\nimport with:\n  import-tier {sources_out} {out / manifest_path.name}")


if __name__ == "__main__":
    main()
