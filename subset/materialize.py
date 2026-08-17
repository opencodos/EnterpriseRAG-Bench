"""Copy the documents named by the manifest into a standalone corpus folder.

Reads subset/manifest.json and subset/include.jsonl, verifies that the corpus it
is copying from is the one the manifest was built against, then reproduces the
selected documents under their original paths. The result is a self-contained
drop-in replacement for generated_data/sources with its own uuid_index.json.

    python subset/materialize.py --out corpus_5k

Every copied file is checked against the manifest: the dsid recorded inside the
document must match the dsid that selected it. A mismatch means the manifest and
the corpus have drifted apart and the copy stops.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "subset"
SOURCES = REPO / "generated_data" / "sources"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpus_5k", help="output directory")
    ap.add_argument(
        "--force", action="store_true", help="overwrite a non-empty output directory"
    )
    ap.add_argument(
        "--skip-input-check",
        action="store_true",
        help="copy even if uuid_index.json no longer matches the manifest",
    )
    args = ap.parse_args()

    manifest = json.loads((MANIFEST_DIR / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (MANIFEST_DIR / "include.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != manifest["counts"]["included"]:
        raise SystemExit(
            f"include.jsonl has {len(rows)} rows, manifest says "
            f"{manifest['counts']['included']}"
        )

    index_path = REPO / "generated_data" / "uuid_index.json"
    if not args.skip_input_check:
        actual = digest(index_path)
        expected = manifest["inputs"]["uuid_index_sha256"]
        if actual != expected:
            raise SystemExit(
                "corpus has changed since the manifest was built\n"
                f"  uuid_index.json is {actual}\n"
                f"  manifest expects  {expected}\n"
                "rebuild the manifest, or pass --skip-input-check to copy anyway"
            )

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to overwrite")
    dest_sources = out / "sources"

    copied = 0
    missing: list[str] = []
    mismatched: list[str] = []
    unverified = 0
    for row in rows:
        src = SOURCES / row["path"]
        if not src.is_file():
            missing.append(row["path"])
            continue
        dst = dest_sources / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        try:
            recorded = json.loads(dst.read_text()).get("dataset_doc_uuid")
        except (json.JSONDecodeError, UnicodeDecodeError):
            recorded = None
        if recorded is None:
            unverified += 1
        elif recorded != row["dsid"]:
            mismatched.append(row["path"])
        copied += 1

    if missing:
        raise SystemExit(
            f"{len(missing)} manifest documents are absent from {SOURCES}: "
            f"{missing[:5]}"
        )
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} documents carry a different dsid than the manifest "
            f"records: {mismatched[:5]}"
        )

    (out / "uuid_index.json").write_text(
        json.dumps({row["dsid"]: row["path"] for row in rows}, indent=None) + "\n"
    )
    shutil.copy2(MANIFEST_DIR / "manifest.json", out / "manifest.json")
    shutil.copy2(MANIFEST_DIR / "include.jsonl", out / "include.jsonl")

    questions_dir = out / "questions"
    questions_dir.mkdir(exist_ok=True)
    for name in ("train.jsonl", "test.jsonl"):
        shutil.copy2(REPO / "splits" / name, questions_dir / name)

    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1

    print(f"copied {copied} documents -> {out}")
    if unverified:
        print(f"  {unverified} documents carried no dataset_doc_uuid to check against")
    for source, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source:14}{n:>6}")


if __name__ == "__main__":
    main()
