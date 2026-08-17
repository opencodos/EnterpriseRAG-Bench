"""Build the corpus-subset manifest.

Selects 5,000 documents out of the full 511,958-document corpus and records the
reason each one is present. Selection is a pure function of (salt, dsid): every
layer is ranked by sha256(salt:dsid) rather than shuffled, so the choice does not
depend on iteration order, pool membership, or Python version. Raising a layer's
budget is additive for gold and distractors -- the previous selection stays a
prefix of the new one. Background is near-additive: filling the last few slots
exactly displaces a document or two (one, measured at 5,000 -> 6,000), because
whole seed groups that did not fit under the smaller budget do fit under the
larger one.

Layers, all seeded from the same undifferentiated pool before expansion so that
gold, distractor and background documents are structurally indistinguishable:

  gold        every expected_doc_id of all 600 candidate questions, not just the
              200 in the suite, so questions added later need no corpus regrow
  distractor  same-source look-alikes of gold docs (>=50% filename-slug overlap)
              that are not themselves gold; these are what keep constrained and
              conflicting questions from collapsing into basic lookups
  background  hash-sampled documents from the rest of the corpus
  neighbor    one sibling either side of every seed within its container, ordered
              by the container's natural key (date, timestamp, ticket or PR id)

Outputs manifest.json (parameters and digests), include.jsonl (one row per
retained document), exclude.jsonl (documents deliberately considered and cut) and
pool_remaining.jsonl (candidate questions not in the suite).
"""

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "subset"

SALT = "erb-5k-v1"
BUDGET = 5000
DISTRACTOR_BUDGET = 500
NEIGHBOR_K = 1
SLUG_OVERLAP = 0.5
# a slug token in more than this many filenames is boilerplate, not a topic
COMMON_TOKEN_CUTOFF = 4000


def rank(dsid: str) -> str:
    """Stable per-document sort key. Independent of pool and iteration order."""
    return hashlib.sha256(f"{SALT}:{dsid}".encode()).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slug_tokens(path: str) -> set[str]:
    """Topic words of a filename, with dates, timestamps and ids stripped."""
    name = os.path.basename(path)[: -len(".json")]
    name = re.sub(r"^(pr-)?\d[\d-]*-", "", name)
    name = re.sub(r"^[A-Z]+-\d+-", "", name)
    return {w for w in name.split("-") if len(w) > 2}


def container_key(path: str) -> tuple[int, int, str]:
    """Natural ordering within a container, so 'neighbour' means adjacent in time.

    Gmail and Fireflies encode a date, Jira/Linear a ticket number, GitHub a PR
    number, Slack an epoch timestamp. Anything else falls back to name order.
    """
    name = os.path.basename(path)
    for pattern in (r"^(\d{8})-", r"^pr-(\d+)-", r"^[A-Z]+-(\d+)-", r"^(\d+)-"):
        m = re.match(pattern, name)
        if m:
            return (0, int(m.group(1)), name)
    m = re.match(r"^(\d{4})-(\d\d)-(\d\d)-", name)
    if m:
        return (0, int("".join(m.groups())), name)
    return (1, 0, name)


def find_distractors(
    gold: set[str], index: dict[str, str], path_to_dsid: dict[str, str]
) -> dict[str, set[str]]:
    """Map each gold dsid to same-source documents that look like it but aren't gold."""
    paths = list(index.values())
    inverted: dict[str, list[int]] = collections.defaultdict(list)
    for i, path in enumerate(paths):
        for token in slug_tokens(path):
            inverted[token].append(i)

    out: dict[str, set[str]] = {}
    for dsid in gold:
        path = index[dsid]
        source = path.split("/")[0] + "/"
        tokens = slug_tokens(path)
        if not tokens:
            continue
        overlap: dict[int, int] = collections.Counter()
        for token in tokens:
            if len(inverted[token]) > COMMON_TOKEN_CUTOFF:
                continue
            for i in inverted[token]:
                overlap[i] += 1
        hits = set()
        for i, shared in overlap.items():
            other = paths[i]
            if other == path or not other.startswith(source):
                continue
            if shared / len(tokens) < SLUG_OVERLAP:
                continue
            other_dsid = path_to_dsid[other]
            if other_dsid not in gold:
                hits.add(other_dsid)
        if hits:
            out[dsid] = hits
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--distractors", type=int, default=DISTRACTOR_BUDGET)
    args = ap.parse_args()

    index: dict[str, str] = json.loads(
        (REPO / "generated_data" / "uuid_index.json").read_text()
    )
    path_to_dsid = {path: dsid for dsid, path in index.items()}

    train = read_jsonl(REPO / "splits" / "train.jsonl")
    test = read_jsonl(REPO / "splits" / "test.jsonl")
    candidates = read_jsonl(REPO / "questions.jsonl") + read_jsonl(
        REPO / "extra_questions.jsonl"
    )

    suite_gold = {d for r in train + test for d in (r["expected_doc_ids"] or [])}
    gold = {d for r in candidates for d in (r["expected_doc_ids"] or [])}
    missing = sorted(d for d in gold if d not in index)
    if missing:
        raise SystemExit(f"gold docs absent from uuid_index: {missing[:5]}")

    # containers and their natural ordering, used for neighbour expansion
    containers: dict[str, list[str]] = collections.defaultdict(list)
    for path in index.values():
        containers[os.path.dirname(path)].append(path)
    for paths in containers.values():
        paths.sort(key=container_key)
    position = {
        path: (parent, i)
        for parent, paths in containers.items()
        for i, path in enumerate(paths)
    }

    def expand(dsid: str) -> list[str]:
        """A seed plus its k nearest siblings, as dsids."""
        parent, i = position[index[dsid]]
        window = containers[parent][max(0, i - NEIGHBOR_K) : i + NEIGHBOR_K + 1]
        return [path_to_dsid[p] for p in window]

    # --- layer assembly -----------------------------------------------------
    layers: dict[str, set[str]] = collections.defaultdict(set)
    seed_of: dict[str, set[str]] = collections.defaultdict(set)
    selected: set[str] = set()

    def admit(dsid: str, layer: str, seed: str) -> None:
        layers[layer].add(dsid)
        seed_of[dsid].add(seed)
        selected.add(dsid)

    for dsid in gold:
        admit(dsid, "gold", dsid)

    distractor_map = find_distractors(gold, index, path_to_dsid)
    # documents shadowing a gold doc of the live suite come first; the rest are
    # kept as candidates so a later suite change can promote them.
    # a document can shadow several gold docs; attribute it to the lowest
    # (tier, dsid) so the recorded provenance does not depend on set iteration
    tiered: dict[str, tuple[int, str]] = {}
    for gold_dsid in sorted(distractor_map):
        tier = 0 if gold_dsid in suite_gold else 1
        for hit in distractor_map[gold_dsid]:
            if hit not in tiered or (tier, gold_dsid) < tiered[hit]:
                tiered[hit] = (tier, gold_dsid)
    ordered = sorted(tiered, key=lambda d: (tiered[d][0], rank(d)))
    chosen_distractors = ordered[: args.distractors]
    for dsid in chosen_distractors:
        admit(dsid, "distractor", tiered[dsid][1])

    for dsid in sorted(layers["gold"] | layers["distractor"]):
        for neighbor in expand(dsid):
            if neighbor not in selected:
                admit(neighbor, "neighbor", dsid)
            else:
                seed_of[neighbor].add(dsid)

    # background seeds are drawn in hash order and expanded by the same rule, so
    # "document sits alone" does not imply "not gold"
    pool = sorted((d for d in index if d not in selected), key=rank)
    background_seeds: list[str] = []
    for dsid in pool:
        if len(selected) >= args.budget:
            break
        group = [n for n in expand(dsid) if n not in selected]
        if len(selected) + len(group) > args.budget:
            continue
        background_seeds.append(dsid)
        admit(dsid, "background", dsid)
        for neighbor in group:
            if neighbor != dsid:
                admit(neighbor, "neighbor", dsid)

    # a final trickle, if no whole group fits in the remaining slots
    for dsid in pool:
        if len(selected) >= args.budget:
            break
        if dsid not in selected:
            admit(dsid, "background", dsid)
            background_seeds.append(dsid)

    # --- verification -------------------------------------------------------
    if len(selected) != args.budget:
        raise SystemExit(f"selected {len(selected)}, expected {args.budget}")
    unmet = sorted(d for d in gold if d not in selected)
    if unmet:
        raise SystemExit(f"gold documents missing from subset: {unmet[:5]}")

    # gold, distractor and background seeds must look alike structurally, or the
    # subset leaks which documents are answers
    def sibling_profile(seeds: set[str]) -> float:
        if not seeds:
            return 0.0
        return sum(
            sum(1 for n in expand(d) if n in selected and n != d) for d in seeds
        ) / len(seeds)

    profile = {
        "gold": round(sibling_profile(layers["gold"]), 3),
        "distractor": round(sibling_profile(layers["distractor"]), 3),
        "background": round(sibling_profile(set(background_seeds)), 3),
    }
    spread = max(profile.values()) - min(profile.values())
    if spread > 0.35:
        raise SystemExit(f"seed classes are structurally distinguishable: {profile}")

    # --- output -------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    rows = []
    for dsid in sorted(selected, key=lambda d: index[d]):
        path = index[dsid]
        rows.append(
            {
                "dsid": dsid,
                "path": path,
                "source": path.split("/")[0],
                "layers": sorted(l for l in layers if dsid in layers[l]),
                "seeds": sorted(seed_of[dsid]),
            }
        )
    with (OUT / "include.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with (OUT / "exclude.jsonl").open("w") as f:
        for dsid in ordered[args.distractors :]:
            f.write(
                json.dumps(
                    {
                        "dsid": dsid,
                        "path": index[dsid],
                        "reason": "distractor_candidate_over_budget",
                        "shadows": tiered[dsid][1],
                        "tier": tiered[dsid][0],
                    }
                )
                + "\n"
            )

    suite_ids = {r["question_id"] for r in train + test}
    original_ids = {r.get("original_question_id") for r in train + test}
    with (OUT / "pool_remaining.jsonl").open("w") as f:
        for row in candidates:
            if row["question_id"] in suite_ids or row["question_id"] in original_ids:
                continue
            f.write(json.dumps(row) + "\n")

    by_source = collections.Counter(r["source"] for r in rows)
    manifest = {
        "version": 1,
        "salt": SALT,
        "budget": args.budget,
        "parameters": {
            "distractor_budget": args.distractors,
            "neighbor_k": NEIGHBOR_K,
            "slug_overlap": SLUG_OVERLAP,
            "common_token_cutoff": COMMON_TOKEN_CUTOFF,
            "gold_seeded_from": "all 600 candidate questions",
        },
        "inputs": {
            "uuid_index_sha256": digest(REPO / "generated_data" / "uuid_index.json"),
            "train_sha256": digest(REPO / "splits" / "train.jsonl"),
            "test_sha256": digest(REPO / "splits" / "test.jsonl"),
            "corpus_documents": len(index),
        },
        "counts": {
            "included": len(selected),
            "gold": len(layers["gold"]),
            "gold_for_suite": len(suite_gold),
            "distractor": len(layers["distractor"]),
            "distractor_candidates": len(tiered),
            "background_seeds": len(background_seeds),
            "neighbor_only": len(
                layers["neighbor"]
                - layers["gold"]
                - layers["distractor"]
                - set(background_seeds)
            ),
            "by_source": dict(by_source.most_common()),
        },
        "checks": {"seed_sibling_profile": profile, "spread": round(spread, 3)},
        "complement": (
            "every dsid in uuid_index.json not listed in include.jsonl; "
            "reproducible from uuid_index_sha256 plus salt and parameters"
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest["counts"], indent=2))
    print("seed sibling profile:", profile)


if __name__ == "__main__":
    main()
