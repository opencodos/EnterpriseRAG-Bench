"""Build the nested tier manifests: the bedrock, the order over everything else, four rungs.

The study's ladder is 28 strictly nested rungs growing by about 1.25x, sharing one
fixed bedrock of relevant and adversarial documents while only the background
grows. This builds four of them -- its tiers 0, 3, 8 and 13, the rungs where both
paper-faithful arms carry a published score -- at the published sizes.

Two properties make a manifest checkable by someone who does not trust us.

**Every tier is a prefix of the next.** Documents are emitted in one order:
the bedrock first, then a single seeded order over the non-bedrock corpus, cut at
each tier's size. Strict nesting is then a `head -n` comparison rather than a claim,
and a tier's checksum is a `sha256sum` of the file's bytes.

**The order is a pure function of (salt, dsid).** Every document is ranked by
sha256(salt:dsid) within its stratum rather than shuffled, so a rebuild does not
depend on iteration order, pool membership or Python version. The strata are the
two dimensions the study stratifies on, source and noise, and the strata are merged
by normalized rank -- a document at position r of a stratum of N sorts at
(r + 0.5) / N -- so that every prefix of the order holds each stratum in its global
proportion, which is what "the source and noise distribution of each added prefix
approximates the global background corpus" has to mean operationally.

Two departures from the study are forced and recorded in each tier's provenance.

The **organizational pages travel beside the manifest, not in it**. The bedrock
carries the corpus's company overview and initiative index, but neither is a corpus
document: both sit outside sources/, carry no dataset_doc_uuid, and appear in no
uuid_index.json. A manifest holds document ids and nothing else, so the two pages
are named in organizational_pages.txt and a tier's document count is its manifest's
line count plus two.

**Tier sizes are the published ones, and the mined layer fills what it fills.** The
study's filter yielded 326 traps and 99 lures; ours is a different judge and will
not land on the same count. Holding the size fixed and letting the background take
up the slack keeps every published quantity that the ladder is validated against --
tier size, gold share, the mean-token curve -- comparable, where holding the mined
count fixed would move all of them. Gold and lures are never dropped; if the mined
layer overflows, traps are kept round-robin across questions -- every question's best
before any question's second -- and what that leaves out is recorded.

Usage:
    python -m ladder.build_manifests [--salt SALT] [--draft]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, NamedTuple

from ladder.census import (
    CENSUS_PATH,
    STATS_PATH,
    PUBLISHED_BEDROCK,
    PUBLISHED_TIERS,
    token_length,
)
from ladder.common import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ORGANIZATIONAL_PAGES,
    OUT,
    REPO,
    UUID_INDEX_PATH,
    chunk_count,
    file_digest,
    gold_dsids,
    load_questions,
    load_uuid_index,
    rank_key,
    read_jsonl,
)
from ladder.mine import LURES_PATH, TRAPS_PATH

SALT = "erb-ladder-v1"

# The rungs this ladder builds, at the study's own sizes. Sizes count every
# document in the tier, the two organizational pages included.
TIERS = {tier: row["documents"] for tier, row in PUBLISHED_TIERS.items()}

# How far a tier's mean document length may sit from the published curve, rescaled
# to our tokenizer, before the build refuses. Wide enough to absorb a different
# tokenizer and a differently-sized mined layer; tight enough that a bedrock missing
# its adversarial half does not pass -- a gold-only ladder comes in 6.8% low at T0,
# which is exactly the kind of build this is here to refuse.
MEAN_TOKEN_TOLERANCE_PCT = 5.0

# How far a tier's source mix may sit from the background corpus's own, in
# percentage points of the largest single source, before the build refuses.
STRATUM_TOLERANCE_PCT = 1.0


class Layer(NamedTuple):
    """One component of the bedrock, with what a cut needs to know to drop it fairly."""

    dsid: str
    layer: str
    strength: float
    question_id: str


def _load_stats() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The corpus keyed by document id, and the ids more than one file claims.

    The corpus carries four dsid collisions -- pairs of files sharing one document
    id. They are kept in the ladder rather than excluded, because they are real
    corpus documents and one of them is gold, so excluding it would put a hole in
    the question set. What they cost is that a tier cannot be imported straight out
    of the full corpus tree, where a wanted id resolves to two files and the import
    stops; ladder.materialize resolves each to the single path uuid_index.json
    records. The collisions are recorded in manifest.json so that this is a known
    property of the ladder rather than a surprise at import time.
    """
    if not STATS_PATH.exists():
        raise SystemExit(
            f"{STATS_PATH} is missing; run `python -m ladder.census` first"
        )
    rows = read_jsonl(STATS_PATH)
    index = load_uuid_index()
    paths: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        paths[row["dsid"]].append(row["path"])
    collisions = [
        {"dsid": dsid, "paths": sorted(claimed), "resolves_to": index.get(dsid)}
        for dsid, claimed in sorted(paths.items())
        if len(claimed) > 1
    ]
    # Keyed by the path uuid_index.json records, so a colliding id carries the same
    # statistics the public index would attribute to it.
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        preferred = index.get(row["dsid"])
        if row["dsid"] not in stats or row["path"] == preferred:
            stats[row["dsid"]] = row
    return stats, collisions


def _load_layer(path: Path, layer: str) -> list[Layer]:
    if not path.exists():
        return []
    return [
        Layer(row["dsid"], layer, row.get("dense_score") or 0.0, row["question_id"])
        for row in read_jsonl(path)
    ]


def build_bedrock(
    stats: dict[str, dict[str, Any]], draft: bool
) -> tuple[list[Layer], dict[str, Any]]:
    """Gold, traps and lures as one deduplicated set, gold winning every collision."""
    gold = gold_dsids(load_questions())
    traps = _load_layer(TRAPS_PATH, "trap")
    lures = _load_layer(LURES_PATH, "lure")

    if not traps and not lures and not draft:
        raise SystemExit(
            f"{TRAPS_PATH} and {LURES_PATH} are missing; run `python -m ladder.mine` first, "
            f"or pass --draft to build a gold-only ladder into tiers_draft/"
        )

    # Gold first, so a document mined as a trap for one question but annotated gold
    # for another is credited as gold -- the collisions the study reports as
    # cross-category duplicates.
    seen: dict[str, Layer] = {}
    for entry in (
        [Layer(dsid, "gold", 1.0, "") for dsid in sorted(gold)] + traps + lures
    ):
        if entry.dsid not in stats:
            continue
        seen.setdefault(entry.dsid, entry)

    counts = collections.Counter(entry.layer for entry in seen.values())
    composition = {
        "gold": counts["gold"],
        "trap": counts["trap"],
        "lure": counts["lure"],
        "organizational_pages": len(ORGANIZATIONAL_PAGES),
        "documents": len(seen) + len(ORGANIZATIONAL_PAGES),
        "cross_category_duplicates": len(gold) + len(traps) + len(lures) - len(seen),
        "unresolved_in_corpus": len(
            [entry for entry in traps + lures if entry.dsid not in stats]
        ),
        "published": PUBLISHED_BEDROCK,
    }
    return list(seen.values()), composition


def fit_bedrock(bedrock: list[Layer], capacity: int) -> tuple[list[str], list[str]]:
    """Order the bedrock, and cut its weakest traps if it overflows the smallest tier.

    Ordered by rank rather than by layer so that gold, traps and lures are
    interleaved: a document's position in a manifest must not reveal what it is.

    Only traps are ever cut. Gold is the question set's own annotation and lures are
    already capped at five per unanswerable question by the recipe itself -- there
    are about a hundred of them and each one is load-bearing for a question that
    would otherwise be solved by retrieving nothing. Traps are the layer that
    overshoots, because how freely a judge calls a near-miss a contradiction is the
    one part of the procedure the study's prose does not pin down, so they are also
    the layer that can absorb a cut.

    Traps are kept **round by round across questions**, not by score across the whole
    set: every question's best trap is taken before any question's second. Cutting on
    a global score instead would let the questions whose candidates happen to score
    high keep a handful each while others keep none -- and how high a question's
    candidates score is a property of its wording and its source family, so a global
    cut would quietly concentrate the adversarial layer on one kind of question.
    Round-robin spreads it the way per-question mining does, and lands near the
    study's own distribution of roughly two traps for every three questions.
    """
    ordered = sorted(bedrock, key=lambda entry: rank_key(SALT, entry.dsid))
    overflow = len(ordered) - capacity
    if overflow <= 0:
        return [entry.dsid for entry in ordered], []

    traps = [entry for entry in ordered if entry.layer == "trap"]
    if len(traps) < overflow:
        raise SystemExit(
            f"bedrock overflows the smallest tier by {overflow} documents but holds only "
            f"{len(traps)} traps to cut; gold and lures are not cuttable, so the ladder "
            f"cannot be built at the published sizes from this mining run"
        )
    dropped = {entry.dsid for entry in _traps_to_cut(traps, len(traps) - overflow)}
    return [entry.dsid for entry in ordered if entry.dsid not in dropped], sorted(
        dropped
    )


def _traps_to_cut(traps: list[Layer], keep: int) -> list[Layer]:
    """The traps that do not survive, keeping `keep` of them round-robin by question."""
    by_question: dict[str, list[Layer]] = collections.defaultdict(list)
    for entry in traps:
        by_question[entry.question_id].append(entry)
    for entries in by_question.values():
        entries.sort(key=lambda entry: (-entry.strength, rank_key(SALT, entry.dsid)))

    survivors: list[Layer] = []
    deepest = max(len(entries) for entries in by_question.values())
    for depth in range(deepest):
        this_round = [
            entries[depth] for entries in by_question.values() if depth < len(entries)
        ]
        # Within a round the strongest go first, so a cut landing mid-round keeps the
        # better half of it rather than whichever question happened to sort first.
        this_round.sort(key=lambda entry: (-entry.strength, rank_key(SALT, entry.dsid)))
        survivors.extend(this_round)
    return survivors[keep:]


def background_order(
    stats: dict[str, dict[str, Any]], bedrock: set[str], salt: str
) -> list[str]:
    """One seeded order over the non-bedrock corpus, stratified by source and noise.

    Each stratum is ranked independently, then the strata are merged on normalized
    rank, so any prefix carries every stratum in its global proportion. That is the
    property that makes a tier a scaled-down background rather than a sample of
    whichever source happened to hash low.
    """
    strata: dict[tuple[str, bool], list[str]] = collections.defaultdict(list)
    for dsid, row in stats.items():
        if dsid not in bedrock:
            strata[(row["source"], row["noise"])].append(dsid)

    merged: list[tuple[float, str, str]] = []
    for members in strata.values():
        members.sort(key=lambda dsid: rank_key(salt, dsid))
        size = len(members)
        for position, dsid in enumerate(members):
            merged.append(((position + 0.5) / size, rank_key(salt, dsid), dsid))
    merged.sort()
    return [dsid for _, _, dsid in merged]


def _organizational_page_tokens(tokenizer: str) -> tuple[int, int]:
    tokens = 0
    chunks = 0
    for relative in ORGANIZATIONAL_PAGES:
        text = (REPO / relative).read_text(encoding="utf-8")
        count = token_length(text, tokenizer)
        tokens += count
        chunks += chunk_count(count)
    return tokens, chunks


def tier_report(
    dsids: list[str],
    stats: dict[str, dict[str, Any]],
    bedrock: set[str],
    tier: str,
    tokenizer: str,
) -> dict[str, Any]:
    page_tokens, page_chunks = _organizational_page_tokens(tokenizer)
    tokens = sum(stats[dsid]["tokens"] for dsid in dsids) + page_tokens
    chunks = sum(stats[dsid]["chunks"] for dsid in dsids) + page_chunks
    documents = len(dsids) + len(ORGANIZATIONAL_PAGES)

    background = [dsid for dsid in dsids if dsid not in bedrock]
    published = PUBLISHED_TIERS[tier]
    return {
        "documents": documents,
        "manifest_lines": len(dsids),
        "organizational_pages": len(ORGANIZATIONAL_PAGES),
        "bedrock_documents": len(dsids) - len(background),
        "background_documents": len(background),
        "corpus_tokens": tokens,
        "chunks": chunks,
        "mean_doc_tokens": round(tokens / documents, 1),
        "by_source": dict(
            collections.Counter(stats[dsid]["source"] for dsid in dsids).most_common()
        ),
        "noise_documents": sum(1 for dsid in dsids if stats[dsid]["noise"]),
        "background_by_source": dict(
            collections.Counter(
                stats[dsid]["source"] for dsid in background
            ).most_common()
        ),
        "published": published,
    }


def validate(
    reports: dict[str, dict[str, Any]],
    manifests: dict[str, list[str]],
    stats: dict[str, dict[str, Any]],
    bedrock: set[str],
    census: dict[str, Any],
    enforce_token_curve: bool = True,
) -> dict[str, Any]:
    """Everything the build refuses to write a manifest without.

    Nesting and size are exact. The token curve and the source mix are compared
    against rescaled published ratios and against the background corpus's own mix,
    both with a stated tolerance, because neither can be an equality: the study's
    tokenizer is unrecoverable and our mined layer is a different size than theirs.

    The token curve is measured but not enforced for a draft, whose whole point is
    to exercise the ordering without a mined layer -- and which therefore fails it.
    """
    failures: list[str] = []
    ordered = sorted(TIERS, key=lambda tier: TIERS[tier])

    for smaller, larger in zip(ordered, ordered[1:]):
        prefix = manifests[larger][: len(manifests[smaller])]
        if prefix != manifests[smaller]:
            failures.append(f"{smaller} is not a line-exact prefix of {larger}")

    for tier in ordered:
        if reports[tier]["documents"] != TIERS[tier]:
            failures.append(
                f"{tier} holds {reports[tier]['documents']} documents, not {TIERS[tier]}"
            )

    full_mean = census["full_corpus"]["measured"]["mean_doc_tokens"]
    published_full = census["full_corpus"]["published"]["mean_doc_tokens"]
    token_curve = {}
    for tier in ordered:
        expected = full_mean * (
            PUBLISHED_TIERS[tier]["mean_doc_tokens"] / published_full
        )
        measured = reports[tier]["mean_doc_tokens"]
        delta = 100 * (measured - expected) / expected
        token_curve[tier] = {
            "expected": round(expected, 1),
            "measured": measured,
            "delta_pct": round(delta, 2),
        }
        if enforce_token_curve and abs(delta) > MEAN_TOKEN_TOLERANCE_PCT:
            failures.append(
                f"{tier} mean document tokens {measured} is {delta:+.2f}% from the published curve"
            )

    background_total = len(stats) - len(bedrock)
    global_mix = collections.Counter(
        row["source"] for dsid, row in stats.items() if dsid not in bedrock
    )
    stratification = {}
    for tier in ordered:
        drawn = reports[tier]["background_documents"]
        if drawn < 100:
            stratification[tier] = {"background_documents": drawn, "checked": False}
            continue
        worst = 0.0
        for source, count in global_mix.items():
            expected_share = 100 * count / background_total
            actual_share = (
                100 * reports[tier]["background_by_source"].get(source, 0) / drawn
            )
            worst = max(worst, abs(actual_share - expected_share))
        stratification[tier] = {
            "background_documents": drawn,
            "checked": True,
            "max_source_deviation_pct": round(worst, 3),
        }
        if worst > STRATUM_TOLERANCE_PCT:
            failures.append(
                f"{tier} background source mix deviates by {worst:.2f} points from the corpus"
            )

    gold = gold_dsids(load_questions())
    for tier in ordered:
        present = len(gold & set(manifests[tier]))
        if present != len(gold & set(stats)):
            failures.append(
                f"{tier} carries {present} gold documents, not all {len(gold & set(stats))}"
            )

    return {
        "nesting": "verified",
        "token_curve": token_curve,
        "stratification": stratification,
        "failures": failures,
    }


def write_outputs(
    out_dir: Path,
    manifests: dict[str, list[str]],
    reports: dict[str, dict[str, Any]],
    composition: dict[str, Any],
    dropped: list[str],
    collisions: list[dict[str, Any]],
    checks: dict[str, Any],
    salt: str,
    census: dict[str, Any],
    draft: bool,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for tier, dsids in manifests.items():
        path = out_dir / f"{tier}.manifest"
        path.write_text("\n".join(dsids) + "\n", encoding="utf-8")
        checksums[tier] = file_digest(path)
        provenance = {
            "tier": tier,
            "draft": draft,
            "salt": salt,
            "manifest_sha256": checksums[tier],
            "tokenizer": census["tokenizer"],
            "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
            "organizational_pages": list(ORGANIZATIONAL_PAGES),
            **reports[tier],
        }
        (out_dir / f"{tier}.provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

    (out_dir / "organizational_pages.txt").write_text(
        "\n".join(ORGANIZATIONAL_PAGES) + "\n", encoding="utf-8"
    )

    manifest = {
        "version": 1,
        "draft": draft,
        "salt": salt,
        "tiers": TIERS,
        "ordering": {
            "rank": "sha256(salt:dsid)",
            "strata": ["source", "noise"],
            "merge": "normalized rank (position + 0.5) / stratum size, ties broken by rank",
        },
        "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "tokenizer": census["tokenizer"],
        "inputs": {
            "uuid_index_sha256": file_digest(UUID_INDEX_PATH),
            "questions_sha256": file_digest(REPO / "questions.jsonl"),
            "corpus_documents": census["full_corpus"]["measured"]["documents"],
        },
        "bedrock": composition,
        "bedrock_dropped_for_capacity": dropped,
        "dsid_collisions": {
            "count": len(collisions),
            "note": (
                "ids more than one corpus file claims; a tier must be imported from a "
                "materialized tree (ladder.materialize), not the full corpus, where a "
                "wanted id resolves twice and the import stops"
            ),
            "collisions": collisions,
        },
        "manifest_sha256": checksums,
        "checks": checks,
        "organizational_pages": {
            "files": list(ORGANIZATIONAL_PAGES),
            "sha256": {
                relative: file_digest(REPO / relative)
                for relative in ORGANIZATIONAL_PAGES
            },
            "why_outside_the_manifest": (
                "neither is a corpus document: both sit outside sources/, carry no "
                "dataset_doc_uuid and appear in no uuid_index.json, so a manifest of "
                "document ids cannot name them"
            ),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the nested tier manifests.")
    parser.add_argument("--salt", default=SALT)
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Build without the mined layer, into tiers_draft/",
    )
    args = parser.parse_args()

    if not CENSUS_PATH.exists():
        raise SystemExit(
            f"{CENSUS_PATH} is missing; run `python -m ladder.census` first"
        )
    census = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    stats, collisions = _load_stats()

    bedrock_layers, composition = build_bedrock(stats, args.draft)
    smallest = min(TIERS.values())
    bedrock_ids, dropped = fit_bedrock(
        bedrock_layers, smallest - len(ORGANIZATIONAL_PAGES)
    )
    bedrock = set(bedrock_ids)

    # What was mined and what the tiers actually carry are different numbers once a
    # cut has happened, and the second is the one a reader of a manifest needs.
    retained_counts = collections.Counter(
        entry.layer for entry in bedrock_layers if entry.dsid in bedrock
    )
    composition["mined"] = {
        "gold": composition.pop("gold"),
        "trap": composition.pop("trap"),
        "lure": composition.pop("lure"),
        "documents": composition.pop("documents"),
    }
    composition["retained"] = {
        "gold": retained_counts["gold"],
        "trap": retained_counts["trap"],
        "lure": retained_counts["lure"],
        "organizational_pages": len(ORGANIZATIONAL_PAGES),
        "documents": len(bedrock_ids) + len(ORGANIZATIONAL_PAGES),
        "questions_keeping_a_trap": len(
            {
                entry.question_id
                for entry in bedrock_layers
                if entry.layer == "trap" and entry.dsid in bedrock
            }
        ),
    }

    mined, kept = composition["mined"], composition["retained"]
    if dropped:
        print(
            f"  bedrock overflowed {smallest}: cut {len(dropped)} traps, round-robin by question"
        )
    print(
        f"  mined:    {mined['gold']} gold + {mined['trap']} traps + {mined['lure']} lures"
    )
    print(
        f"  retained: {kept['gold']} gold + {kept['trap']} traps + {kept['lure']} lures "
        f"+ {kept['organizational_pages']} pages = {kept['documents']} documents "
        f"(study: {PUBLISHED_BEDROCK['gold']} + {PUBLISHED_BEDROCK['traps']} + "
        f"{PUBLISHED_BEDROCK['lures']} + {PUBLISHED_BEDROCK['organizational_pages']} "
        f"= {PUBLISHED_BEDROCK['documents']})"
    )
    print(f"  traps spread over {kept['questions_keeping_a_trap']} question(s)")

    order = background_order(stats, bedrock, args.salt)
    print(
        f"  ordered {len(order):,} background documents over {len({(r['source'], r['noise']) for r in stats.values()})} strata"
    )

    manifests: dict[str, list[str]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for tier, size in sorted(TIERS.items(), key=lambda item: item[1]):
        lines = size - len(ORGANIZATIONAL_PAGES)
        if lines < len(bedrock_ids):
            raise SystemExit(
                f"{tier} holds {lines} manifest lines but the bedrock needs {len(bedrock_ids)}"
            )
        manifests[tier] = bedrock_ids + order[: lines - len(bedrock_ids)]
        reports[tier] = tier_report(
            manifests[tier], stats, bedrock, tier, census["tokenizer"]
        )

    checks = validate(
        reports, manifests, stats, bedrock, census, enforce_token_curve=not args.draft
    )
    out_dir = OUT / ("tiers_draft" if args.draft else "tiers")
    checksums = write_outputs(
        out_dir,
        manifests,
        reports,
        composition,
        dropped,
        collisions,
        checks,
        args.salt,
        census,
        args.draft,
    )

    print(
        f"\n  {'tier':<5} {'docs':>7} {'lines':>7} {'tokens':>13} {'chunks':>9} {'mean':>8} {'vs curve':>10}"
    )
    for tier in sorted(TIERS, key=lambda name: TIERS[name]):
        report = reports[tier]
        delta = checks["token_curve"][tier]["delta_pct"]
        print(
            f"  {tier:<5} {report['documents']:>7,} {report['manifest_lines']:>7,} "
            f"{report['corpus_tokens']:>13,} {report['chunks']:>9,} {report['mean_doc_tokens']:>8.1f} {delta:>9.2f}%"
        )
    print("\n  stratification (max source deviation from the background corpus):")
    for tier, result in checks["stratification"].items():
        if result["checked"]:
            print(
                f"    {tier:<5} {result['max_source_deviation_pct']:>6.3f} points over {result['background_documents']:,} drawn"
            )
        else:
            print(
                f"    {tier:<5} not checked ({result['background_documents']} background documents)"
            )

    print("\n  checksums:")
    for tier, checksum in checksums.items():
        print(f"    {tier:<5} {checksum}")

    if checks["failures"]:
        print("\n  FAILED:")
        for failure in checks["failures"]:
            print(f"    - {failure}")
        raise SystemExit(1)
    print(f"\n  wrote {out_dir}{' (DRAFT: no mined layer)' if args.draft else ''}")


if __name__ == "__main__":
    main()
