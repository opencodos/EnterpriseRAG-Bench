"""What a published cell is, and the checks it must pass before it may be charted.

The ladder's results are sixteen cells -- four arms across four tiers -- and the store
is a bucket prefix whose path encodes the cell, so a listing is the tracking. This
module fixes what a cell *is* on disk, so that the four arms, which are produced by
three different runners on three different hosts, publish one shape rather than three.

A cell is a directory holding four files:

    answers.jsonl      the arm's answers -- the durable artifact
    results.json       the official scorer's output over them
    run.json           what the runner recorded about its own run
    provenance.json    what no runner could know; this module's subject

**Answers are durable and scores are derived.** A rubric or reference change costs
sixteen re-scorings and no re-asking, which is the property that makes a rung
re-judgeable years later without a deployment. So ``answers.jsonl`` is written once
and never rewritten, and ``results.json`` may be replaced by a later scoring session.

**One fact, one home.** ``provenance.json`` deliberately does *not* restate the tier
manifest checksum: ``run.json`` records it and :mod:`arms.gate` checks it there. A
checksum written in two files is a checksum that can disagree with itself, and the
failure would be silent -- both files well-formed, the cell measured against neither.
The same reasoning keeps the arm's own settings (``top_k``, ``granularity``,
``max_llm_calls``, ``reader_thinking``) in ``run.json`` where its runner wrote them.

What lands here is exactly what falls outside any single run: which scoring session
graded it, which judge, which revisions of the runner and the scorer produced it, what
the reader actually was, which host timed it, and -- for the product arm -- which build
of the system under test answered.

**Why a host, when the plan calls latency a non-goal across arms.** Precisely because
of that: the arms are timed on different machines by necessity, so a latency figure is
only meaningful beside the host that produced it. Recording the host is what lets the
final report show four latencies without implying they are one curve.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arms.common import Tier, load_tier
from arms.run import FAILED_ANSWER_TEXT, read_rows

# The suite every cell is measured over. A cell holding a different number of rows is
# not comparable to the published values or to its siblings, and a short suite scores
# *higher* when the questions it lost were the hard ones.
EXPECTED_QUESTIONS = 500

# The four arms, as they appear in a cell's path. The two reproduced arms keep the
# names their runners already use, so a cell directory is addressable by the same
# string the gate takes.
ARMS = ("bm25", "agent", "bm25-aethos-reader", "aethos")

# The arm whose answers come from the system under test rather than from a harness.
# Only this one carries a build sha, because only this one *has* a build: the other
# three are a lexical index, a shell loop, and a single completion call.
PRODUCT_ARM = "aethos"

CELL_FILES = ("answers.jsonl", "results.json", "run.json", "provenance.json")


@dataclass(frozen=True)
class Provenance:
    """The facts about a cell that no runner writes, and every chart depends on.

    Every field is required. A cell that cannot say which judge graded it, or which
    build answered it, is not a measurement -- it is a number whose meaning was lost,
    and the only honest thing to do with it is refuse to chart it.
    """

    tier: str
    arm: str

    # Scoring. Both are arguments to the official scorer and appear nowhere in its
    # output, so a cell that does not record them cannot be shown to share a session
    # with its siblings -- which is the one property that makes sixteen cells one
    # measurement rather than sixteen.
    scoring_session: str
    judge_model: str
    scorer_revision: str

    # What produced the answers. A revision rather than a name, because the harnesses
    # are under active development and "the BM25 arm" meant something different three
    # weeks ago.
    runner_revision: str

    # The reader. ``reader_model`` is the served model id; ``reader_artifact_digest``
    # pins *which weights* answered, since a model id is a moving target on any hub.
    # The reproduced arms carry the study's Qwen reader here and the two Aethos-reader
    # arms carry the deployment's own chat tier, so the column is comparable across
    # all four without a special case.
    reader_model: str
    reader_artifact_digest: str

    # Which machine timed it. Latency is per arm and per host by construction.
    measured_on: str

    # The system under test, for the product arm only; empty for the other three.
    aethos_build_sha: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def write_provenance(cell: Path, provenance: Provenance) -> Path:
    """Write a cell's ``provenance.json``, refusing to fill a gap with a default.

    Written at publish time rather than at run time, in one shot: a file filled in two
    passes spends the interval looking complete while carrying a field nothing set.
    """

    missing = [key for key, value in asdict(provenance).items() if value == "" and key != "aethos_build_sha"]
    if missing:
        raise ValueError(f"provenance for {cell} is missing {', '.join(sorted(missing))}")
    if provenance.arm == PRODUCT_ARM and not provenance.aethos_build_sha:
        raise ValueError(f"the {PRODUCT_ARM} arm's provenance must name the build that answered it")
    if provenance.arm != PRODUCT_ARM and provenance.aethos_build_sha:
        raise ValueError(f"only the {PRODUCT_ARM} arm has a build sha; {provenance.arm} carries none")
    cell.mkdir(parents=True, exist_ok=True)
    path = cell / "provenance.json"
    path.write_text(provenance.to_json(), encoding="utf-8")
    return path


@dataclass
class CellCheck:
    name: str
    passed: bool
    detail: str


def check_cell(cell: Path, tier: Tier, *, pinned_sha: str, session: str, judge: str) -> list[CellCheck]:
    """The four properties a cell is checked on rather than trusted for.

    Each guards a fault that yields a *number* instead of an error: a rung measured
    against another manifest, a product arm answered by a build that has since moved,
    a suite scored short, or a cell graded in its own session and charted beside ones
    that were not.
    """

    checks: list[CellCheck] = []

    for name in CELL_FILES:
        if not (cell / name).is_file():
            checks.append(CellCheck(f"file {name}", False, f"{cell / name} is missing"))
    if checks:
        return checks

    identity = json.loads((cell / "run.json").read_text(encoding="utf-8"))
    provenance = json.loads((cell / "provenance.json").read_text(encoding="utf-8"))
    answers = read_rows(cell / "answers.jsonl")

    # 1. The rung. Read from run.json, where the runner wrote it -- not restated in
    #    provenance, so the two cannot disagree.
    recorded = identity.get("manifest_sha256")
    checks.append(
        CellCheck(
            "tier identity",
            recorded == tier.manifest_sha256,
            f"measured against {tier.name} ({tier.manifest_sha256[:12]}...)"
            if recorded == tier.manifest_sha256
            else f"run.json records {recorded}, tier is {tier.manifest_sha256}",
        )
    )

    # 2. The system under test, for the one arm that has one.
    if provenance.get("arm") == PRODUCT_ARM:
        got = provenance.get("aethos_build_sha")
        checks.append(
            CellCheck(
                "pinned build",
                got == pinned_sha,
                f"answered by the pinned build ({pinned_sha[:12]}...)"
                if got == pinned_sha
                else f"provenance records {got}, programme pinned {pinned_sha}",
            )
        )

    # 3. The denominator. Answered-or-failed, not answered: a failure is written with
    #    the sentinel and scores as a wrong answer, so it stays in the denominator.
    failed = sum(1 for row in answers if row.get("answer") == FAILED_ANSWER_TEXT)
    checks.append(
        CellCheck(
            "question count",
            len(answers) == EXPECTED_QUESTIONS,
            f"{len(answers)} rows ({failed} failed, scored as zero)"
            if len(answers) == EXPECTED_QUESTIONS
            else f"{len(answers)} rows, expected {EXPECTED_QUESTIONS}",
        )
    )

    # 4. One session, one judge, across all sixteen.
    for key, want in (("scoring_session", session), ("judge_model", judge)):
        got = provenance.get(key)
        checks.append(
            CellCheck(
                key.replace("_", " "),
                got == want,
                f"{got}" if got == want else f"provenance records {got}, ladder is on {want}",
            )
        )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check published cells before charting them.")
    parser.add_argument("--tier-tree", required=True, type=Path)
    parser.add_argument("--cell", action="append", required=True, help="arm=path, repeatable")
    parser.add_argument("--pinned-sha", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--judge", required=True)
    args = parser.parse_args()

    tier = load_tier(args.tier_tree)
    ok = True
    for spec in args.cell:
        arm, _, path = spec.partition("=")
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
        print(f"\n{arm}  {path}")
        for check in check_cell(
            Path(path), tier, pinned_sha=args.pinned_sha, session=args.session, judge=args.judge
        ):
            print(f"  {'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}")
            ok = ok and check.passed
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
