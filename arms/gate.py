"""The bedrock validity gate: does our ladder behave like the one it reproduces?

The programme's go/no-go. Both paper-faithful arms are run at T0 and scored by the
benchmark's own metrics evaluation, and their official combined scores are compared
against the values the scaling study published for the same two paradigms at the same
rung. If they land, the ladder construction, the reader configuration, the scorer and
the whole harness are validated at a fraction of the programme's cost. If they miss,
the ladder stops here rather than climbing on an anchor that does not hold.

**The acceptance rule is declared in this module and not derived from any result.**
Both arms must land within ``TOLERANCE_POINTS`` of their published value. A threshold
chosen after seeing the number is the first thing a skeptical reader attacks, so it is
a constant here, it was written into the plan before the harness existed, and the
later rungs reuse it unchanged.

Why 3.5 and not something tighter: the study's own cross-judge measurement moved
combined scores by -3.56 to +1.18 when its predictions were re-scored by an
independent judge under the same official protocol. We are not running its judge, so
a tighter rule would fail on judge choice while the ladder was perfectly faithful.

Bootstrap intervals over questions are reported beside the point estimates and are
deliberately *not* used to widen the rule. A wider interval is a reason to trust the
point estimate less, not a licence to accept a point estimate further from the
published value.

Usage:
    python -m arms.gate --tier-tree /data/tier-T0 \
        --arm bm25=/data/results/T0/bm25 \
        --arm agent=/data/results/T0/agent \
        --judge <model that scored both cells> \
        --session <scoring session id>
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arms.common import RETRIEVAL_GRANULARITY, Tier, load_tier
from arms.published import published_score
from arms.run import FAILED_ANSWER_TEXT, read_rows

# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

# How far a reproduced arm may land from its published value, in points of the
# official combined score. Set from the study's own cross-judge spread; see above.
TOLERANCE_POINTS = 3.5

# How many questions may have been cut off by the agent arm's wall-clock backstop
# rather than by the study's call budget, as a fraction of the suite.
#
# The clock is a backstop and the budget is the experiment. A question the clock cut
# off reports a call budget it never spent, at a point that moves with how loaded the
# box is -- a load-dependent number inside a curve that is read as a property of
# corpus size. A handful is an artifact of individual questions; more than this and
# the binding ceiling was the clock, so the arm did not run under the budget it names
# and its score is not the study's paradigm.
MAX_CLOCK_CUTOFF_FRACTION = 0.01

# The suite the published scores are over. A different denominator is not comparable
# to them, which is why a short run is a failure rather than a smaller sample.
EXPECTED_QUESTIONS = 500

# Resampling for the reported intervals. Seeded, so the interval a cell reports is a
# property of its answers and not of when the gate happened to run.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260825
CONFIDENCE = 0.95

# The study's shared settings, as they appear in an arm's run.json. Checked rather
# than trusted: the arms warn when these are overridden but still run, because the
# override exists for smoke tests -- and a smoke-test setting that reached a gate run
# would produce a score rather than an error.
#
# ``granularity`` is here for the same reason the others are, and it is the one that
# cost a measurement to learn: a chunk-level cell and a document-level cell are both
# well-formed, both report five retrieved units, and differ by twelve points of
# combined score, because chunk-level hands the reader one window of a gold document
# and calls the document retrieved.
REQUIRED_SETTINGS = {
    "bm25": {"top_k": 5, "granularity": RETRIEVAL_GRANULARITY},
    "agent": {"max_llm_calls": 80},
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def combined_score(row: dict[str, Any]) -> float:
    """One question's official combined score: completeness, gated by correctness.

    The metric the study publishes. Completeness counts only when the holistic
    correctness verdict passed, so a fluent answer that is wrong scores zero however
    many gold facts it happens to mention.

    Recomputed here from the per-question rows rather than read from the scorer's
    aggregate, so that the interval and the point estimate come from one array and
    the gate can check its own reading against what the scorer reported.
    """
    return float(row["completeness_pct"]) if row["answer_correct"] else 0.0


def bootstrap_interval(
    values: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """A percentile bootstrap interval for the mean of *values*, resampling questions.

    Questions are the unit of resampling because they are what varies between one
    suite and another: the 500 are a sample of the questions the benchmark could have
    asked, and two systems measured on them differ partly by which questions were
    drawn. Resampling answers or facts would describe a different uncertainty.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * resamples)]
    high = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return (low, high)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One thing the gate refuses to trust, and what it found."""

    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        return f"  [{'ok' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class ArmVerdict:
    """One reproduced arm at one rung, against the value the study published."""

    arm: str
    tier: str
    published: float
    measured: float
    interval: tuple[float, float]
    checks: list[Check] = field(default_factory=list)

    @property
    def band(self) -> tuple[float, float]:
        """The acceptance band, which is the published value plus or minus the rule."""
        return (self.published - TOLERANCE_POINTS, self.published + TOLERANCE_POINTS)

    @property
    def offset(self) -> float:
        """Measured minus published. The reproduction's own headline number.

        Reported whether or not the arm passes, and carried into the methodology
        note: a reader judging the reproduction wants the distance, not the verdict.
        """
        return self.measured - self.published

    @property
    def within_band(self) -> bool:
        return abs(self.offset) <= TOLERANCE_POINTS

    @property
    def passed(self) -> bool:
        return self.within_band and all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        low, high = self.band
        return {
            "arm": self.arm,
            "tier": self.tier,
            "published_combined_score": self.published,
            "measured_combined_score": round(self.measured, 2),
            "offset": round(self.offset, 2),
            "acceptance_band": [round(low, 2), round(high, 2)],
            "within_band": self.within_band,
            "bootstrap_ci": [round(self.interval[0], 2), round(self.interval[1], 2)],
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "confidence": CONFIDENCE,
                "seed": BOOTSTRAP_SEED,
                "unit": "question",
            },
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Reading a cell
# ---------------------------------------------------------------------------


def _load(path: Path, what: str) -> Any:
    if not path.is_file():
        raise SystemExit(f"{path} is missing; {what}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_arm(arm: str, cell: Path, tier: Tier) -> ArmVerdict:
    """Score one arm's cell against its published value, checking what produced it.

    Every check here guards a fault that produces a *number* rather than an error --
    a rung measured under a different manifest, a suite scored short, a correction
    pass that rewrote the gold set, a smoke-test setting that reached a real run. A
    gate that only compared two floats would pass all of them.
    """
    results = _load(cell / "results.json", "score this cell before gating it")
    identity = _load(cell / "run.json", "the cell cannot identify what produced it")
    answers = read_rows(cell / "answers.jsonl")

    checks: list[Check] = []

    # The rung. An arm measured against a different manifest is not on this ladder,
    # and nothing in its score would show it.
    recorded = identity.get("manifest_sha256")
    checks.append(
        Check(
            "tier identity",
            recorded == tier.manifest_sha256,
            f"run.json records {recorded}, tier is {tier.manifest_sha256}"
            if recorded != tier.manifest_sha256
            else f"measured against {tier.name} ({tier.manifest_sha256[:12]}...)",
        )
    )

    # The study's shared settings, which the arms warn about but do not enforce.
    for key, want in REQUIRED_SETTINGS.get(arm, {}).items():
        got = identity.get(key)
        checks.append(
            Check(
                f"setting {key}",
                got == want,
                f"{got} (the study's value)" if got == want else f"{got}, expected {want}",
            )
        )

    # The denominator, checked on both sides of the scorer. The published scores are
    # over 500 questions, and a short suite scores as a smaller sample rather than a
    # worse system.
    #
    # Counted rather than read off ``skipped_rows``, which the scorer reports as the
    # string "N/A" on a resumed session and as a count otherwise -- so it is not a
    # number to compare against zero. Counting both sides says more anyway: a row the
    # scorer skipped is one that reached it and did not come back, and it shows up
    # here as answered-but-not-scored whatever the header says.
    stats = results.get("aggregate_stats", {})
    rows = results.get("questions", [])
    scored_ids = {row["question_id"] for row in rows}
    answered_ids = {row["question_id"] for row in answers}
    unscored = sorted(answered_ids - scored_ids)
    sized = (
        len(answered_ids) == EXPECTED_QUESTIONS
        and len(scored_ids) == EXPECTED_QUESTIONS
        and not unscored
    )
    checks.append(
        Check(
            "suite size",
            sized,
            f"{len(answered_ids)} answered, {len(scored_ids)} scored"
            + (
                f" (scorer reports {stats.get('skipped_rows')} skipped)"
                if sized
                else f" -- expected {EXPECTED_QUESTIONS} of each"
                + (f", unscored starting {unscored[:3]}" if unscored else "")
            ),
        )
    )

    # --no-correction. The consensus correction flow rewrites gold answers, facts and
    # expected document ids when a system's documents differ from the gold set, so a
    # corrected question is one where the system helped choose what it was graded on.
    # Every arm has to be held to one gold set or the four are not comparable.
    corrected = [r["question_id"] for r in rows if r.get("corrected")]
    checks.append(
        Check(
            "no-correction honoured",
            not corrected and not stats.get("num_corrected_questions"),
            "no question was corrected"
            if not corrected
            else f"{len(corrected)} question(s) corrected, starting {corrected[:3]}",
        )
    )

    per_question = [combined_score(row) for row in rows]
    measured = sum(per_question) / len(per_question) if per_question else 0.0

    # The gate's own arithmetic against the scorer's. A mismatch means this module is
    # reading a field that is not the metric it believes it is -- the one fault that
    # would silently move every number the programme reports.
    reported = stats.get("combined_correctness_completeness_score")
    agrees = reported is not None and abs(measured - float(reported)) < 0.01
    checks.append(
        Check(
            "metric agrees with scorer",
            agrees,
            f"recomputed {measured:.2f} against reported {reported}",
        )
    )

    # Which ceiling bound the agent arm. Counted from the answers rather than from the
    # run's printed summary, because the answers are the durable artifact.
    if arm == "agent":
        cut_off = [r["question_id"] for r in answers if r.get("timed_out")]
        allowed = int(MAX_CLOCK_CUTOFF_FRACTION * EXPECTED_QUESTIONS)
        exhausted = sum(1 for r in answers if r.get("budget_exhausted"))
        checks.append(
            Check(
                "call budget was the binding ceiling",
                len(cut_off) <= allowed,
                f"{len(cut_off)} question(s) cut off by the wall clock "
                f"(at most {allowed} allowed), {exhausted} spent the call budget",
            )
        )

    # A sentinel answer is a question the arm did not answer. It scores zero rather
    # than leaving the denominator, which is what keeps the suite at 500 -- but a run
    # with many of them is a broken harness reported as a weak system.
    failed = sum(1 for r in answers if r.get("answer") == FAILED_ANSWER_TEXT)
    checks.append(
        Check(
            "questions answered",
            failed == 0,
            "every question answered"
            if failed == 0
            else f"{failed} question(s) scored zero as unanswered, not as wrong",
        )
    )

    return ArmVerdict(
        arm=arm,
        tier=tier.name,
        published=published_score(arm, tier.name),
        measured=measured,
        interval=bootstrap_interval(per_question),
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The bedrock validity gate: our two faithful arms against theirs."
    )
    parser.add_argument("--tier-tree", required=True, type=Path)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="A scored cell, e.g. bm25=/data/results/T0/bm25. Repeat for each arm.",
    )
    parser.add_argument(
        "--judge",
        required=True,
        help=(
            "The model that scored both cells. The official scorer does not record "
            "it, and a gate whose two arms were graded by different judges is "
            "measuring the judges."
        ),
    )
    parser.add_argument(
        "--session",
        required=True,
        help="An identifier for the one scoring session both cells were scored in.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Where to write gate.json")
    args = parser.parse_args()

    tier = load_tier(args.tier_tree)
    cells = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=DIR, got {spec!r}")
        name, _, directory = spec.partition("=")
        cells[name] = Path(directory)

    missing = set(REQUIRED_SETTINGS) - set(cells)
    if missing:
        raise SystemExit(
            f"the gate is both paper-faithful arms or neither; missing {sorted(missing)}"
        )

    # The rule, printed before any measured value, in the order it was decided.
    print(f"Bedrock validity gate -- {tier.name}, {tier.documents:,} documents")
    print(f"  manifest      {tier.manifest_sha256}")
    print(f"  judge         {args.judge}")
    print(f"  session       {args.session}")
    print(f"\nThe rule, declared before any score was seen:")
    print(f"  both arms within {TOLERANCE_POINTS} points of their published value")
    for arm in sorted(cells):
        published = published_score(arm, tier.name)
        print(
            f"    {arm:<6} published {published:>5.1f}  -> accept "
            f"{published - TOLERANCE_POINTS:.1f}-{published + TOLERANCE_POINTS:.1f}"
        )

    verdicts = [evaluate_arm(arm, cells[arm], tier) for arm in sorted(cells)]

    print("\nMeasured:")
    for verdict in verdicts:
        low, high = verdict.interval
        print(
            f"\n  {verdict.arm}: {verdict.measured:.2f}  "
            f"({'+' if verdict.offset >= 0 else ''}{verdict.offset:.2f} against "
            f"published {verdict.published:.1f})"
        )
        print(
            f"    {int(CONFIDENCE * 100)}% bootstrap CI over questions: "
            f"[{low:.2f}, {high:.2f}] -- reported, not used to widen the rule"
        )
        for check in verdict.checks:
            print(check.render())

    passed = all(verdict.passed for verdict in verdicts)
    payload = {
        "tier": tier.name,
        "tier_documents": tier.documents,
        "manifest_sha256": tier.manifest_sha256,
        "judge": args.judge,
        "scoring_session": args.session,
        "no_correction": True,
        "tolerance_points": TOLERANCE_POINTS,
        "arms": [verdict.as_dict() for verdict in verdicts],
        "passed": passed,
    }
    out = args.out or (args.tier_tree.parent / f"gate-{tier.name}.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'PASS' if passed else 'FAIL'} -- written to {out}")
    if not passed:
        # A miss stops the ladder rather than being worked around, including a miss
        # in the direction that flatters us: our bedrock scoring higher than theirs
        # most likely means the adversarial layer is thin.
        raise SystemExit(
            "the bedrock anchor did not hold; the ladder stops here. Log it as a "
            "deviation rather than adjusting the rule."
        )


if __name__ == "__main__":
    main()
