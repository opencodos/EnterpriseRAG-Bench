"""What both arms do the same way: the question set, resume, and the run's own report.

Kept together rather than beside either arm because the two have to agree on all of it.
A rung is four cells scored in one session against one gold set, so an arm that resumed
differently, or counted a failure differently, would put a number in that table that
does not mean what its neighbour means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arms.common import SCAFFOLD_PAGES, Tier
from ladder.common import QUESTIONS_PATH, load_questions

# The same sentinel the Aethos-side runner writes, and for the same reason: the
# official metrics evaluation *skips* a row carrying neither an answer nor document
# ids, so an unanswered question written as an empty row would leave the denominator
# and 480 answered questions would be scored as the whole suite. A failure is a wrong
# answer -- zero correctness, zero completeness -- with the real error beside it.
FAILED_ANSWER_TEXT = "[no answer: the system under test did not answer this question]"


def load_core_questions(limit: int | None = None) -> list[dict[str, Any]]:
    """The benchmark's 500 core questions, in file order.

    ``extra_questions.jsonl`` is deliberately not read: the published scores the ladder
    is anchored against are over these 500, and a different denominator would not be
    comparable to them.
    """
    questions = load_questions(QUESTIONS_PATH)
    return questions[:limit] if limit is not None else questions


def write_row(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row, flushed, so an interrupted run keeps what it finished."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every well-formed row of a JSONL output file, in file order."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A row half-written when the process died. Dropping it is what makes
            # resume possible at all; it is re-asked below.
            continue
        if isinstance(row, dict) and row.get("question_id"):
            rows.append(row)
    return rows


def _nonblank_lines(path: Path) -> int:
    """How many lines a JSONL file holds, whether or not this reader can parse them."""
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if line.strip())


def _rewrite(path: Path, rows: list[dict[str, Any]], keep: set[str]) -> None:
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            qid = row["question_id"]
            if qid not in keep or qid in seen:
                continue
            seen.add(qid)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reconcile_outputs(*paths: Path) -> set[str]:
    """Trim every output file back to the questions all of them hold, and return those.

    The BM25 arm writes an answer *and* a context per question, and an interruption
    between the two writes leaves the pair uneven. Neither direction is safe to ignore:
    a context with no answer would be re-asked and written twice, and the control arm
    refuses a file naming a question more than once; an answer with no context would
    leave that file short of the question set, which the control arm also refuses. So
    both are trimmed to their intersection here, before anything is asked, and each
    file is rewritten free of duplicates.

    A file holding a line this reader had to drop is rewritten too, even when what it
    can read is already exactly the complete set. A row half-written when the process
    died leaves bytes with no trailing newline, and the next append would land on that
    same line -- fusing the two into one unparseable row and taking a good answer down
    with the broken one, silently and for the rest of the run.

    Returns the question ids that are complete in every file.
    """
    contents = {path: read_rows(path) for path in paths}
    complete: set[str] | None = None
    for rows in contents.values():
        ids = {row["question_id"] for row in rows}
        complete = ids if complete is None else complete & ids
    complete = complete or set()

    for path, rows in contents.items():
        unreadable = _nonblank_lines(path) != len(rows)
        duplicated = len({row["question_id"] for row in rows}) != len(rows)
        if unreadable or duplicated or len(rows) != len(complete):
            _rewrite(path, rows, complete)
    return complete


IDENTITY_FILENAME = "run.json"


def bind_run_identity(out_dir: Path, identity: dict[str, Any]) -> None:
    """Pin an output directory to the configuration that wrote it, once.

    Resume keys off the question id alone, so without this an output directory will
    happily absorb rows from two different configurations: a ``--top-k 1`` run
    followed by a ``--top-k 5`` one over the same directory makes no calls at all and
    reports the first run's answers under the second run's settings. Nothing
    downstream would show it -- the file is the right length and every row is
    well-formed -- which is the shape of fault this module exists to refuse.

    So the settings that shape an answer are written beside the answers on the first
    run and compared on every later one. A directory whose identity differs is a
    different cell and gets its own directory.

    Raises:
        SystemExit: if *out_dir* already holds answers from another configuration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / IDENTITY_FILENAME
    if not path.is_file():
        path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        return

    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded == identity:
        return
    differing = sorted(
        key
        for key in set(recorded) | set(identity)
        if recorded.get(key) != identity.get(key)
    )
    detail = "\n".join(
        f"  {key}: this run {identity.get(key)!r}, {path.name} {recorded.get(key)!r}"
        for key in differing
    )
    raise SystemExit(
        f"{out_dir} already holds a run under a different configuration:\n{detail}\n"
        f"Resuming would report those answers under this run's settings. Point "
        f"--out-dir at a new directory."
    )


def report_run(
    answers_path: Path,
    questions: list[dict[str, Any]],
    *,
    arm: str,
    tier: Tier,
    parallelism: int = 1,
    extra: dict[str, Any] | None = None,
) -> None:
    """Print the run's own account of itself, and refuse to call a short run complete.

    A cell is 500 questions answered or failed against one manifest. Every way this run
    can fall short of that -- a missing row, a sentinel answer, a document id the tier
    does not contain -- produces a plausible-looking file rather than an error, so each
    is counted here and a short run exits non-zero.
    """
    rows = read_rows(answers_path)
    answered = {row["question_id"] for row in rows}
    expected = {question["question_id"] for question in questions}
    missing = sorted(expected - answered)
    failed = [row for row in rows if row.get("answer") == FAILED_ANSWER_TEXT]

    # The two organizational pages are inside the tier the study describes but outside
    # the manifest, which is frozen as corpus document ids; a retrieved scaffold is
    # therefore in the rung even though no manifest line names it.
    tier_dsids = set(tier.dsids) | set(SCAFFOLD_PAGES)
    foreign = sorted(
        {
            dsid
            for row in rows
            for dsid in (row.get("document_ids") or [])
            if dsid not in tier_dsids
        }
    )

    summary = {
        "tier": tier.name,
        "arm": arm,
        "manifest_sha256": tier.manifest_sha256,
        "tier_documents": tier.documents,
        "questions": len(expected),
        "answered": len(answered),
        "failed": len(failed),
        "missing": len(missing),
        "parallelism": parallelism,
        "answers_file": str(answers_path),
        **(extra or {}),
    }
    print("\n" + json.dumps(summary, indent=2))

    if foreign:
        raise SystemExit(
            f"{len(foreign)} document id(s) in {answers_path} are not in {tier.name}, "
            f"starting {foreign[:3]}; the arm searched outside the rung it reports"
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} question(s) have no row in {answers_path}, starting "
            f"{missing[:3]}; re-run to resume. A short file scores as a shorter suite, "
            f"not as a worse one."
        )


def preflight_reader(model: str | None = None, *, tools: bool = False) -> None:
    """Check the reader is fit to run before a multi-hour run starts.

    Both checks fail into a *result* rather than an error if left unmade -- a template
    ignoring the thinking flag puts reasoning text in front of the judge, and a server
    that cannot emit tool calls leaves the agent talking to itself for its whole budget.
    Neither is visible in the output file, so both are one call here instead.

    Only meaningful for the vLLM-served reader; a run pointed at another provider skips
    it rather than failing, since both checks are about how that server was started.

    Args:
        tools: Also probe tool calling. Set by the agent arm, which is tool-driven;
            the BM25 arm makes one tool-less call per question and does not need it.
    """
    from src.llm.factory import LLM_PROVIDER

    if LLM_PROVIDER.lower() != "vllm":
        print(f"[warn] LLM_PROVIDER is '{LLM_PROVIDER}', not the study's vLLM-served reader")
        return

    from src.llm.vllm_llm import probe_thinking_disabled, probe_tool_calls

    probe_thinking_disabled(model)
    checks = "reachable, thinking disabled"
    if tools:
        probe_tool_calls(model)
        checks += ", tool calls served"
    print(f"  reader preflight: {checks}")
