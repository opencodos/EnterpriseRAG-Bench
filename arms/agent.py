"""Arm 2: the study's File-System Agent over one tier, read by the study's reader.

The agent itself is the repository's own -- its tools, its prompt, its shell allowlist,
its truncation and compaction rules all come from
``src.scripts.answer_generation.agent_retrieval`` unchanged. What this module adds is
the three things a ladder rung needs and that runner did not have: a corpus root that
is one tier rather than the whole corpus, the study's 80-LLM-call budget per question,
and a wall clock loose enough that the budget is what actually binds.

That last one is not a detail. The shipped runner bounds a question at 600 seconds, which
is a *second* ceiling and not the study's -- an arm cut off by it reports an 80-call
budget it never spent, at a point that moves with how loaded the box is. Measured at T0:
a mean of 37 calls at roughly 4.3s each, so 80 calls is about six minutes of that pacing
and the shipped clock would not always have bound; but per-call latency rises with
conversation length, and that margin is thin enough to bind sometimes and silently. The
run's report counts both endings separately so the difference is visible rather than
assumed.

Usage:
    python -m arms.agent --tier-tree /data/tier-T0 --out-dir results/T0/agent
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from arms.common import MAX_LLM_CALLS, Tier, load_tier
from arms.run import (
    FAILED_ANSWER_TEXT,
    bind_run_identity,
    load_core_questions,
    preflight_reader,
    read_rows,
    reconcile_outputs,
    report_run,
    write_row,
)
from src.llm.factory import get_llm
from src.scripts.answer_generation.agent_retrieval import (
    build_system_prompt,
    build_tools,
    check_available_commands,
    run_agent_for_question,
)
from src.tools.tool_implementations.document_read import DocumentReadTool

# Loose enough that 80 calls can complete against a locally-served reader. It is a
# backstop against a wedged question, not the arm's budget -- see the module docstring.
DEFAULT_QUESTION_TIMEOUT_SECONDS = 3600.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="File-System Agent over one tier of the ladder."
    )
    parser.add_argument("--tier-tree", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=MAX_LLM_CALLS,
        help=f"The study's per-question call budget (default: {MAX_LLM_CALLS})",
    )
    parser.add_argument(
        "--question-timeout",
        type=float,
        default=DEFAULT_QUESTION_TIMEOUT_SECONDS,
        help="Wall-clock backstop per question; keep it well above the call budget",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help=(
            "Questions in flight. Above 1 the recorded latency is a queueing time "
            "rather than the arm's, so hold it at whatever every other tier used."
        ),
    )
    parser.add_argument("--reasoning-level", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test escape hatch")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if args.max_llm_calls < 1:
        # Checked here as well as in the loop: without it the ValueError surfaces
        # inside every question's worker and the run turns 500 questions into 500
        # identical failures instead of stopping on the first line.
        raise SystemExit(
            f"--max-llm-calls must be at least 1, got {args.max_llm_calls}"
        )
    if args.max_llm_calls != MAX_LLM_CALLS:
        print(
            f"[warn] call budget {args.max_llm_calls} is not the study's {MAX_LLM_CALLS}; "
            f"this run is not comparable to the published curve"
        )

    tier: Tier = load_tier(args.tier_tree)
    # The tier's uuid index is what the select_doc tool validates against, so an agent
    # cannot name a document outside the rung it is exploring.
    uuid_index = tier.uuid_index()
    questions = load_core_questions(limit=args.limit)

    missing = check_available_commands()
    if missing:
        print(
            f"[warn] {len(missing)} shell command(s) not on this box: "
            f"{', '.join(sorted(missing))}. The agent is told only about the rest, so "
            f"a tier measured here is not comparable to one measured where they exist."
        )

    # The prompt states the search space, and the agent's search space is sources/ --
    # the tier's manifest lines, not its document count, which also counts the two
    # organizational pages that sit outside the tree the agent walks.
    system_prompt = build_system_prompt(corpus_size=len(tier.dsids))
    tools = build_tools(
        DocumentReadTool(
            base_dir=str(tier.sources),
            generated_doc_contents=True,
            include_dsid=True,
        )
    )

    # Before anything is read back: resume keys off the question id alone, so an
    # output directory has to be pinned to the settings that wrote it or a second run
    # under different ones would report the first run's answers as its own.
    bind_run_identity(
        args.out_dir,
        {
            "arm": "agent",
            "tier": tier.name,
            "manifest_sha256": tier.manifest_sha256,
            "max_llm_calls": args.max_llm_calls,
            "question_timeout": args.question_timeout,
            "model": args.model,
            "reasoning_level": args.reasoning_level,
            "parallelism": args.parallelism,
        },
    )
    answers_path = args.out_dir / "answers.jsonl"
    done = reconcile_outputs(answers_path)

    if not args.skip_preflight:
        preflight_reader(args.model, tools=True)

    pending = [q for q in questions if q["question_id"] not in done]
    print(
        f"{tier.name} / agent: {len(questions)} question(s), {len(done)} already "
        f"answered, {len(pending)} pending -> {args.out_dir}"
    )
    print(
        f"  budget {args.max_llm_calls} call(s)/question, wall-clock backstop "
        f"{args.question_timeout:.0f}s, root {tier.sources}"
    )
    if not pending:
        report_run(
            answers_path,
            questions,
            arm="agent",
            tier=tier,
            parallelism=args.parallelism,
            extra={"max_llm_calls": args.max_llm_calls},
        )
        return

    quiet = args.parallelism > 1
    lock = threading.Lock()

    def process(question: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            result = run_agent_for_question(
                question_id=question["question_id"],
                question=question["question"],
                llm=get_llm(
                    tools=tools,
                    quiet=quiet,
                    reasoning_level=args.reasoning_level,
                    model=args.model,
                ),
                system_prompt=system_prompt,
                uuid_index=uuid_index,
                quiet=quiet,
                model=args.model,
                reasoning_level=args.reasoning_level,
                corpus_root=str(tier.sources),
                max_llm_calls=args.max_llm_calls,
                timeout_seconds=args.question_timeout,
            )
            failure = None
        except Exception as exc:  # noqa: BLE001 -- the row records it either way
            result = {"answer": "", "document_ids": []}
            failure = str(exc)

        # An agent that ran out of road without answering is a wrong answer, never an
        # empty row: the official metrics evaluation skips a row carrying neither an
        # answer nor document ids, so an empty one would take the question out of the
        # denominator and the hardest questions would quietly stop counting. The
        # shipped runner writes the empty row; this arm does not.
        answer = result.get("answer") or ""
        if not answer.strip() and not result.get("document_ids"):
            answer = FAILED_ANSWER_TEXT

        with lock:
            write_row(
                answers_path,
                {
                    "question_id": question["question_id"],
                    "answer": answer,
                    "document_ids": result.get("document_ids") or [],
                    "latency_seconds": time.perf_counter() - started,
                    "llm_calls": result.get("llm_calls"),
                    "budget_exhausted": result.get("budget_exhausted", False),
                    "timed_out": result.get("timed_out", False),
                    "failure": failure,
                },
            )

    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {pool.submit(process, q): q["question_id"] for q in pending}
        with tqdm(total=len(pending), desc="Questions") as bar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((futures[future], exc))
                bar.update(1)

    for qid, exc in failures:
        print(f"  {qid} did not produce a row: {exc}")

    parsed = read_rows(answers_path)
    report_run(
        answers_path,
        questions,
        arm="agent",
        tier=tier,
        parallelism=args.parallelism,
        extra={
            "max_llm_calls": args.max_llm_calls,
            # Which ceiling actually bound each question. The two are exclusive, and
            # if the second is not near zero the arm was cut off by the clock -- so
            # the call budget it claims to run under is not the one it ran under.
            "budget_exhausted": sum(1 for row in parsed if row.get("budget_exhausted")),
            "cut_off_by_clock": sum(1 for row in parsed if row.get("timed_out")),
            "mean_llm_calls": (
                round(
                    sum(row.get("llm_calls") or 0 for row in parsed) / len(parsed), 1
                )
                if parsed
                else 0
            ),
        },
    )


if __name__ == "__main__":
    main()
