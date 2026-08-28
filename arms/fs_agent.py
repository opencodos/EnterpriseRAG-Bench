"""One question, answered by the study's raw File-System Agent.

The study's agent is three read-only tools over the corpus tree and nothing else
(Appendix C.2, Table 9). The repository's shipped agent is a different and strictly
more capable system -- a ``run`` tool executing arbitrary shell with pipes, regex and
command chaining, a document reader, and an explicit ``select_doc_by_dsid`` -- under a
system prompt of its own that coaches search strategy at length. An arm built on that
one measures a paradigm the study never ran, so this module builds the study's.

The conversation loop itself is reused unchanged from ``src.llm.auto_conversation``:
the call budget, the wall-clock backstop, the forced finish and the accounting of which
ceiling bound are the same machinery both arms have always used. Only the tools, the
system prompt and where the document ids come from are the study's rather than the
repository's.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from arms.common import SCAFFOLD_PAGES
from arms.fs_tools import make_tools
from arms.paper_prompts import AGENT_SYSTEM_PROMPT
from src.llm.auto_conversation import run_agent_conversation
from src.llm.factory import get_llm
from src.llm.interface import Message, ReasoningLevel

# Injected when the wall-clock backstop is close, and again to force a text answer.
# The budget is the study's 80 calls; this is the backstop's message, not the budget's.
OUT_OF_TIME_MESSAGE = (
    "You are out of time. Answer the question now from what you have already read, "
    "and cite the relative file paths you used. If you cannot answer, say so."
)


def run_fs_agent_for_question(
    *,
    question_id: str,
    question: str,
    sources: Path,
    uuid_index: dict[str, str],
    scaffolds: dict[str, Path] | None = None,
    max_llm_calls: int,
    timeout_seconds: float,
    model: str | None = None,
    reasoning_level: ReasoningLevel = "medium",
    quiet: bool = False,
) -> dict[str, Any]:
    """Answer one question with the study's three-tool agent over *sources*.

    Args:
        uuid_index: The tier's id-to-path map, used in reverse to turn the paths the
            agent read into benchmark document ids.

    Returns:
        The row fields the arm records: answer, document_ids, llm_calls,
        budget_exhausted, timed_out.
    """
    read_paths: set[str] = set()
    schemas, executors = make_tools(sources, read_paths, scaffolds)

    messages: list[Message] = [
        Message(role="system", content=AGENT_SYSTEM_PROMPT),
        Message(role="user", content=question),
    ]

    llm = get_llm(
        tools=schemas, quiet=quiet, reasoning_level=reasoning_level, model=model
    )
    # Tool-less, so a question cut off by the clock still produces text rather than
    # another tool call it has no time to run.
    force_finish_llm = get_llm(
        tools=None, quiet=True, reasoning_level=reasoning_level, model=model
    )

    result = run_agent_conversation(
        llm=llm,
        executors=executors,
        messages=messages,
        timeout_seconds=timeout_seconds,
        max_llm_calls=max_llm_calls,
        shutdown_warning_seconds=30,
        shutdown_message=OUT_OF_TIME_MESSAGE,
        force_finish_llm=force_finish_llm,
        force_finish_message=OUT_OF_TIME_MESSAGE,
        parallel_tool_execution=False,
        tool_timeout=120,
        quiet=quiet,
    )

    answer = ""
    for message in reversed(messages):
        if message.role == "assistant" and message.content:
            answer = message.content
            break

    # The study's agent has no "select these documents" tool -- the shipped one does --
    # so what it *read* is the only record of what it retrieved. Paths are mapped back
    # through the tier's own index, so a path outside the rung resolves to nothing
    # rather than to a document the arm never searched.
    by_path = {path: dsid for dsid, path in uuid_index.items()}
    # The two organizational pages carry no corpus id, so they map back through
    # SCAFFOLD_PAGES rather than through the tier's index.
    by_path.update({filename: dsid for dsid, filename in SCAFFOLD_PAGES.items()})
    document_ids = sorted({by_path[p] for p in read_paths if p in by_path})

    return {
        "question_id": question_id,
        "answer": answer,
        "document_ids": document_ids,
        "llm_calls": result.llm_calls,
        "budget_exhausted": result.budget_exhausted,
        "timed_out": result.timed_out,
        "documents_read": len(read_paths),
    }


def preflight_tools(
    sources: Path,
    uuid_index: dict[str, str],
    scaffolds: dict[str, Path] | None = None,
) -> None:
    """Check the three tools work over this tier before a multi-hour run starts.

    Every one of these fails into a *result* rather than an error if left unmade: an
    agent whose ``grep`` matches nothing explores blindly and answers from nothing, and
    a ``read_doc`` whose paths do not map back to ids yields an arm reporting perfect
    answers standing on no documents.
    """
    read_paths: set[str] = set()
    _, executors = make_tools(sources, read_paths, scaffolds)

    listing = executors["list_dir"](".")
    folders = [line for line in listing.splitlines() if not line.startswith("...")]
    if not folders:
        raise SystemExit(f"list_dir found no source folders under {sources}")

    hits = [
        line
        for line in executors["grep"]("the").splitlines()
        if not line.startswith("...") and line != "No matches."
    ]
    if not hits:
        raise SystemExit(f"grep matched nothing under {sources}; the agent would be blind")

    body = executors["read_doc"](hits[0])
    if body.startswith("Error"):
        raise SystemExit(f"read_doc failed on {hits[0]}: {body}")

    by_path = {path: dsid for dsid, path in uuid_index.items()}
    by_path.update({filename: dsid for dsid, filename in SCAFFOLD_PAGES.items()})
    unmapped = [p for p in read_paths if p not in by_path]
    if unmapped:
        raise SystemExit(
            f"read_doc served {unmapped[0]}, which the tier's index does not map to a "
            f"document id; the arm would report answers standing on no documents"
        )
    print(
        f"  agent preflight: {len(folders)} source folder(s), grep and read_doc serve, "
        f"paths map back to ids"
    )
