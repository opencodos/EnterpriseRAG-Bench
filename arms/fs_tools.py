"""The three read-only tools the study's raw File-System Agent is given.

Table 9 of arXiv 2607.26497 exposes exactly these, with exactly these caps:

    list_dir   relative path   up to 200 child names
    grep       fixed string    up to 30 matching paths
    read_doc   relative path   first 8,000 characters

and states that "Path resolution rejects traversal outside the corpus root".

**The caps are the paradigm, not ergonomics.** The agent's 80-call budget only means
what the study measured it to mean if each call returns what theirs returned: an agent
that can read a whole document in one call, or see 2,000 filenames, spends its budget
differently from one that cannot. So the caps are constants here and nothing takes them
from the environment.

This is deliberately *not* the repository's shipped agent surface, which exposes a
``run`` tool executing arbitrary shell -- pipes, regex, command chaining, parallel
dispatch -- plus a document reader and an explicit ``select_doc_by_dsid``. That agent is
strictly more capable than the one the study describes, so an arm built on it measures a
different system. ``grep`` here is a fixed-string scan for the same reason: the study
says "fixed string", and regex is a different search primitive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

# Table 9's caps.
LIST_DIR_MAX_CHILDREN = 200
GREP_MAX_PATHS = 30
READ_DOC_MAX_CHARS = 8_000


class Traversal(Exception):
    """A path argument tried to leave the corpus root."""


def _resolve(root: Path, relative: str) -> Path:
    """Resolve *relative* under *root*, refusing anything that escapes it.

    Checked on the resolved path rather than by inspecting the string, so that "..",
    an absolute path and a symlink pointing outward are all one rule.
    """
    candidate = (root / relative.lstrip("/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise Traversal(relative)
    return candidate


def make_tools(
    root: Path, read_paths: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]]]:
    """Build the schemas and executors for one question's agent.

    Args:
        root: The tier tree's ``sources/`` directory. The agent sees this and nothing
            above it, so the search space is the rung and not the box.
        read_paths: Accumulates every path ``read_doc`` served, in call order. The
            study's agent has no explicit "select these documents" tool -- the shipped
            one does -- so what the agent *read* is the only record of what it
            retrieved, and the arm's document ids come from here.

    Returns:
        (tool schemas in OpenAI format, name -> executor).
    """
    root = root.resolve()

    def list_dir(path: str = ".") -> str:
        try:
            target = _resolve(root, path)
        except Traversal:
            return f"Error: '{path}' is outside the corpus root."
        if not target.is_dir():
            return f"Error: '{path}' is not a directory."
        names = sorted(entry.name for entry in target.iterdir())
        shown = names[:LIST_DIR_MAX_CHILDREN]
        out = "\n".join(shown)
        if len(names) > LIST_DIR_MAX_CHILDREN:
            out += (
                f"\n... {len(names) - LIST_DIR_MAX_CHILDREN} more child(ren) not shown "
                f"(limit {LIST_DIR_MAX_CHILDREN})"
            )
        return out or "(empty directory)"

    def grep(query: str) -> str:
        """Fixed-string, case-insensitive scan; returns matching paths, not lines.

        Table 9's return value is "up to 30 matching paths", so a match is a property
        of a file and the tool says which files matched rather than showing context.
        Case-insensitive because the study's own tooling is, and because a
        case-sensitive fixed string would make the tool nearly unusable on prose.
        """
        needle = query.lower()
        hits: list[str] = []
        for path in sorted(root.rglob("*.json")):
            try:
                if needle in path.read_text(encoding="utf-8", errors="ignore").lower():
                    hits.append(os.path.relpath(path, root))
            except OSError:
                continue
            if len(hits) >= GREP_MAX_PATHS:
                break
        if not hits:
            return "No matches."
        out = "\n".join(hits)
        if len(hits) >= GREP_MAX_PATHS:
            out += f"\n... results truncated at {GREP_MAX_PATHS} paths"
        return out

    def read_doc(path: str) -> str:
        try:
            target = _resolve(root, path)
        except Traversal:
            return f"Error: '{path}' is outside the corpus root."
        if not target.is_file():
            return f"Error: '{path}' is not a file."
        try:
            raw = target.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return f"Error reading '{path}': {exc}"
        # Rendered as the document's own text where it is a corpus document, so the
        # agent reads prose rather than JSON punctuation -- the study's agent reads
        # documents, and the corpus's on-disk form is an artifact of this repository.
        try:
            doc = json.loads(raw)
            from ladder.common import document_text

            raw = document_text(doc)
        except Exception:  # noqa: BLE001 -- a non-corpus file is served as it lies
            pass
        read_paths.add(os.path.relpath(target, root))
        if len(raw) > READ_DOC_MAX_CHARS:
            return (
                raw[:READ_DOC_MAX_CHARS]
                + f"\n... [truncated at {READ_DOC_MAX_CHARS} characters]"
            )
        return raw

    schemas: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": (
                    f"List the children of a directory, relative to the corpus root. "
                    f"Returns up to {LIST_DIR_MAX_CHILDREN} names."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path; '.' for the corpus root.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": (
                    f"Find documents containing a fixed string. Returns up to "
                    f"{GREP_MAX_PATHS} matching paths."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Fixed string to search for (not a regex).",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_doc",
                "description": (
                    f"Read a document by relative path. Returns its first "
                    f"{READ_DOC_MAX_CHARS:,} characters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the document.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    executors: dict[str, Callable[..., str]] = {
        "list_dir": list_dir,
        "grep": grep,
        "read_doc": read_doc,
    }
    return schemas, executors
