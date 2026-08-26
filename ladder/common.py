"""Shared vocabulary for the ladder build: the corpus, its text, and its tokens.

Every stage reads the corpus the same way, so a document's token count, its
stratum and its rank are the same number wherever they are computed. The text of
a document is exactly what the BM25 indexer indexes -- ``f"{title}\\n\\n{content}"``
over the labelled fields -- because a tier's published token count has to describe
the same string the retrievers see, not a differently-joined one.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, NamedTuple

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "ladder"

SOURCES_DIR = REPO / "generated_data" / "sources"
UUID_INDEX_PATH = REPO / "generated_data" / "uuid_index.json"
QUESTIONS_PATH = REPO / "questions.jsonl"

# The scaling study's chunking: 1,200-token chunks with 100-token overlap. The
# repository's own indexer takes an overlap *fraction* (0.1, so 120 tokens) which
# is a different setting; the published chunk counts are the paper's, so the
# paper's numbers are what a tier is validated against.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# The tokenizer the repository indexes with. Named here rather than at each call
# site because a tier's published token count is only meaningful next to it.
ENCODING_NAME = "cl100k_base"

# The two organizational pages the bedrock carries: a company overview and an
# initiative index. Neither is a corpus document -- they are generation
# scaffolding, sit outside sources/, and carry no dataset_doc_uuid -- so they
# cannot be named in a manifest and travel beside one instead.
ORGANIZATIONAL_PAGES = (
    "generated_data/company_overview.md",
    "generated_data/initiatives.md",
)


class Document(NamedTuple):
    """A corpus document as every stage of the build sees it."""

    dsid: str
    relative_path: str
    source: str
    is_noise: bool
    text: str


def rank_key(salt: str, dsid: str) -> str:
    """Stable per-document sort key: a pure function of (salt, dsid).

    Ranking rather than shuffling is what makes the order independent of iteration
    order, pool membership and Python version, so a rebuild is byte-identical.
    """
    return hashlib.sha256(f"{salt}:{dsid}".encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document_text(doc: dict[str, Any]) -> str:
    """The indexed form of a document: title and content, joined as BM25 joins them.

    Raises KeyError when the document carries no field labels, which is how a file
    under sources/ that is not a corpus document announces itself.
    """
    title_field = doc["title_field_name"]
    content_fields = doc["content_field_names"]
    title = str(doc[title_field])
    if len(content_fields) == 1:
        content = str(doc[content_fields[0]])
    else:
        parts = []
        for field in content_fields:
            value = doc[field]
            if isinstance(value, list):
                value = "\n".join(str(item) for item in value)
            parts.append(f"{field}:\n{value}")
        content = "\n\n".join(parts)
    return f"{title}\n\n{content}"


def read_document(path: Path, sources_dir: Path = SOURCES_DIR) -> Document | None:
    """Read one corpus file, or None when it is not a document the ladder can name."""
    try:
        doc = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    dsid = doc.get("dataset_doc_uuid")
    if not dsid:
        return None
    try:
        text = document_text(doc)
    except (KeyError, TypeError):
        return None
    relative = path.relative_to(sources_dir).as_posix()
    return Document(
        dsid=dsid,
        relative_path=relative,
        source=relative.split("/")[0],
        # Documents that a noise-generation step shuffled or invented carry this
        # marker; it is one half of the order's stratification.
        is_noise="dataset_noise_document" in doc,
        text=text,
    )


def corpus_paths(sources_dir: Path = SOURCES_DIR) -> list[Path]:
    """Every candidate corpus file, in a deterministic order."""
    return sorted(path for path in sources_dir.rglob("*.json") if path.is_file())


def iter_documents(
    paths: list[Path], sources_dir: Path = SOURCES_DIR
) -> Iterator[Document]:
    for path in paths:
        document = read_document(path, sources_dir)
        if document is not None:
            yield document


def chunk_count(
    token_count: int, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> int:
    """Chunks a document of this many tokens yields under a fixed-stride window."""
    if token_count <= size:
        return 1
    stride = size - overlap
    return 1 + -(-(token_count - size) // stride)


def load_uuid_index(path: Path = UUID_INDEX_PATH) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        index: dict[str, str] = json.load(handle)
    return index


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gold_dsids(questions: list[dict[str, Any]]) -> set[str]:
    gold: set[str] = set()
    for question in questions:
        gold.update(question.get("expected_doc_ids") or [])
    return gold


def not_found_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The unanswerable questions, which are the ones lures are mined for."""
    return [
        question
        for question in questions
        if question["question_type"] == "info_not_found"
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def worker_count() -> int:
    return max(1, (os.cpu_count() or 2) - 2)
