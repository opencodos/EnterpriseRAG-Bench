"""Shared vocabulary for the two reproduced arms: the settings, the tier, the chunk.

Both arms answer the same 500 questions against one rung of the ladder under the
study's shared settings, and the settings are the reproduction — so they live here as
named constants rather than as defaults on two argument parsers that could drift apart
between tiers.

The text of a document and the size of a chunk come from ``ladder.common`` rather than
being restated. A tier's published token and chunk counts were validated against the
study's Table 7 over exactly that string and that window, so a retriever that chunked a
differently-joined document would be searching a corpus the ladder never measured. One
definition, imported twice, is what keeps the two from drifting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from ladder.common import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENCODING_NAME,
    document_text,
    file_digest,
)

# The study's shared retrieval settings. Both arms carry them; neither may tune them.
TOP_K = 5
MAX_LLM_CALLS = 80

# What the top-5 counts. Settled by the study's Table 8 -- "Retriever depth: top-5
# **chunks** where applicable" -- and by its Appendix C: "BM25, DenseRAG, and HippoRAG 2
# use the same shared chunks." Chunk-level is the specification.
#
# This was read the other way for one measurement, on the argument that the benchmark's
# own BM25 baseline retrieves whole documents so a chunked BM25 must be our
# construction. The paper says otherwise, and the paper is what is being reproduced.
#
# Measured at T0, the two are within half a point of each other, which is the reason
# the setting is *recorded* in a cell's run.json rather than inferred: nothing in a
# results file distinguishes them, so a ladder built from a mixture would look
# entirely consistent.
#
#     chunk-level     combined 62.15   correctness 79.4   completeness 66.6
#     document-level  combined 61.66   correctness 80.8   completeness 66.3
#
RETRIEVAL_GRANULARITY = "chunk"
GRANULARITIES = ("document", "chunk")

# The reader stack the study serves. The embedding model is part of that stack and is
# stood up beside the reader, but neither reproduced arm queries it: BM25 is lexical
# and the File-System Agent greps. It is here so the host that serves the reader is
# described in one place, not because these two arms retrieve with it.
READER_MODEL = "Qwen/Qwen3.6-27B"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

_encoding = tiktoken.get_encoding(ENCODING_NAME)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split *text* into fixed-stride token windows.

    The overlap is an absolute token count, which is what the study publishes.  The
    repository's own vector indexer takes an overlap *fraction* instead, so at 1,200
    tokens its 0.1 default is 120 rather than 100 -- close enough to look right and
    wrong enough to move a chunk count the ladder is validated against.
    """
    tokens = _encoding.encode(text, disallowed_special=())
    if len(tokens) <= size:
        return [text]
    stride = size - overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), stride):
        chunks.append(_encoding.decode(tokens[start : start + size]))
        if start + size >= len(tokens):
            break
    return chunks


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a window of one document's indexed text.

    A whole document is the degenerate case, ``index=0`` of ``total=1`` -- which is
    already what a document shorter than the window produces, so document-level
    retrieval is not a second kind of thing here but the same one at its limit.
    """

    dsid: str
    index: int
    total: int
    title: str
    text: str

    @property
    def point_id(self) -> str:
        """The id this chunk is stored under, deterministic in (document, position)."""
        return f"{self.dsid}:{self.index}"

    @property
    def is_whole_document(self) -> bool:
        return self.total == 1


def whole_document(doc: dict[str, Any]) -> Chunk:
    """One corpus document as a single retrievable unit, unwindowed.

    The text is exactly what ``document_chunks`` would have windowed, so the two
    granularities differ in where the text is cut and not in what it is.

    Raises KeyError when the document carries no field labels, which is how a file
    under ``sources/`` that is not a corpus document announces itself.
    """
    return Chunk(
        dsid=doc["dataset_doc_uuid"],
        index=0,
        total=1,
        title=str(doc[doc["title_field_name"]]),
        text=document_text(doc),
    )


def document_chunks(doc: dict[str, Any]) -> list[Chunk]:
    """Every chunk of one corpus document, in reading order.

    Raises KeyError when the document carries no field labels, which is how a file
    under ``sources/`` that is not a corpus document announces itself.
    """
    text = document_text(doc)
    title = str(doc[doc["title_field_name"]])
    pieces = chunk_text(text)
    return [
        Chunk(
            dsid=doc["dataset_doc_uuid"],
            index=position,
            total=len(pieces),
            title=title,
            text=piece,
        )
        for position, piece in enumerate(pieces)
    ]


def format_context_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks into the context block the reader is given.

    Deliberately parallel to ``src.utils.retrieval.format_context_documents`` -- same
    delimiter, same ``Title:`` line -- because the study's BM25 and the shipped
    document-level runner should differ in what they retrieve and not in how the
    result is presented.  The chunk's position in its document is stated, so a reader
    handed the middle of a long document can tell that it is holding a fragment.

    **This rendering is a published interface.** The control arm reads this exact
    string verbatim rather than reassembling one from the chunks, so that it differs
    from the BM25 arm in the model alone; changing the wording here changes what that
    arm is a control for, and cells measured before and after are not comparable.
    """
    parts: list[str] = []
    for position, chunk in enumerate(chunks, 1):
        where = (
            f", chunk {chunk.index + 1}/{chunk.total}"
            if chunk.total > 1
            else ""
        )
        parts.append(
            f"--- Chunk {position} (Document ID: {chunk.dsid}{where}) ---\n"
            f"Title: {chunk.title}\n\n{chunk.text}"
        )
    return "\n\n".join(parts)


def documents_of(chunks: list[Chunk]) -> list[str]:
    """The documents behind retrieved chunks, deduplicated, best rank first.

    The official metrics grade documents, not chunks, and two of the top five
    routinely come from one document -- so an arm retrieving five chunks can offer
    fewer than five documents, and that is the honest count rather than something to
    pad back up to the retrieval depth.
    """
    seen: list[str] = []
    for chunk in chunks:
        if chunk.dsid not in seen:
            seen.append(chunk.dsid)
    return seen


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


class TierError(RuntimeError):
    """The tier tree is not the tier it claims to be."""


@dataclass(frozen=True)
class Tier:
    """One rung of the ladder, as materialized on the box that measures it."""

    name: str
    root: Path
    sources: Path
    dsids: tuple[str, ...]
    manifest_sha256: str
    provenance: dict[str, Any]

    @property
    def documents(self) -> int:
        """The tier's document count, which is not its manifest's line count.

        The bedrock's two organizational pages are not corpus documents -- they sit
        outside ``sources/`` and carry no id -- so they travel beside the manifest and
        the count is its lines plus two.  Neither reproduced arm reads them: the agent
        explores ``sources/`` and BM25 indexes ids, exactly as the study's do.
        """
        return len(self.dsids) + int(self.provenance.get("organizational_pages", 0))

    def uuid_index(self) -> dict[str, str]:
        """This tier's id-to-path map, built from the tier tree and cached beside it.

        Built here rather than read from the corpus's ``uuid_index.json`` because that
        file maps the whole corpus: an arm given it could resolve, and an agent could
        select, a document the tier does not contain.
        """
        cache = self.root / "uuid_index.json"
        if cache.is_file():
            index: dict[str, str] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            from src.utils.document_index import build_uuid_index

            index = build_uuid_index(str(self.sources))
            cache.write_text(json.dumps(index, indent=2), encoding="utf-8")

        missing = set(self.dsids) - index.keys()
        if missing:
            raise TierError(
                f"{len(missing)} manifest document(s) are not in {self.sources}, "
                f"starting {sorted(missing)[:3]}; a short tier is worse than no tier"
            )
        extra = index.keys() - set(self.dsids)
        if extra:
            raise TierError(
                f"{self.sources} holds {len(extra)} document(s) the manifest does not "
                f"name, starting {sorted(extra)[:3]}; the arms would search a corpus "
                f"wider than the rung they report"
            )
        return index


def load_tier(root: Path) -> Tier:
    """Read a materialized tier tree, refusing one that is not intact.

    Checked rather than trusted, because every way this goes wrong produces a number
    rather than a crash: a truncated manifest measures a smaller rung under a larger
    rung's name, and nothing downstream -- not the scorer, not the chart -- would show
    it.  The manifest's checksum is verified against the provenance the ladder build
    committed, so the rung being measured is the rung a third party can rebuild.
    """
    root = root.resolve()
    manifests = sorted(root.glob("*.manifest"))
    if len(manifests) != 1:
        raise TierError(
            f"expected exactly one *.manifest in {root}, found {len(manifests)}; "
            f"point at a directory written by `python -m ladder.materialize`"
        )
    manifest_path = manifests[0]
    name = manifest_path.stem

    provenance_path = root / f"{name}.provenance.json"
    if not provenance_path.is_file():
        raise TierError(f"{provenance_path} is missing; the tier cannot identify itself")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    digest = file_digest(manifest_path)
    if digest != provenance.get("manifest_sha256"):
        raise TierError(
            f"{manifest_path} does not match the checksum its provenance records\n"
            f"  file is  {digest}\n"
            f"  expected {provenance.get('manifest_sha256')}"
        )

    dsids = tuple(
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(set(dsids)) != len(dsids):
        raise TierError(f"{manifest_path} names a document more than once")

    recorded = provenance.get("manifest_lines")
    if recorded is not None and recorded != len(dsids):
        raise TierError(
            f"{manifest_path} holds {len(dsids)} lines, provenance records {recorded}"
        )

    sources = root / "sources"
    if not sources.is_dir():
        raise TierError(f"{sources} is missing; materialize the tier first")

    # Counted as well as keyed, because the two catch different faults. A second copy
    # of a document already named carries an id the manifest knows, so it survives any
    # comparison of id sets -- and it would be chunked and indexed twice, giving that
    # document two shots at every query. The corpus this tree is cut from has four
    # such collisions, which is why ``ladder.materialize`` asserts one file per id.
    files = sum(1 for path in sources.rglob("*.json") if path.is_file())
    if files != len(dsids):
        raise TierError(
            f"{sources} holds {files:,} files for {len(dsids):,} manifest document(s); "
            f"the tree is not one file per id"
        )

    return Tier(
        name=name,
        root=root,
        sources=sources,
        dsids=dsids,
        manifest_sha256=digest,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# The bedrock's two organizational pages
# ---------------------------------------------------------------------------

# The study counts them inside the tier -- "Together with the benchmark's two
# organizational overview pages, which we include as scaffolds, the full evaluation
# tier holds 511,959 documents" (S3.1) -- and 10 of the 500 questions are
# "scaffold-supported high-level" ones whose only evidence they are (Figure 2). A
# retriever that cannot return them cannot answer those questions at all, so indexing
# them is what the paper specifies, not an embellishment of it.
#
# They are not corpus documents: both sit outside ``sources/``, carry no
# ``dataset_doc_uuid`` and appear in no ``uuid_index.json``, which is why the manifest
# -- frozen as document ids and nothing else -- names neither, and why a tier's
# document count is its manifest's line count plus two. The ids below exist so that a
# retrieved scaffold has something to be reported as; they are this harness's, they are
# stable, and they are deliberately outside the ``dsid_<hex>`` shape a corpus id takes
# so that no join to the corpus can silently match one.
SCAFFOLD_PAGES: dict[str, str] = {
    "scaffold_company_overview": "company_overview.md",
    "scaffold_initiatives": "initiatives.md",
}


def scaffold_chunks(tier: "Tier", granularity: str = RETRIEVAL_GRANULARITY) -> list[Chunk]:
    """The two organizational pages as retrievable units, chunked as documents are.

    Rendered as ``title\\n\\ncontent`` and windowed by the same chunker, so a scaffold
    competes with a corpus document on the same terms rather than on a longer or
    shorter unit. The title is the page's first markdown heading where it has one and
    the file's stem otherwise.

    A page the tier tree does not carry is an error rather than a skip: the tier would
    then hold 1,143 documents under a name that promises 1,144, and the ten high-level
    questions would score zero for a reason no results file records.
    """
    chunks: list[Chunk] = []
    for dsid, filename in SCAFFOLD_PAGES.items():
        path = tier.root / filename
        if not path.is_file():
            raise TierError(
                f"{path} is missing; the tier's two organizational pages are part of "
                f"its {tier.documents} documents and the high-level questions have no "
                f"other evidence"
            )
        body = path.read_text(encoding="utf-8").strip()
        heading = next(
            (
                line.lstrip("#").strip()
                for line in body.splitlines()
                if line.startswith("#") and line.lstrip("#").strip()
            ),
            path.stem,
        )
        text = f"{heading}\n\n{body}"
        pieces = chunk_text(text) if granularity == "chunk" else [text]
        chunks.extend(
            Chunk(dsid=dsid, index=position, total=len(pieces), title=heading, text=piece)
            for position, piece in enumerate(pieces)
        )
    return chunks
