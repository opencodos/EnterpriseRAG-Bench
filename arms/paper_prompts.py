"""The scaling study's own prompts, transcribed verbatim from its appendix.

**Nothing here may be tuned.** These strings are the specification, not a starting
point: an arm that answers under a prompt of ours is not the paradigm the study
measured, however similar the wording looks. The repository ships its own prompts and
they are *not* these -- using them was the defect this module exists to correct.

Source: arXiv 2607.26497, *BM25 Wins at Scale*, Appendix C.1 (reader) and C.2
(File-System Agent).

What the repository's prompts would otherwise have supplied, and why it mattered: the
shipped ``src.prompts.vector_search_answer_gen.ANSWER_GEN_PROMPT`` adds "Many of the
documents provided are likely to be irrelevant", "Be concise and **only provide
information directly relevant to the query**" and "do not include any additional text
or formatting", and sends the whole thing as one user turn with no system message.
Measured at T0 under that prompt, 96.2% of the gold facts the scorer did not credit
were present in the context the reader was given and simply not stated, and the arm
scored 61.66 against a published 74.7. The study's prompt says "be concise" and stops
there.

**One transcription uncertainty, disclosed for the methodology note.** The appendix
renders the user template across four lines and a PDF cannot distinguish a paragraph
break from a line wrap, so whether a blank line separates ``{context}`` from
``QUESTION:`` is not recoverable. The literal reading is taken here.
"""

from __future__ import annotations

# Appendix C.1. Reflowed from the appendix's line wrapping into running prose; no
# word, mark or ordering is changed.
READER_SYSTEM_PROMPT = (
    "You are a retrieval-based QA assistant. Answer the QUESTION using ONLY the "
    "information in the CONTEXT. Do not use outside knowledge or invent facts. If "
    "the CONTEXT does not contain enough information to answer, say so explicitly. "
    "Answer in the same language as the QUESTION; be concise."
)

# Appendix C.1. The bare template -- no headings, no output instruction, and the
# trailing "ANSWER:" is the whole of the cue the reader is given.
READER_USER_TEMPLATE = "CONTEXT:\n{context}\nQUESTION: {question}\nANSWER:"

# Appendix C.2. The source folders are named in the prompt itself, so they are part
# of the transcription rather than something to derive from the tier tree.
AGENT_SYSTEM_PROMPT = (
    "You are a research assistant over an enterprise document corpus, organized as "
    "files under source folders (slack, gmail, jira, confluence, google_drive, "
    "linear, github, fireflies, hubspot). Use list_dir to orient, grep to locate "
    "relevant documents by keyword, and read_doc to read them. Answer ONLY from the "
    "documents. Be concise and factual, and cite the relative file paths you used."
)


def reader_messages(context: str, question: str) -> list[tuple[str, str]]:
    """The reader's two messages as ``(role, content)`` pairs, in order.

    Returned as pairs rather than as provider objects so that this module imports
    nothing and stays a transcription. The caller builds whatever the client wants.
    """
    return [
        ("system", READER_SYSTEM_PROMPT),
        ("user", READER_USER_TEMPLATE.format(context=context, question=question)),
    ]
