"""Prompts for the corpus ladder's method-blind trap and lure filter.

Two systems rather than one, because the two judgements are not the same question.
For an answerable question a candidate is adversarial when it competes with the gold
answer and loses; for an unanswerable one there is no gold answer to compete with,
and the property that matters is topical closeness without the fact.
"""

TRAP_FILTER_SYSTEM = """You classify retrieval candidates for an enterprise RAG benchmark.

You see a question, the gold documents that answer it, and candidate documents a \
retriever surfaced. For each candidate return exactly one verdict:

- "trap": the candidate is about the same entity, project, deal, ticket or topic as \
a gold document, but what it reports is superseded, contradicted or otherwise wrong \
for this question -- an older version number, a different date, a decision that was \
later reversed, a figure that was later corrected. A trap is dangerous precisely \
because a retriever that lands on it looks right.
- "unrelated": the candidate shares no entity or topic with the gold documents, or \
shares one but reports nothing that competes with the gold answer.
- "answers": the candidate genuinely answers the question. Gold documents themselves \
would fall here; so would a document that duplicates their content.

Judge only what the text says. Do not infer a conflict from a candidate merely being \
older or shorter. When a candidate is on-topic but reports nothing that could be \
mistaken for an answer, it is "unrelated", not "trap"."""

LURE_FILTER_SYSTEM = """You classify retrieval candidates for an enterprise RAG benchmark.

The question is one the corpus deliberately cannot answer. You see the question and \
candidate documents a retriever surfaced. For each candidate return exactly one verdict:

- "cannot_answer": the candidate is topically close to the question -- it mentions \
the same entity, product, team or theme -- but does not contain the fact the question \
asks for. These are what keep an unanswerable question from being solved by the mere \
absence of retrieved text.
- "answers": the candidate does in fact contain the fact the question asks for. Say \
this whenever the question is answerable from the candidate, even partially.
- "unrelated": the candidate has nothing to do with the question's subject.

Be strict about "answers": it means the benchmark's own premise is wrong for this \
question, so only say it when the text really does carry the fact."""

CANDIDATE_FILTER_PROMPT = """Question: {question}

{gold_block}Candidates:
{candidates}

Return JSON and nothing else:
{{"verdicts": [{{"candidate": <int>, "verdict": "<verdict>", "why": "<one short clause>"}}]}}

Return one entry per candidate, in the order given."""
