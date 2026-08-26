"""The scaling study's own published scores, transcribed once.

Tables 11 and 12 of *BM25 Wins at Scale* report seven RAG paradigms against corpus
size on the official combined score. Two of those paradigms are re-measured on this
ladder; the other five are carried into the final report as published-not-measured,
so a reader sees the reproduction against the thing it reproduces rather than being
asked to take the fidelity on trust.

Transcribed here rather than restated at each call site because the gate is measured
against these numbers and the report is drawn beside them, and a curve whose anchor
and whose overlay disagreed by a transcription slip would look like a finding.

The tier rows of Table 7 -- documents, tokens, chunks -- are not here: they describe
the corpus rather than an arm, and ``ladder.census`` owns them.
"""

from __future__ import annotations

from typing import NamedTuple

# The four rungs this ladder builds, in ascending order. The study's ladder is 28
# strictly nested tiers; these are its 0, 3, 8 and 13 -- the rungs where both
# paper-faithful arms carry a published score.
TIERS = ("T0", "T3", "T8", "T13")


class Paradigm(NamedTuple):
    """One published curve: a paradigm's official combined score at each rung."""

    name: str
    scores: dict[str, float | None]
    reproduced_as: str | None

    def score_at(self, tier: str) -> float | None:
        """The published combined score at *tier*, or None where none was published.

        A None is not a zero and not a gap in the transcription: it is a rung above
        the corpus size at which that paradigm's construction stopped. MS-GraphRAG
        completes through 8,750 documents and LightRAG through 2,254, so neither has
        a value at every rung, and a chart must omit those points rather than plot
        them at the axis.
        """
        return self.scores[tier]


# Tables 11 and 12, official combined correctness x completeness score.
PUBLISHED = (
    Paradigm(
        "BM25",
        {"T0": 74.7, "T3": 71.4, "T8": 70.1, "T13": 64.9},
        reproduced_as="bm25",
    ),
    Paradigm(
        "File-System Agent",
        {"T0": 77.4, "T3": 75.4, "T8": 69.9, "T13": 62.6},
        reproduced_as="agent",
    ),
    Paradigm(
        "DenseRAG",
        {"T0": 58.1, "T3": 55.7, "T8": 51.0, "T13": 44.2},
        reproduced_as=None,
    ),
    Paradigm(
        "HippoRAG 2",
        {"T0": 66.2, "T3": 63.1, "T8": 58.6, "T13": 53.8},
        reproduced_as=None,
    ),
    Paradigm(
        "LinearRAG",
        {"T0": 46.2, "T3": 44.1, "T8": 38.8, "T13": 34.3},
        reproduced_as=None,
    ),
    Paradigm(
        "MS-GraphRAG",
        {"T0": 45.9, "T3": 44.0, "T8": 38.4, "T13": None},
        reproduced_as=None,
    ),
    Paradigm(
        "LightRAG",
        {"T0": 48.0, "T3": 42.5, "T8": None, "T13": None},
        reproduced_as=None,
    ),
)

# The two arms measured here, keyed by the arm name their output directories carry.
REPRODUCED = {
    paradigm.reproduced_as: paradigm
    for paradigm in PUBLISHED
    if paradigm.reproduced_as is not None
}


def published_score(arm: str, tier: str) -> float:
    """The published combined score the reproduced *arm* is measured against.

    Raises:
        KeyError: if *arm* is not one of the two reproduced here, or *tier* is not a
            rung of this ladder. Both are programming errors rather than run-time
            conditions -- an arm with no published counterpart has nothing to be
            validated against and must not silently score against something else.
    """
    if arm not in REPRODUCED:
        raise KeyError(
            f"{arm!r} is not a reproduced arm; expected one of {sorted(REPRODUCED)}"
        )
    if tier not in TIERS:
        raise KeyError(f"{tier!r} is not a rung of this ladder; expected one of {TIERS}")
    score = REPRODUCED[arm].score_at(tier)
    if score is None:  # pragma: no cover -- both reproduced arms score at every rung
        raise KeyError(f"{arm!r} carries no published score at {tier}")
    return score
