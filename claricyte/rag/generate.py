"""Prompt assembly and citation checking.

Pure: no network, no provider. Building the prompt and validating what comes back
are separate from sending it, so both are testable without an API key.

The grounding guarantee lives in the system prompt and is enforced afterwards by
validate. A model that cites [4] when only three sources were supplied has
invented something, and that is caught mechanically rather than trusted.
"""

from __future__ import annotations

import re

from claricyte.rag.corpus import Chunk

# What the model says when the sources do not answer the question. Abstention is
# a feature, so it gets a fixed string the UI and the eval can both recognise.
ABSTAIN = "The available sources do not cover this."

# A question about what this particular cell looks like is refused for a
# different reason: the sources may describe the cell type perfectly well, and
# the refusal is that the model has not seen the image and the CBM owns that
# claim. Measured, not anticipated: asked how many lobes the cell had, the model
# answered "2 to 5 lobes joined by a thin filament" and cited it.
DECLINE_VISUAL = "That is a question about this image, which this panel cannot see."

# No persona line. Role prompts do not reliably help factual QA and sometimes hurt
# (Zheng et al., EMNLP Findings 2024), so the work is done by explicit rules. The
# audience is stated because it sets the register, which is a different job.
SYSTEM_PROMPT = f"""Answer questions about white blood cells for laboratory \
science students and technologists, using only the numbered source excerpts \
provided.

Rules:
1. Every factual claim must cite its source inline, one bracket per source, as \
[1][2]. Cite only the numbers you were given.
2. You have not seen the cell, so never state or estimate anything about how \
THIS one looks: its lobe count, its granules, its size, or whether its \
identification is correct. When that is the whole question, reply exactly \
"{DECLINE_VISUAL}" and add nothing else. When the cell is only referred to in \
passing and the question can be answered about the cell TYPE in general, answer \
it normally from the sources; describing a cell type is expected.
3. Answer whichever parts of the question the excerpts support and silently drop \
the rest. Never write a sentence about what the excerpts do or do not cover. \
Only if they support none of it, reply exactly: "{ABSTAIN}" Never fall back on \
your own knowledge or pad a thin answer.
4. Describe associations, workup and management only in the general terms the \
sources use. Never frame anything as advice about a particular patient or case.
5. If sources disagree, say so and cite both rather than picking one.
6. Be brief: three or four sentences. Open with the substance. Never begin by \
restating the question or by referring to the excerpts.
"""

# Matches [1] and also [1, 5], which models emit despite being asked for one
# bracket per source. Missing that form marked properly cited sentences as
# ungrounded, which would have made the eval's groundedness number meaningless.
CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_NUMBER = re.compile(r"\d+")


def format_sources(chunks: list[Chunk]) -> str:
    """Number the retrieved chunks for the prompt. Numbering is 1-based and
    positional, so [2] means the second chunk passed in."""
    return "\n\n".join(
        f"[{i}] ({chunk.title}, {chunk.section})\n{chunk.text}"
        for i, chunk in enumerate(chunks, 1)
    )


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    """The chat messages for one question against its retrieved sources."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{format_sources(chunks)}\n\nQuestion: {question}",
        },
    ]


def cited(answer: str) -> set[int]:
    """Source numbers the answer refers to, from either bracket form."""
    return {
        int(n) for group in CITATION.findall(answer) for n in _NUMBER.findall(group)
    }


def invalid_citations(answer: str, source_count: int) -> set[int]:
    """Cited numbers that were never supplied. Non-empty means fabrication."""
    return {n for n in cited(answer) if n < 1 or n > source_count}


def abstained(answer: str) -> bool:
    """True if the model declined, for either reason.

    Callers care that it declined, not why: the UI renders both the same and the
    eval counts both as a refusal. The two strings are separate so the reader
    gets the right reason.
    """
    lowered = answer.lower()
    return any(phrase.lower() in lowered for phrase in (ABSTAIN, DECLINE_VISUAL))


def uncited_sentences(answer: str) -> list[str]:
    """Sentences making a claim with no citation.

    Rough by design: it exists to flag drift in the eval, not to gate output.
    Abstention and short connective fragments are not claims.
    """
    if abstained(answer):
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    return [s for s in sentences if not CITATION.search(s) and len(s.split()) > 5]
