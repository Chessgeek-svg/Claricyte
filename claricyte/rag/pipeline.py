"""Question to cited answer: query, retrieve, generate, check.

The live counterpart to the precomputed panels, and the reason retrieval is
architecturally necessary rather than a dictionary keyed by class. Both paths
end in the same Panel shape so the UI renders them with one function.

Invalid citations are reported, not raised. Offline in precompute_panels a
fabricated citation should stop the build; in front of a user it should show a
warning and still hand over the answer, which is mostly sound.
"""

from __future__ import annotations

from typing import NamedTuple

from claricyte.rag.generate import abstained, build_messages, invalid_citations
from claricyte.rag.panels import Panel, Source
from claricyte.rag.providers import Provider, get_provider
from claricyte.rag.query import build_query
from claricyte.rag.store import search

# Lower than the panels' 8: those are generated once offline, these are paid for
# per question in latency and tokens.
DEFAULT_K = 5


class Answer(NamedTuple):
    """A generated answer, its sources, and what checking found."""

    panel: Panel
    abstained: bool
    invalid: tuple[int, ...]


def ask(
    question: str,
    label: str,
    result: dict[str, tuple[str, float]] | None = None,
    k: int = DEFAULT_K,
    provider: Provider | None = None,
) -> Answer:
    """Answer `question` about a predicted `label` from the corpus.

    Args:
        question: the user's wording, used both to retrieve and to prompt.
        label: the cell class on screen, which becomes the metadata filter.
        result: {attribute: (value, confidence)}, so a notable finding can steer
            retrieval. Optional: a question needs no prediction to be answerable.
        k: chunks retrieved.
        provider: generation backend. Defaults to OpenAI.
    """
    text, where = build_query(result or {}, label, question=question)
    chunks = [chunk for chunk, _ in search(text, where=where, k=k)]
    if not chunks:
        return Answer(Panel(text="", sources=[]), abstained=True, invalid=())

    answer = (provider or get_provider()).generate(build_messages(question, chunks))
    panel = Panel(
        text=answer,
        sources=[
            Source(i, chunk.title, chunk.section, chunk.url)
            for i, chunk in enumerate(chunks, 1)
        ],
    )
    return Answer(
        panel=panel,
        abstained=abstained(answer),
        invalid=tuple(sorted(invalid_citations(answer, len(chunks)))),
    )
