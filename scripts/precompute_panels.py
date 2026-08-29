"""Generate the clinical context panels and commit them.

    python scripts/precompute_panels.py

Offline and one-shot, about $0.002 for all eight. The quiz path reads these, so
browsing cells costs nothing and cannot be abused. Rerun after changing the
corpus, the prompt or the query.
"""

import argparse

from claricyte.rag.generate import build_messages, invalid_citations
from claricyte.rag.panels import (
    PANELS_PATH,
    VACUOLE_VARIANTS,
    Panel,
    Source,
    panel_key,
    save_panels,
)
from claricyte.rag.providers import get_provider
from claricyte.rag.query import build_query, panel_question
from claricyte.rag.store import search
from claricyte.vocab import CLASSES

VACUOLATED = {"cytoplasm_vacuole": ("yes", 0.99)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=PANELS_PATH)
    # Higher than the live default of 5: panels are generated once, offline, so
    # a wider net costs nothing and stops a thin retrieval forcing an abstention.
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--provider", default="openai")
    args = parser.parse_args()

    provider = get_provider(args.provider)
    wanted = [(label, {}) for label in CLASSES]
    wanted += [(label, VACUOLATED) for label in VACUOLE_VARIANTS]

    panels = {}
    for label, result in wanted:
        findings = ["cytoplasmic vacuolation"] if result else []
        key = panel_key(label, findings)
        text, where = build_query(result, label)
        chunks = [chunk for chunk, _ in search(text, where=where, k=args.k)]
        # The retrieval query is not the question: it is shaped for embedding
        # similarity and reads as a topic, which makes the model abstain.
        answer = provider.generate(
            build_messages(panel_question(label, findings), chunks)
        )

        bad = invalid_citations(answer, len(chunks))
        if bad:
            raise RuntimeError(f"{key}: cites sources that do not exist: {sorted(bad)}")

        panels[key] = Panel(
            text=answer,
            sources=[
                Source(i, chunk.title, chunk.section, chunk.url)
                for i, chunk in enumerate(chunks, 1)
            ],
        )
        print(f"{key:34} {len(answer):4} chars, {len(chunks)} sources")

    save_panels(panels, args.out)
    print(f"\nwrote {len(panels)} panels -> {args.out}")


if __name__ == "__main__":
    main()
