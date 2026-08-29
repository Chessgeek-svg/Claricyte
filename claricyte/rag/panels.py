"""Precomputed clinical context panels.

One panel per class, generated offline and committed, so the quiz path costs no
API call and has no abuse surface. Only the question box calls out live.

Vacuolation gets its own variant for the two classes where it means something.
A cell whose class has no variant falls back to the base panel rather than
having nothing, since a missing panel is worse than a slightly generic one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

PANELS_PATH = "rag_data/panels.json"

# Vacuolation points at sepsis and toxic change in these two. In lymphocytes,
# eosinophils and basophils it is rare enough that the literature has little to
# say, so a variant would retrieve the same sources as the base panel.
VACUOLE_VARIANTS = ("Monocyte", "Segmented Neutrophil")


class Source(NamedTuple):
    """One cited source, as the panel needs to render it."""

    number: int
    title: str
    section: str
    url: str


class Panel(NamedTuple):
    """A generated answer and the sources it cites."""

    text: str
    sources: list[Source]


def panel_key(label: str, findings: list[str] | None = None) -> str:
    """Key for the panel matching this cell, falling back to the base panel."""
    if findings and label in VACUOLE_VARIANTS:
        return f"{label}|vacuolated"
    return label


def load_panels(path: str | Path = PANELS_PATH) -> dict[str, Panel]:
    """Read the committed panels."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        key: Panel(
            text=entry["text"],
            sources=[Source(**source) for source in entry["sources"]],
        )
        for key, entry in raw.items()
    }


def save_panels(panels: dict[str, Panel], path: str | Path = PANELS_PATH) -> None:
    """Write panels as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                key: {
                    "text": panel.text,
                    "sources": [source._asdict() for source in panel.sources],
                }
                for key, panel in panels.items()
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
