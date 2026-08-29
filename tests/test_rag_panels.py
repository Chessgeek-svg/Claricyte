"""Tests for the precomputed panels.

The fallback is what matters: a cell whose class has no vacuolated variant must
still get a panel. A missing key would be a KeyError in the reveal path, on the
happy path of the demo.
"""

import pytest

from claricyte.rag.generate import abstained, cited, invalid_citations
from claricyte.rag.panels import (
    VACUOLE_VARIANTS,
    Panel,
    Source,
    load_panels,
    panel_key,
    save_panels,
)
from claricyte.vocab import CLASSES

FINDINGS = ["cytoplasmic vacuolation"]


def test_no_findings_gives_the_base_panel():
    assert panel_key("Monocyte", []) == "Monocyte"
    assert panel_key("Monocyte", None) == "Monocyte"


@pytest.mark.parametrize("label", VACUOLE_VARIANTS)
def test_vacuolation_selects_the_variant_where_one_exists(label):
    assert panel_key(label, FINDINGS) == f"{label}|vacuolated"


@pytest.mark.parametrize("label", [c for c in CLASSES if c not in VACUOLE_VARIANTS])
def test_vacuolation_falls_back_when_no_variant_exists(label):
    """Vacuoles in a basophil are rare enough that no panel was generated. The
    base panel is worse than a tailored one and far better than a KeyError."""
    assert panel_key(label, FINDINGS) == label


def test_every_key_the_app_can_ask_for_exists():
    """Walks every class with and without vacuolation, which is the full set of
    lookups the reveal path can make."""
    panels = load_panels()
    for label in CLASSES:
        for findings in ([], FINDINGS):
            assert panel_key(label, findings) in panels


def test_panels_round_trip(tmp_path):
    original = {
        "Basophil": Panel(
            text="Basophilia is associated with CML [1].",
            sources=[Source(1, "A Title", "Introduction", "https://example.org/")],
        )
    }
    path = tmp_path / "panels.json"
    save_panels(original, path)
    assert load_panels(path) == original


def test_every_committed_panel_is_cited_and_in_range():
    """A panel with no citation is either an abstention that slipped through or
    an ungrounded answer, and one citing a source it was not given is invented.
    Neither should ship."""
    for key, panel in load_panels().items():
        assert cited(panel.text), key
        assert not invalid_citations(panel.text, len(panel.sources)), key


def test_no_committed_panel_is_an_abstention():
    """Abstaining is correct behaviour live, but a shipped panel that declines is
    a build that should have been rerun."""
    for key, panel in load_panels().items():
        assert not abstained(panel.text), key
