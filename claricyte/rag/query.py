"""Turn a model prediction into a retrieval query.

Produces the two things store.search needs: a string to embed and a metadata
filter. Torch-free, so it tests without a model.

The class filters rather than being searched for. It is a fact, and a filter
guarantees where a query term only nudges.

The query is prose because the corpus is prose. Embedding "basophil, coarse
purple granules" against clinical paragraphs compares two registers and
retrieves badly.

Attributes mostly stay out. Every neutrophil has pink granules, so saying so
narrows nothing; only a finding that is abnormal and clinically associated earns
a place, which in this vocabulary means vacuolation and nothing else.
"""

from __future__ import annotations

from claricyte.rag.corpus import GENERAL

# Count abnormalities per class. The corpus is organised around these terms
# rather than around cell names, so they connect a prediction to the literature
# about it. CLINICAL REVIEW: these are the standard pairs, worth a check.
CONDITION_TERMS: dict[str, tuple[str, ...]] = {
    "Segmented Neutrophil": ("neutrophilia", "neutropenia"),
    "Band Neutrophil": ("left shift", "bandemia", "neutrophilia"),
    "Lymphocyte": ("lymphocytosis", "lymphopenia"),
    "Monocyte": ("monocytosis",),
    "Eosinophil": ("eosinophilia",),
    "Basophil": ("basophilia",),
}

# Attributes worth mentioning when abnormal. Presence is informative, absence is
# not, so each maps to the single value that earns a place in the query.
NOTABLE_FINDINGS: dict[str, tuple[str, str]] = {
    "cytoplasm_vacuole": ("yes", "cytoplasmic vacuolation"),
}

# Below this the model's own reading is too uncertain to steer retrieval with.
# Matches explain.LOW_CONFIDENCE, set from measured calibration rather than taste.
LOW_CONFIDENCE = 0.75


def class_filter(label: str) -> dict:
    """Chunks about `label`, plus cross-cutting ones.

    GENERAL is always let through. Smear review and artifact passages name no
    cell type, so a strict filter would put them permanently out of reach.
    """
    return {
        "$or": [
            {"cell_classes": {"$contains": label}},
            {"cell_classes": {"$contains": GENERAL}},
        ]
    }


def notable_findings(
    result: dict[str, tuple[str, float]], low_confidence: float = LOW_CONFIDENCE
) -> list[str]:
    """Abnormal findings the model saw confidently, as English fragments."""
    found = []
    for attribute, (notable_value, phrase) in NOTABLE_FINDINGS.items():
        if attribute not in result:
            continue
        value, confidence = result[attribute]
        if value == notable_value and confidence >= low_confidence:
            found.append(phrase)
    return found


def panel_question(label: str, findings: list[str] | None = None) -> str:
    """The question the standing clinical-context panel answers.

    Separate from the retrieval query, which is shaped to sit near the corpus in
    embedding space and reads as a topic heading.

    Phrasing measured, not chosen: "clinical significance of X" made the model
    abstain on classes it could answer perfectly well when asked concretely. An
    abstract question invites a general claim no single chunk supports.
    """
    question = (
        f"What should a student know about {label.lower()}s in peripheral blood, "
        "and what conditions are they seen in?"
    )
    if findings:
        question += f" What does {findings[0]} in this cell type suggest?"
    return question


def build_query(
    result: dict[str, tuple[str, float]],
    label: str,
    question: str | None = None,
    low_confidence: float = LOW_CONFIDENCE,
) -> tuple[str, dict]:
    """Build (text to embed, metadata filter) for one predicted cell.

    Args:
        result: {attribute: (value, confidence)} from predict.predict.
        label: the cell class being explained.
        question: the user's question. Without one, builds the standing
            clinical-context query for the class.
        low_confidence: below this, a finding is ignored.
    """
    findings = notable_findings(result, low_confidence)

    if question:
        # The user's wording leads: it is already prose and already specific.
        # The class is prepended only to disambiguate pronouns and bare terms
        # ("why is this not a monocyte"), which carry no cell name of their own.
        parts = [f"In a {label.lower()}: {question}"]
    else:
        terms = " or ".join(CONDITION_TERMS.get(label, ()))
        parts = [
            f"Clinical significance of {label.lower()}s in peripheral blood"
            + (f", and conditions causing {terms}." if terms else ".")
        ]

    if findings:
        parts.append(f"The cell shows {' and '.join(findings)}.")

    return " ".join(parts), class_filter(label)
