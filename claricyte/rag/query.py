"""Turn a model prediction into a retrieval query.

Pure and torch-free, like ``corpus`` and ``explain``: takes the dict
``predict.predict`` returns and produces the two things ``store.search`` needs,
a string to embed and a metadata filter to narrow on first.

Two design decisions worth stating, because both are load-bearing.

The class is a hard filter, not a search term. It is a known categorical, so
asking the embedding to infer it wastes the one fact we are certain about.

The query is prose, not a list of attribute values. Embedding
"basophil, coarse purple granules" against clinical paragraphs compares text in
two different registers and retrieves badly. Templating into a sentence puts the
query in the same register as the corpus. Same insight as HyDE, minus the
generation step.

Attributes mostly do NOT appear in the query. For clinical retrieval the class
name carries almost all the signal, and a cell's normal defining features are
uninformative: every neutrophil has pink granules, so saying so narrows nothing.
The exception is a finding that is abnormal and clinically associated, which in
the current 11-attribute vocabulary means vacuolation and nothing else.
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
    """Chroma filter matching chunks about `label`, plus cross-cutting ones.

    GENERAL is always included: smear review and pre-analytical passages name no
    cell type, so a strict class filter would make them permanently unreachable.
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
        question: a user's free-text question, if any. Without one this builds
            the standing clinical-context query for the class.
        low_confidence: threshold below which a finding is ignored.

    Returns:
        The string to embed, and a Chroma `where` filter.
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
