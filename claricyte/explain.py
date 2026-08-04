"""Template-based explanation engine for the concept bottleneck model.

Pure, model-free, torch-free: given the attributes the model predicted for one
cell (value + confidence), the per-attribute contribution scores toward a label,
and that label, it produces an English explanation of the morphological features
supporting the classification.

Faithful by construction — it can only speak from the bottleneck's own outputs
(the model's per-image predictions), never a class prototype. Quiz / given-label
mode: the label is the known-correct answer, and the text explains why the observed
features fit it.

Two layers pick what to say:
  - Set A (REQUIRED_FEATURES): clinically inalienable features always stated; if the
    model's prediction violates one, the contradiction is surfaced. In theory the model
    should never violate these, but in practice it sometimes does, and the explanation
    should flag that.
  - Set B: discretionary features ranked by the model's own contribution score
    (from predict.contributions), top few mentioned.
"""

from __future__ import annotations

# (attribute, value) -> English fragment. Mirrors vocab.ATTRIBUTE_VOCAB. The granule
# cluster (granularity / granule_type / granule_colour) is rendered together by
# _describe_granules, so its fragments live there
PHRASING: dict[str, dict[str, str]] = {
    "cell_size": {"small": "a small cell", "big": "a large cell"},
    "cell_shape": {"round": "a round shape", "irregular": "an irregular shape"},
    "nucleus_shape": {
        "segmented-multilobed": "a multilobed nucleus",
        "segmented-bilobed": "a bilobed nucleus",
        "unsegmented-band": "a band-shaped nucleus",
        "unsegmented-round": "a round nucleus",
        "unsegmented-indented": "an indented nucleus",
        "irregular": "an irregular nucleus",
    },
    "nuclear_cytoplasmic_ratio": {
        "high": "a high nuclear-to-cytoplasmic ratio",
        "low": "a low nuclear-to-cytoplasmic ratio",
    },
    "chromatin_density": {
        "loosely": "loosely packed chromatin",
        "densely": "densely packed chromatin",
    },
    "cytoplasm_vacuole": {"yes": "cytoplasmic vacuoles", "no": "no vacuoles"},
    "cytoplasm_texture": {"clear": "clear cytoplasm", "frosted": "frosted cytoplasm"},
    "cytoplasm_colour": {
        "light blue": "light blue cytoplasm",
        "blue": "blue cytoplasm",
        "purple blue": "purple-blue cytoplasm",
    },
}

# Set A. Inalienable features per class: {class: {attribute: {acceptable values}}}.
# If the model's predicted value is in the acceptable set it confirms the label; if
# not, that contradiction is surfaced as an explicit atypical-finding note.
#
# Values that aren't a simple equality (e.g. band vs. seg nucleus is a judgment call) 
# are given as an acceptable *set*, not a single value. Empty {} = no strictly mandatory
# single feature.
REQUIRED_FEATURES: dict[str, dict[str, set[str]]] = {
    "Basophil": {"granularity": {"yes"}, "granule_colour": {"purple"}},
    "Eosinophil": {"granularity": {"yes"}, "granule_colour": {"red", "pink"}},
    "Band Neutrophil": {"nucleus_shape": {"unsegmented-band", "unsegmented-indented"}},
    "Segmented Neutrophil": {},
    "Lymphocyte": {},
    "Monocyte": {},
}

# The three attributes folded into one granule clause.
GRANULE_ATTRS = {"granularity", "granule_type", "granule_colour"}

LOW_CONFIDENCE = 0.6  # below this, a mentioned feature is hedged
MAX_DISCRETIONARY = 3  # cap on Set B features per explanation


def _article(word: str) -> str:
    """'a' or 'an' for the following word (crude vowel check, good enough here)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _readable(attr: str) -> str:
    """Attribute key -> human phrase, e.g. 'nucleus_shape' -> 'nucleus shape'."""
    return attr.replace("_", " ")


def _join(items: list[str]) -> str:
    """Comma-join with an Oxford 'and': [a, b, c] -> 'a, b, and c'."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _describe_granules(result: dict[str, tuple[str, float]]) -> str | None:
    """Render the granularity/type/colour cluster as one fragment, or None.

    Folds the three granule attributes into a single clause ('coarse purple
    granules', 'no granules') so we never emit them as three separate features.
    Returns None if granularity wasn't predicted.
    """
    if "granularity" not in result:
        return None
    if result["granularity"][0] == "no":
        return "no granules"
    # granularity == "yes": prepend type then colour when present (skip nil).
    parts = []
    gtype = result.get("granule_type", (None,))[0]
    gcolour = result.get("granule_colour", (None,))[0]
    if gtype and gtype != "nil":
        parts.append(gtype)
    if gcolour and gcolour != "nil":
        parts.append(gcolour)
    parts.append("granules")
    return " ".join(parts)


def explain(
    result: dict[str, tuple[str, float]],
    contributions: dict[str, float],
    label: str,
    low_confidence: float = LOW_CONFIDENCE,
) -> str:
    """Explain why `label` fits the features the model observed on one cell.

    Args:
        result: {attr: (value, confidence)} from predict.predict.
        contributions: {attr: signed_float} from predict.contributions, toward `label`.
        label: the given (known-correct) class name to explain.
        low_confidence: threshold below which a mentioned feature is hedged.

    Returns:
        A plain-English paragraph, faithful to the model's predictions.
    """
    required = REQUIRED_FEATURES.get(label, {})

    included: list[str] = []  # attrs to describe, in display order
    contradictions: list[str] = []  # required attrs whose predicted value is off-label

    # 1. Required (Set A): always mention; route off-label values to contradictions.
    for attr, acceptable in required.items():
        if attr not in result:
            continue
        if result[attr][0] in acceptable:
            included.append(attr)
        else:
            contradictions.append(attr)

    # 2. Discretionary (Set B): positive contributors the model leaned on, top few.
    ranked = sorted(
        (a for a in contributions if contributions[a] > 0 and a not in required),
        key=lambda a: contributions[a],
        reverse=True,
    )
    for attr in ranked[:MAX_DISCRETIONARY]:
        included.append(attr)

    # Build fragments, folding the granule cluster and collecting low-conf hedges.
    fragments: list[str] = []
    hedged: list[str] = []
    granules_done = False
    for attr in included:
        value, confidence = result[attr]
        if attr in GRANULE_ATTRS:
            if not granules_done:
                granules_done = True
                frag = _describe_granules(result)
                if frag is not None:
                    fragments.append(frag)
            continue
        if attr not in PHRASING or value not in PHRASING[attr]:
            continue
        fragments.append(PHRASING[attr][value])
        if confidence < low_confidence:
            hedged.append(attr)

    label_lower = label.lower()
    article = _article(label_lower)
    if fragments:
        sentences = [f"Consistent with {article} {label_lower}: {_join(fragments)}."]
    else:
        sentences = [f"No distinguishing features were identified for {label_lower}."]

    if hedged:
        names = ", ".join(_readable(a) for a in hedged)
        sentences.append(f"Confidence was low for {names}.")

    for attr in contradictions:
        value = result[attr][0]
        sentences.append(
            f"Note: {_readable(attr)} was read as '{value}', "
            f"which is atypical for {_article(label_lower)} {label_lower}."
        )

    return " ".join(sentences)
