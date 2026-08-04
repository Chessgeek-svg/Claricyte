"""Canonical morphological-attribute vocabulary for the concept bottleneck model.

Single source of truth for the model's attribute label space: which 11 attributes
exist, what values each can take, and the integer encoding that maps value strings
to head output indices and back. The data pipeline (encoding training targets) and
inference/explanation (decoding predictions) both import from here, so the encoding
can never drift between training and serving.

Values are copied verbatim from ``labeling/wbcatt_config.xml`` (the WB-CAtt schema),
so WB-CAtt's ``attributes.csv`` and the manually labeled ``manual_attributes.csv``
share one encoding. The config's ``unexpected_case`` flag is a QC field, not a
morphological attribute, and is deliberately excluded here.
"""

from __future__ import annotations

import pandas as pd

# attribute -> ordered list of allowed values. The list POSITION is the class
# index for that attribute's head, so ORDER IS LOAD-BEARING: appending a new value
# is safe, but reordering or inserting silently remaps every label ever encoded or
# any weights already trained
ATTRIBUTE_VOCAB: dict[str, list[str]] = {
    "cell_size": ["small", "big"],
    "cell_shape": ["round", "irregular"],
    "nucleus_shape": [
        "segmented-multilobed",
        "segmented-bilobed",
        "unsegmented-band",
        "unsegmented-round",
        "unsegmented-indented",
        "irregular",
    ],
    "nuclear_cytoplasmic_ratio": ["high", "low"],
    "chromatin_density": ["loosely", "densely"],
    "cytoplasm_vacuole": ["yes", "no"],
    "cytoplasm_texture": ["clear", "frosted"],
    "cytoplasm_colour": ["light blue", "blue", "purple blue"],
    "granule_type": ["small", "round", "coarse", "nil"],
    "granule_colour": ["pink", "red", "purple", "nil"],
    "granularity": ["yes", "no"],
}

# The 11 attribute names, in a fixed order
ATTRIBUTES: list[str] = list(ATTRIBUTE_VOCAB)

# attribute -> {value string: head index}, derived from ATTRIBUTE_VOCAB.
VALUE_TO_INDEX: dict[str, dict[str, int]] = {
    attr: {value: i for i, value in enumerate(values)}
    for attr, values in ATTRIBUTE_VOCAB.items()
}

# The current 6 classes in alphabetical order. 
# As more are added, they can be appended to the end
CLASSES: list[str] = [
    "Band Neutrophil",
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Segmented Neutrophil",
]

CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


def num_classes(attribute: str) -> int:
    """Number of output units the given attribute's classification head needs."""
    return len(ATTRIBUTE_VOCAB[attribute])


def encode(attribute: str, value: str) -> int:
    """Map an attribute value string to its head index. Raises KeyError if the
    value is not in the vocabulary: validate with find_invalid_rows first."""
    return VALUE_TO_INDEX[attribute][value]


def decode(attribute: str, index: int) -> str:
    """Map a head's predicted index back to its value string (for explanations)."""
    return ATTRIBUTE_VOCAB[attribute][index]


def find_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows whose attribute columns hold any value outside the vocab.
    Missing values (NaN) count as invalid. An empty result means every row is
    clean.
    """
    bad = pd.Series(False, index=df.index)
    for attribute, allowed in ATTRIBUTE_VOCAB.items():
        if attribute not in df.columns:
            continue
        # NaN.isin(...) is False, so missing values get flagged too
        bad |= ~df[attribute].isin(allowed)
    return df[bad]
