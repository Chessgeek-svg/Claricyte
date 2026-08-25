"""Streamlit demo UI for the Claricyte concept bottleneck model.

Quiz / given-label mode: draw a real validation-set cell, let the user guess its
type, then reveal the model's predicted morphological attributes (with confidences)
and a plain-English explanation of why those features fit the known label.

The explanation is driven by the true label, not the model's own class call. Run with:
    streamlit run app.py

Reads the self-contained package under demo_data/ (the validation split plus
cut-down CSVs, built by scripts/make_demo_data.py) rather than the full corpus, so
a fresh clone runs without acquiring any datasets.
"""

import random
from html import escape

import streamlit as st
import torch
from PIL import Image

from claricyte.data import MorphologyDataset
from claricyte.explain import LOW_CONFIDENCE, _article, explain
from claricyte.model import Model
from claricyte.predict import contributions, predict
from claricyte.vocab import CLASSES

ATTR_PATH, METADATA_PATH = "demo_data/attributes.csv", "demo_data/metadata.csv"
CHECKPOINT = "checkpoints/demo.pt"
CSS_PATH = "assets/claricyte.css"

# Sidebar scope for "any cell type" quiz mode. Specific classes are study mode.
QUIZ_SCOPE = "Quiz me!"


@st.cache_data
def load_css() -> str:
    """Read the demo stylesheet once and cache it across reruns."""
    with open(CSS_PATH, encoding="utf-8") as handle:
        return handle.read()


def attribute_table_html(result: dict[str, tuple[str, float]]) -> str:
    """Render the predicted attributes as a bordered HTML table.

    Rows whose confidence falls below the threshold the explanation hedges at are
    flagged, so the table and the paragraph agree about what is uncertain.
    """
    rows = []
    for attribute, (value, confidence) in result.items():
        hedged = ' class="hedged"' if confidence < LOW_CONFIDENCE else ""
        rows.append(
            f"<tr{hedged}>"
            f"<td>{escape(attribute.replace('_', ' '))}</td>"
            f"<td>{escape(str(value))}</td>"
            f'<td class="confidence">{confidence:.0%}</td>'
            "</tr>"
        )
    return (
        '<table class="claricyte-attrs"><thead><tr>'
        "<th>Attribute</th><th>Value</th>"
        '<th class="confidence">Conf.</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


@st.cache_resource
def load_model():
    """Load the trained CBM once and cache it across reruns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model.from_checkpoint(CHECKPOINT, device=device)
    model.eval()
    return model


@st.cache_resource
def load_valset():
    """Load the validation split once and cache it across reruns."""
    return MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")


def pick_index(valset, scope):
    """Random dataframe index of a val cell within `scope` (a class, or any)."""
    labels = valset.df["claricyte_label"]
    if scope == QUIZ_SCOPE:
        candidates = labels.index.tolist()
    else:
        candidates = labels.index[labels == scope].tolist()
    return random.choice(candidates)


def advance(valset, scope):
    """Draw a fresh cell within scope and reset the guess phase."""
    st.session_state.index = pick_index(valset, scope)
    st.session_state.guess = None


st.set_page_config(page_title="Claricyte", layout="centered")
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

model = load_model()
valset = load_valset()

st.title("Claricyte WBC morphology tutor")

# Sidebar: quiz on any cell, or browse a specific type.
scope = st.sidebar.selectbox("Mode", [QUIZ_SCOPE, *CLASSES])

# Draw a new cell when first loading or when the mode changes.
if "index" not in st.session_state or st.session_state.get("scope") != scope:
    st.session_state.scope = scope
    advance(valset, scope)

index = st.session_state.index
row = valset.df.iloc[index]
image_tensor, _, _ = valset[index]
true_label = row["claricyte_label"]

quiz_mode = scope == QUIZ_SCOPE
# In quiz mode the label stays hidden until the user commits a guess.
revealed = (not quiz_mode) or st.session_state.guess is not None

left, right = st.columns(2)

with left:
    caption = f"True label: {true_label}" if revealed else "Mystery cell"
    st.image(Image.open(row["image_path"]), caption=caption)

with right:
    if not revealed:
        # Guess phase: one click per candidate class commits the guess.
        st.subheader("What type is this cell?")
        cols = st.columns(2)
        for i, cls in enumerate(CLASSES):
            if cols[i % 2].button(cls, type="primary", use_container_width=True):
                st.session_state.guess = cls
                st.rerun()
    else:
        # Reveal phase: score the guess (quiz only), then explain the true label.
        if quiz_mode:
            guess = st.session_state.guess
            if guess == true_label:
                st.success(f"Correct: {true_label}")
            else:
                article = _article(true_label)
                st.error(f"You guessed {guess}. It's {article} {true_label}.")

        result, class_dist, concepts = predict(model, image_tensor)
        scores = contributions(model, result, concepts, true_label)

        st.subheader("Explanation")
        st.write(explain(result, scores, true_label))

        st.subheader("Predicted attributes")
        st.markdown(attribute_table_html(result), unsafe_allow_html=True)

        # Honest reveal of the raw class head.
        with st.expander("Model internals (raw class prediction)"):
            predicted = max(class_dist, key=lambda c: class_dist[c])
            st.write(
                f"Model's own class call: **{predicted}** ({class_dist[predicted]:.0%})"
            )
            st.caption(
                "The demo explains the known-correct label, not this prediction."
            )

    st.button("Next cell", key="next_cell", on_click=advance, args=(valset, scope))
