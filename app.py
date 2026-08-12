"""Streamlit demo UI for the Claricyte concept bottleneck model.

Quiz / given-label mode: draw a real validation-set cell, let the user guess its
type, then reveal the model's predicted morphological attributes (with confidences)
and a plain-English explanation of why those features fit the known label.

The explanation is driven by the true label, not the model's own class call. Run with:
    streamlit run app.py
(needs the local dataset under metadata/ and a checkpoint under checkpoints/).
"""

import random

import streamlit as st
import torch
from PIL import Image

from claricyte.data import MorphologyDataset
from claricyte.explain import explain
from claricyte.model import Model
from claricyte.predict import contributions, predict
from claricyte.vocab import CLASSES

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
CHECKPOINT = "checkpoints/best_model_seq.pt"

# Sidebar scope for "any cell type" quiz mode. Specific classes are study mode.
QUIZ_SCOPE = "Quiz me!"


@st.cache_resource
def load_model():
    """Load the trained CBM once and cache it across reruns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model("resnet50")
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device, weights_only=True)
    )
    model.to(device)
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
            if cols[i % 2].button(cls, use_container_width=True):
                st.session_state.guess = cls
                st.rerun()
    else:
        # Reveal phase: score the guess (quiz only), then explain the true label.
        if quiz_mode:
            guess = st.session_state.guess
            if guess == true_label:
                st.success(f"Correct: {true_label}")
            else:
                st.error(f"You guessed {guess}. It's a {true_label}.")

        result, class_dist = predict(model, image_tensor)
        scores = contributions(model, result, true_label)

        st.subheader("Explanation")
        st.write(explain(result, scores, true_label))

        st.subheader("Predicted attributes")
        st.dataframe(
            {
                "attribute": [attr.replace("_", " ") for attr in result],
                "value": [value for value, _ in result.values()],
                "confidence": [f"{conf:.0%}" for _, conf in result.values()],
            },
            hide_index=True,
        )

        # Honest reveal of the raw class head.
        with st.expander("Model internals (raw class prediction)"):
            predicted = max(class_dist, key=lambda c: class_dist[c])
            st.write(
                f"Model's own class call: **{predicted}** ({class_dist[predicted]:.0%})"
            )
            st.caption(
                "The demo explains the known-correct label, not this prediction."
            )

    st.button("Next cell", on_click=advance, args=(valset, scope))
