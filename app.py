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
from claricyte.rag.generate import cited
from claricyte.rag.panels import Panel, load_panels, panel_key
from claricyte.rag.pipeline import ask
from claricyte.rag.providers import OPENAI_MODEL
from claricyte.rag.query import notable_findings
from claricyte.rag.store import EMBEDDING_MODEL
from claricyte.vocab import CLASSES

ATTR_PATH, METADATA_PATH = "demo_data/attributes.csv", "demo_data/metadata.csv"
CHECKPOINT = "checkpoints/demo.pt"
CSS_PATH = "assets/claricyte.css"

# Sidebar scope for "any cell type" quiz mode. Specific classes are study mode.
QUIZ_SCOPE = "Quiz me!"

# Questions per session. Trivially bypassed by reloading, so the real backstop is
# the spend limit on the account; this is here to make casual abuse tedious.
MAX_QUESTIONS = 10

# Names the stack rather than describing it, since the audience for this line is
# as likely to be a reader of the repo as a student. Model names come from the
# modules that use them so the caption cannot drift from what actually ran.
PROVENANCE = (
    f"Retrieved from open-licensed literature by {EMBEDDING_MODEL.split('/')[-1]} "
    f"embeddings in ChromaDB, then written by OpenAI {OPENAI_MODEL}, which never "
    "sees the image. Every claim cites a retrieved source."
)


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


@st.cache_data
def load_context_panels() -> dict[str, Panel]:
    """Read the committed clinical context panels once."""
    return load_panels()


def sources_markdown(panel: Panel) -> str:
    """The panel's cited sources as a numbered list of links.

    Only the ones the answer actually cites: retrieval hands over k chunks and a
    three-sentence answer rarely uses them all, so listing the rest implies
    support that was never claimed. Numbering stays as generated, so [2] in the
    text still points at the entry labelled 2.
    """
    used = cited(panel.text)
    return "\n".join(
        f"{source.number}. [{source.title}, {source.section}]({source.url})"
        for source in panel.sources
        if source.number in used
    )


def render_panel(panel: Panel, invalid: tuple[int, ...] = ()) -> None:
    """Show an answer with its sources, precomputed or live."""
    if not panel.text:
        st.info("Nothing in the corpus matched that.")
        return
    st.write(panel.text)
    links = sources_markdown(panel)
    if links:
        with st.expander("Sources"):
            st.markdown(links)
    if invalid:
        st.warning(
            f"The model cited sources that were not retrieved: {list(invalid)}. "
            "Those claims are unsupported."
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
    # The answer belongs to the cell that was on screen, not to the next one.
    st.session_state.answer = None


st.set_page_config(page_title="Claricyte", layout="wide")
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

if not revealed:
    # Guess phase: image beside the candidate classes, nothing else to show.
    left, right = st.columns(2)
    with left:
        st.image(Image.open(row["image_path"]), caption="Mystery cell")
    with right:
        st.subheader("What type is this cell?")
        cols = st.columns(2)
        for i, cls in enumerate(CLASSES):
            if cols[i % 2].button(cls, type="primary", use_container_width=True):
                st.session_state.guess = cls
                st.rerun()
else:
    result, class_dist, concepts = predict(model, image_tensor)
    scores = contributions(model, result, concepts, true_label)
    findings = notable_findings(result)

    # Three columns: the cell, what the model read off it, and what the
    # literature says about it. The keyed container is what the stylesheet
    # scopes its wrapping rules to; it holds no nested columns of its own, so
    # those rules cannot reach anything else.
    with st.container(key="reveal"):
        cell, model_output, context = st.columns([1, 1.15, 1.35])

        with cell:
            st.image(Image.open(row["image_path"]), caption=f"True label: {true_label}")
            if quiz_mode:
                guess = st.session_state.guess
                if guess == true_label:
                    st.success(f"Correct: {true_label}")
                else:
                    st.error(
                        f"You guessed {guess}. "
                        f"It's {_article(true_label)} {true_label}."
                    )
            st.button(
                "Next cell", key="next_cell", on_click=advance, args=(valset, scope)
            )
            # Honest reveal of the raw class head.
            with st.expander("Model internals (raw class prediction)"):
                predicted = max(class_dist, key=lambda c: class_dist[c])
                st.write(
                    f"Model's own class call: **{predicted}** "
                    f"({class_dist[predicted]:.0%})"
                )
                st.caption(
                    "The demo explains the known-correct label, not this prediction."
                )

        with model_output:
            st.subheader("Explanation")
            st.write(explain(result, scores, true_label))
            st.subheader("Predicted attributes")
            st.markdown(attribute_table_html(result), unsafe_allow_html=True)

        with context:
            st.subheader("Clinical context")
            st.caption(PROVENANCE)

            panel = load_context_panels().get(panel_key(true_label, findings))
            if panel is None:
                st.info("No clinical context panel for this cell type yet.")
            else:
                render_panel(panel)

            asked = st.session_state.get("questions_asked", 0)
            st.subheader("Ask about this cell")
            question = st.text_input(
                "Question",
                # Keyed by cell, so moving on clears the box rather than carrying
                # a question about the previous cell onto this one.
                key=f"question_{index}",
                placeholder="Why is this not a monocyte?",
                label_visibility="collapsed",
                disabled=asked >= MAX_QUESTIONS,
            )

            stored = st.session_state.get("answer")
            if asked >= MAX_QUESTIONS:
                st.caption("Question limit reached for this session. Reload to reset.")
            elif question and (stored is None or stored[0] != question):
                # Streamlit reruns the whole script on every interaction, so the
                # answer is kept in session state and regenerated only when the
                # question changes.
                with st.spinner("Searching the literature..."):
                    stored = (question, ask(question, true_label, result))
                st.session_state.answer = stored
                st.session_state.questions_asked = asked + 1

            if stored:
                render_panel(stored[1].panel, stored[1].invalid)
                left_over = MAX_QUESTIONS - st.session_state.get("questions_asked", 0)
                st.caption(f"{left_over} questions left this session.")
