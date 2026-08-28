"""Tests for the retrieval corpus schema, chunker and class tagger.

corpus is pure, network-free and torch-free precisely so it can be tested like
this, the same way ``claricyte.explain`` is. The properties that matter are the
ones a silent failure would destroy downstream:

  - chunk ids are unique, because Chroma treats a repeat id as an overwrite and
    would drop the colliding text without erroring
  - chroma_keys emits only types Chroma accepts as metadata
  - the tagger does not fire on words that merely contain a cell name, since a
    false tag makes a chunk reachable under a filter it has no business under
  - a corpus survives the jsonl round trip unchanged
"""

import pytest

from claricyte.rag.corpus import (
    GENERAL,
    Chunk,
    _overlap_cut,
    chunk_article,
    chunk_section,
    read_jsonl,
    tag_classes,
    write_jsonl,
)
from claricyte.rag.pmc import Article


def make_chunk(**overrides) -> Chunk:
    """A fully populated chunk, with overrides applied."""
    base = dict(
        text="Basophil granules are water soluble and may wash out.",
        pmcid="PMC7563270",
        section="Morphology",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7563270/",
        license="CC BY",
        cell_classes=("Basophil",),
        title="Basophils in health and disease",
        chunk_index=0,
    )
    base.update(overrides)
    return Chunk(**base)


def sentences(count: int, words: int = 5) -> str:
    """`count` distinct sentences of roughly `words` words each."""
    return " ".join(
        f"Sentence {i} " + "filler " * (words - 2) + "end." for i in range(count)
    )


# --- schema ---------------------------------------------------------------


def test_id_combines_article_section_and_position():
    assert make_chunk(chunk_index=2).id == "PMC7563270:Morphology:2"


def test_ids_differ_within_a_section():
    """The case that matters: several chunks from one section must not collide."""
    ids = {make_chunk(chunk_index=i).id for i in range(3)}
    assert len(ids) == 3


def test_chunk_is_frozen():
    with pytest.raises(Exception):
        make_chunk().text = "reassigned"


def test_chroma_keys_returns_id_text_metadata():
    chunk = make_chunk()
    chunk_id, text, metadata = chunk.chroma_keys()
    assert chunk_id == chunk.id
    assert text == chunk.text
    assert metadata["pmcid"] == "PMC7563270"
    assert metadata["title"] == chunk.title  # citation needs link text, not just url


def test_chroma_metadata_holds_only_types_chroma_accepts():
    """Chroma takes scalars or arrays of scalars. A tuple is neither."""
    _, _, metadata = make_chunk().chroma_keys()
    for value in metadata.values():
        assert isinstance(value, (str, int, float, bool, list))
        if isinstance(value, list):
            assert all(isinstance(item, str) for item in value)


# --- chunking -------------------------------------------------------------


def test_short_text_is_one_chunk():
    assert len(list(chunk_section(sentences(3), max_words=300))) == 1


def test_chunks_respect_max_words_when_sentences_are_small():
    chunks = list(chunk_section(sentences(60, words=5), max_words=40, overlap_words=10))
    assert len(chunks) > 1
    assert all(len(c.split()) <= 40 for c in chunks)


def test_no_source_sentence_is_lost():
    text = sentences(40)
    joined = " ".join(chunk_section(text, max_words=50, overlap_words=10))
    for i in range(40):
        assert f"Sentence {i} " in joined


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    chunks = list(chunk_section(sentences(40, words=5), max_words=40, overlap_words=15))
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail in chunks[1]


def test_zero_overlap_produces_disjoint_chunks():
    chunks = list(chunk_section(sentences(40, words=5), max_words=40, overlap_words=0))
    total = sum(len(c.split()) for c in chunks)
    assert total == len(sentences(40, words=5).split())


def test_overlap_cut_stops_at_the_budget():
    cut, carried = _overlap_cut(["a b c", "d e", "f"], budget=3)
    assert carried <= 3
    assert cut >= 1


def test_oversized_sentence_carries_nothing():
    """The bug this replaced: a single long sentence duplicated itself forward."""
    cut, carried = _overlap_cut(["word " * 100], budget=10)
    assert (cut, carried) == (1, 0)


def make_article(sections) -> Article:
    return Article(
        pmcid="PMC1", title="T", url="u", license="CC BY", sections=tuple(sections)
    )


def test_chunk_article_stamps_provenance_on_every_chunk():
    article = make_article([("Intro", sentences(3)), ("Methods", sentences(3))])
    chunks = chunk_article(article, cell_classes=("Basophil",))
    assert {c.pmcid for c in chunks} == {"PMC1"}
    assert {c.section for c in chunks} == {"Intro", "Methods"}
    assert all(c.cell_classes == ("Basophil",) for c in chunks)


def test_repeated_section_headings_still_yield_unique_ids():
    """Real papers repeat headings: two "Diagnostic criteria" sections, one per
    disease. A per-section counter gave both chunk_index 0 and so the same id, and
    Chroma treats a repeat id as an overwrite rather than an error."""
    article = make_article(
        [("Diagnostic criteria", sentences(3)), ("Diagnostic criteria", sentences(3))]
    )
    chunks = chunk_article(article, cell_classes=())
    assert len({c.id for c in chunks}) == len(chunks)


# --- tagging --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Band pass filters are used in the optical bench.",  # 'band'
        "Basophilic stippling is seen in lead poisoning.",  # staining adjective
        "Eosinophilic cytoplasm was noted in the hepatocytes.",  # staining adjective
        "The patient was discharged on Tuesday.",
    ],
)
def test_words_that_merely_contain_a_cell_name_do_not_tag(text):
    assert tag_classes(text) == ()


def test_bare_neutrophil_tags_both_neutrophil_classes():
    """Deliberate recall-over-precision call: literature rarely qualifies it."""
    tags = tag_classes("The neutrophil shows toxic granulation.")
    assert "Segmented Neutrophil" in tags
    assert "Band Neutrophil" in tags


def test_cross_cutting_text_gets_the_general_tag():
    tags = tag_classes("Prolonged EDTA storage degrades cells on the blood film.")
    assert tags == (GENERAL,)


def test_specific_cell_names_tag_only_their_class():
    assert tag_classes("Monocytes show a folded nucleus.") == ("Monocyte",)


# --- persistence ----------------------------------------------------------


def test_jsonl_round_trip_preserves_chunks(tmp_path):
    original = [make_chunk(chunk_index=i) for i in range(3)]
    path = tmp_path / "corpus.jsonl"
    write_jsonl(original, path)
    assert read_jsonl(path) == original


def test_jsonl_round_trip_keeps_cell_classes_a_tuple(tmp_path):
    """JSON has no tuple, so this would silently come back a list and break
    equality and hashing everywhere downstream."""
    path = tmp_path / "corpus.jsonl"
    write_jsonl([make_chunk()], path)
    assert isinstance(read_jsonl(path)[0].cell_classes, tuple)
