import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pysbd

from claricyte.rag.pmc import Article

# Surface forms indicating a passage is about a class. Matched on word boundaries,
# so "band" cannot fire on "bandwidth" or "band pass filter".
# TODO with the immature classes: "left shift", "immature granulocytes" and
# "leukocytosis" describe the myelocyte series but currently tag Band or nothing.
CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "Segmented Neutrophil": (
        "neutrophil",
        "neutrophils",
        "segmented neutrophil",
        "polymorphonuclear",
        "pmn",
        "seg neutrophil",
        "mature neutrophil",
    ),
    "Band Neutrophil": (
        "neutrophil",
        "neutrophils",
        "band neutrophil",
        "band form",
        "band cell",
        "stab cell",
        "left shift",
        "bandemia",
        "non-segmented neutrophil",
        "nonsegmented neutrophil",
    ),
    "Lymphocyte": (
        "lymphocyte",
        "lymphocytes",
        "lymphocytic",
        "large granular lymphocyte",
        "reactive lymphocyte",
        "atypical lymphocyte",
    ),
    "Monocyte": ("monocyte", "monocytes", "monocytic", "monocytoid"),
    "Eosinophil": ("eosinophil", "eosinophils", "eosinophilia"),
    "Basophil": ("basophil", "basophils", "basophilia"),
}

# Passages about smear review, staining or pre-analytical artifact are relevant to
# every class. They get their own tag, which the retriever includes alongside the
# class filter.
GENERAL = "General"
GENERAL_ALIASES: tuple[str, ...] = (
    "blood film",
    "blood smear",
    "peripheral smear",
    "leukocyte differential",
    "white blood cell differential",
    "differential count",
    "wright-giemsa",
    "romanowsky",
    "smear review",
    "film review",
    "edta",
)

_PATTERNS: dict[str, re.Pattern[str]] = {
    label: re.compile(
        r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE
    )
    for label, aliases in {**CLASS_ALIASES, GENERAL: GENERAL_ALIASES}.items()
}

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def tag_classes(text: str) -> tuple[str, ...]:
    """Classes a passage is about, plus GENERAL if it is cross-cutting.

    Empty means drop it: no cell type and no smear-level topic, so no filter
    could ever reach it.
    """
    return tuple(label for label, pattern in _PATTERNS.items() if pattern.search(text))


# What Chroma accepts as a metadata value: scalars, or arrays of scalars.
ChromaMetadata = dict[str, str | list[str]]


@dataclass(frozen=True)
class Chunk:
    text: str
    source_id: str
    section: str
    url: str
    license: str
    cell_classes: tuple[str, ...]
    title: str
    chunk_index: int

    @property
    def id(self) -> str:
        return f"{self.source_id}:{self.section}:{self.chunk_index}"

    def chroma_keys(self) -> tuple[str, str, ChromaMetadata]:
        metadata: ChromaMetadata = dict()
        metadata["source_id"] = self.source_id
        metadata["section"] = self.section
        metadata["url"] = self.url
        metadata["license"] = self.license
        metadata["cell_classes"] = list(self.cell_classes)
        metadata["title"] = self.title
        return self.id, self.text, metadata


def split_sentences(text: str) -> list[str]:
    """Sentences, stripped. A naive split on ". " breaks on "1.5 x 10^9/L" and
    on every abbreviation in a clinical paper."""
    return [s.strip() for s in _SEGMENTER.segment(text) if s.strip()]


def _overlap_cut(sentences: list[str], budget: int) -> tuple[int, int]:
    """Where to slice for the tail fitting in `budget`, and its word count.

    An index, so the caller slices the list it already has. A sentence longer
    than the budget carries nothing whole, which is what _overlap_tail is for.
    """
    total = 0
    cut = len(sentences)
    for i in range(len(sentences) - 1, -1, -1):
        words = len(sentences[i].split())
        if total + words > budget:
            break
        total += words
        cut = i
    return cut, total


def _overlap_tail(sentences: list[str], budget: int) -> tuple[list[str], int]:
    """The tail to carry into the next chunk: whole sentences if any fit,
    otherwise the last `budget` words.

    Whole sentences are preferred because a fragment starting mid-clause reads
    badly and embeds badly. But a section can be one 200-word "sentence": a
    table flattened into prose, a list of attributes with a single full stop at
    the end. Carrying nothing there put a label at the end of one chunk and its
    values at the start of the next, and the model answered with the values
    belonging to the neighbouring row. Carrying the whole sentence instead is
    not the fix, since it would spend most of the new chunk on overlap.
    """
    cut, total = _overlap_cut(sentences, budget)
    if cut < len(sentences):
        return sentences[cut:], total
    if budget <= 0:
        # Guard the slice, not just the caller: words[-0:] is the whole list,
        # so a zero budget would carry everything forward.
        return [], 0
    words = " ".join(sentences).split()[-budget:]
    return [" ".join(words)], len(words)


def chunk_section(
    text: str, max_words: int = 300, overlap_words: int = 50
) -> Iterator[str]:
    batch, total = [], 0
    # pysbd keeps terminal punctuation and a trailing space, so strip here and
    # rejoin with a plain space below.
    sentences = split_sentences(text)
    for sentence in sentences:
        if total + len(sentence.split()) > max_words and batch:
            yield " ".join(batch)
            batch, total = _overlap_tail(batch, overlap_words)
        batch.append(sentence)
        total += len(sentence.split())
    if batch:
        yield " ".join(batch)


def chunk_article(
    article: Article,
    cell_classes: tuple[str, ...],
    max_words: int = 300,
) -> list[Chunk]:
    chunks = []
    # Counted across the whole article, not restarted per section: real papers
    # repeat headings (two "Diagnostic criteria" sections, one per disease), and a
    # per-section counter gives both a chunk_index of 0 and so the same id. Chroma
    # treats a repeat id as an overwrite, so that silently drops text.
    index = 0
    for section, text in article.sections:
        for passage in chunk_section(text, max_words):
            chunks.append(
                Chunk(
                    text=passage,
                    chunk_index=index,
                    source_id=article.source_id,
                    section=section,
                    url=article.url,
                    license=article.license,
                    cell_classes=cell_classes,
                    title=article.title,
                )
            )
            index += 1
    return chunks


def write_jsonl(chunks: Iterable[Chunk], path: str | Path) -> None:
    """Write chunks as one JSON object per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Chunk]:
    """Read a corpus file back. cell_classes round-trips via list, so retuple it."""
    chunks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["cell_classes"] = tuple(row["cell_classes"])
            chunks.append(Chunk(**row))
    return chunks
