"""Selectable chunking strategies, backed by langchain_text_splitters.

Every strategy receives the chosen chunk_size / chunk_overlap and returns
LangChain Documents that preserve the loader metadata (page numbers etc.).
Shared guarantees, independent of the strategy:
- empty and whitespace-only chunks are dropped;
- no chunk is ever longer than chunk_size characters;
- output is deterministic (same input -> same chunks, every time).

No extra dependencies: sentence and paragraph splitting are small regexes
that run entirely locally (no NLTK, spaCy, tiktoken, no downloads).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_STRATEGY_KEY = "recursive"


class UnknownChunkingStrategyError(ValueError):
    """Raised when a chunking strategy key is not in the registry."""


# --- shared, tiny building blocks ------------------------------------------

def _drop_empty(chunks: list[str]) -> list[str]:
    return [chunk for chunk in chunks if chunk.strip()]


def _fixed_windows(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Deterministic character windows, sliding by (size - overlap).

    Used both as the "fixed" strategy and as the oversize-sentence escape
    hatch. The remainder window may be shorter than chunk_size but never
    longer; windows overlap by at most chunk_overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    step = max(1, chunk_size - chunk_overlap)
    windows: list[str] = []
    for start in range(0, len(text), step):
        window = text[start : start + chunk_size]
        if window.strip():
            windows.append(window)
        if start + chunk_size >= len(text):
            break
    return windows


def _pack_units(
    units: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    separator: str,
) -> list[str]:
    """Greedily pack whole units into chunks of at most chunk_size chars.

    Units are joined with `separator`. The chunk after a flush starts with
    the trailing units whose joined length fits within chunk_overlap; if
    even that would overflow chunk_size, the tail is trimmed from the front
    until it fits. Oversize units (longer than chunk_size alone, e.g. one
    huge sentence) are split with fixed windows instead.
    """
    sep_len = len(separator)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        extra = (sep_len + len(unit)) if current else len(unit)
        if current_len + extra <= chunk_size:
            current.append(unit)
            current_len += extra
            continue
        if current:
            chunks.append(separator.join(current))
            # Trailing units that fit inside the requested overlap...
            tail: list[str] = []
            tail_len = 0
            for item in reversed(current):
                needed = len(item) if not tail else sep_len + len(item) + tail_len
                if needed <= chunk_overlap:
                    tail.insert(0, item)
                    tail_len = needed
                else:
                    break
            # ...but never at the cost of exceeding chunk_size.
            extra = (sep_len + len(unit)) if tail else len(unit)
            while tail and tail_len + extra > chunk_size:
                dropped = tail.pop(0)
                tail_len = tail_len - (sep_len + len(dropped)) if tail else 0
            if tail:
                current = [*tail, unit]
                current_len = tail_len + extra
            else:
                current = [unit]
                current_len = len(unit)
        else:
            current = [unit]
            current_len = len(unit)
    if current:
        chunks.append(separator.join(current))
    return chunks


# --- sentence strategy -------------------------------------------------------

# A sentence ends at . ! ? or … (with closing quotes/brackets) followed by
# whitespace or the end of the text. Deliberately small and dependency-free.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)]*\s+|[.!?…]+[\"'”’)]*$")


def _sentences(text: str) -> list[str]:
    """Split text into sentences (stripped, non-empty, ordered)."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        segment = text[start : match.end()]
        if segment.strip():
            sentences.append(segment.strip())
        start = match.end()
    remainder = text[start:]
    if remainder.strip():
        sentences.append(remainder.strip())
    return sentences


class _SentenceSplitter:
    """Packs whole sentences up to chunk_size; overlong ones -> fixed windows."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [
            Document(page_content=text, metadata=dict(doc.metadata))
            for doc in documents
            for text in self.split_text(doc.page_content)
        ]

    def split_text(self, text: str) -> list[str]:
        return _drop_empty(
            _pack_units(
                _sentences(text),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=" ",
            )
        )


# --- paragraph strategy ------------------------------------------------------

def _paragraphs(text: str) -> list[str]:
    """Split on blank lines; non-empty, stripped paragraphs in order."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


class _ParagraphSplitter:
    """Packs whole paragraphs; an oversize paragraph falls back to sentences."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._sentence = _SentenceSplitter(chunk_size, chunk_overlap)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [
            Document(page_content=text, metadata=dict(doc.metadata))
            for doc in documents
            for text in self.split_text(doc.page_content)
        ]

    def split_text(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in _paragraphs(text):
            if len(paragraph) > self.chunk_size:
                # Sentence-pack the oversized paragraph (its sentences are
                # themselves capped by the sentence strategy's own fallback).
                units.extend(self._sentence.split_text(paragraph))
            else:
                units.append(paragraph)
        return _drop_empty(
            _pack_units(
                units,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n",
            )
        )


# --- fixed strategy ----------------------------------------------------------

class _FixedSizeSplitter:
    """Character windows with overlap, ignoring natural boundaries."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [
            Document(page_content=text, metadata=dict(doc.metadata))
            for doc in documents
            for text in self.split_text(doc.page_content)
        ]

    def split_text(self, text: str) -> list[str]:
        return _drop_empty(_fixed_windows(text, self.chunk_size, self.chunk_overlap))


# --- registry ----------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingStrategy:
    key: str
    label: str
    description: str
    # (chunk_size, chunk_overlap) -> splitter exposing split_documents().
    factory: Callable[[int, int], object]


STRATEGIES: dict[str, ChunkingStrategy] = {
    "recursive": ChunkingStrategy(
        key="recursive",
        label="Recursive",
        description=(
            "Splits on natural boundaries (paragraphs, sentences, words) "
            "around the target size. The original behaviour."
        ),
        factory=lambda size, overlap: RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
        ),
    ),
    "fixed": ChunkingStrategy(
        key="fixed",
        label="Fixed",
        description=(
            "Fixed-size character windows with the requested overlap, "
            "ignoring natural boundaries."
        ),
        factory=lambda size, overlap: _FixedSizeSplitter(size, overlap),
    ),
    "sentence": ChunkingStrategy(
        key="sentence",
        label="Sentence",
        description=(
            "Packs whole sentences up to the target size; a single sentence "
            "longer than the target is split with fixed windows."
        ),
        factory=lambda size, overlap: _SentenceSplitter(size, overlap),
    ),
    "paragraph": ChunkingStrategy(
        key="paragraph",
        label="Paragraph",
        description=(
            "Packs whole paragraphs up to the target size; an oversized "
            "paragraph falls back to the sentence strategy."
        ),
        factory=lambda size, overlap: _ParagraphSplitter(size, overlap),
    ),
}


def ordered_strategies() -> list[ChunkingStrategy]:
    """Strategies in a stable, display order."""
    return [STRATEGIES[key] for key in ("recursive", "fixed", "sentence", "paragraph")]


def allowed_strategy_keys() -> list[str]:
    return [strategy.key for strategy in ordered_strategies()]


def resolve_strategy(key: str) -> ChunkingStrategy:
    try:
        return STRATEGIES[key]
    except (KeyError, TypeError) as error:
        raise UnknownChunkingStrategyError(
            f"Unknown chunking strategy '{key}'. "
            f"Allowed strategies: {', '.join(allowed_strategy_keys())}"
        ) from error


def describe_strategies() -> list[dict[str, str]]:
    """Human-readable entries for GET /config."""
    return [
        {"key": s.key, "label": s.label, "description": s.description}
        for s in ordered_strategies()
    ]


def split_documents(
    documents: list[Document],
    *,
    strategy_key: str = DEFAULT_STRATEGY_KEY,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split loaded documents with the named strategy."""
    strategy = resolve_strategy(strategy_key)
    splitter = strategy.factory(chunk_size, chunk_overlap)
    # Every factory returns an object exposing split_documents().
    return splitter.split_documents(documents)  # type: ignore[attr-defined]
