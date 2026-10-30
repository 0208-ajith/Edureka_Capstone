"""Split document sections into overlapping chunks for embedding (Task 4)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

# Try to break on paragraphs first, then lines, sentences and finally words.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_pieces(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursively split text into pieces no longer than `size`, keeping separators."""
    if len(text) <= size:
        return [text]
    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return [text[i : i + size] for i in range(0, len(text), size)]
    remaining = separators[separators.index(sep) + 1 :]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep
        if not part:
            continue
        if len(part) > size:
            pieces.extend(_split_pieces(part, size, remaining))
        else:
            pieces.append(part)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    # Start the overlap on a word boundary so chunks don't begin mid-word.
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for piece in _split_pieces(text, chunk_size, _SEPARATORS):
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = _overlap_tail(current, overlap)
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_sections(sections: Iterable, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Chunk loader sections, carrying each section's metadata onto its chunks."""
    chunks: list[Chunk] = []
    for section in sections:
        for piece in split_text(section.text, chunk_size, overlap):
            index = len(chunks)
            metadata = {**section.metadata, "chunk_index": index}
            digest = hashlib.sha1(f"{metadata.get('source', '')}|{index}|{piece}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(id=digest[:24], text=piece, metadata=metadata))
    return chunks
