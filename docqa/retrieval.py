"""Similarity search with a relevance threshold and citation labels (Task 6)."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from .guardrails import cited_labels
from .vector_store import SearchHit, VectorStore


@dataclass
class Passage:
    label: str  # citation label shown to the LLM and the user, e.g. "S3"
    text: str
    source: str
    location: str
    score: float
    chunk_id: str


def describe_location(metadata: dict) -> str:
    if "page" in metadata:
        return f"page {metadata['page']}"
    rows = metadata.get("rows")
    if rows == "summary":
        rows_text = "table summary"
    elif rows:
        rows_text = f"rows {rows}"
    else:
        rows_text = ""
    if "sheet" in metadata:
        return f"sheet '{metadata['sheet']}'" + (f", {rows_text}" if rows_text else "")
    if rows_text:
        return rows_text
    return f"chunk {int(metadata.get('chunk_index', 0)) + 1}"


def retrieve(
    store: VectorStore, query: str, k: int, min_relevance: float, source: str | None = None
) -> list[SearchHit]:
    """Return the top-k hits whose similarity clears the relevance threshold."""
    return [hit for hit in store.search(query, k=k, source=source) if hit.score >= min_relevance]


class EvidenceSet:
    """Passages gathered while answering one question, labelled S1, S2, ...

    The same chunk retrieved by several searches keeps its first label, so
    citations stay stable across an agent's tool calls.
    """

    def __init__(self) -> None:
        self._by_chunk: dict[str, Passage] = {}

    def add(self, hits: list[SearchHit]) -> list[Passage]:
        passages = []
        for hit in hits:
            passage = self._by_chunk.get(hit.chunk_id)
            if passage is None:
                passage = Passage(
                    label=f"S{len(self._by_chunk) + 1}",
                    text=hit.text,
                    source=hit.metadata.get("source", "unknown"),
                    location=describe_location(hit.metadata),
                    score=round(hit.score, 3),
                    chunk_id=hit.chunk_id,
                )
                self._by_chunk[hit.chunk_id] = passage
            passages.append(passage)
        return passages

    @property
    def passages(self) -> list[Passage]:
        return list(self._by_chunk.values())

    @property
    def labels(self) -> set[str]:
        return {p.label for p in self._by_chunk.values()}

    def cited_in(self, answer: str) -> list[Passage]:
        by_label = {p.label: p for p in self._by_chunk.values()}
        return [by_label[label] for label in cited_labels(answer) if label in by_label]


def format_passages(passages: list[Passage]) -> str:
    """Wrap passages in tags so the model can tell document data from instructions."""
    return "\n\n".join(
        f'<passage id="{p.label}" source="{escape(p.source)}" location="{escape(p.location)}" '
        f'relevance="{p.score:.2f}">\n{p.text}\n</passage>'
        for p in passages
    )
