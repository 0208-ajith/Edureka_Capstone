"""Persistent vector knowledge store backed by ChromaDB (Task 5).

Embeddings come from ChromaDB's built-in all-MiniLM-L6-v2 ONNX model, which runs
locally on CPU, so no separate embeddings API key is needed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_BATCH_SIZE = 256


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict
    score: float  # cosine similarity, higher is more relevant


class VectorStore:
    def __init__(self, path: str, collection: str = "enterprise_docs"):
        self._client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
        self._embedding_fn = DefaultEmbeddingFunction()
        self._name = collection
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            name=self._name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def replace_source(self, source: str, chunks: list) -> None:
        """Index a document's chunks, removing any chunks from an earlier upload of it."""
        self.delete_source(source)
        for start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[start : start + _BATCH_SIZE]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[SearchHit]:
        total = self.count()
        if total == 0 or not query.strip():
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, total),
            where={"source": source} if source else None,
        )
        return [
            SearchHit(chunk_id=cid, text=doc, metadata=meta or {}, score=1.0 - dist)
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def list_sources(self) -> dict[str, int]:
        """Return {document name: chunk count} for every indexed document."""
        if self.count() == 0:
            return {}
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return dict(sorted(Counter(m.get("source", "unknown") for m in metadatas).items()))

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open()
