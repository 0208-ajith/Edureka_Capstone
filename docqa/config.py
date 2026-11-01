"""Application settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".csv", ".xlsx")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    model: str = "openai/gpt-oss-120b"
    chroma_dir: str = "data/chroma"
    collection: str = "enterprise_docs"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    # Cosine similarity (0..1) a passage needs before it is shown to the LLM.
    min_relevance: float = 0.25
    max_file_mb: int = 20
    max_question_chars: int = 2000
    max_agent_steps: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            model=os.getenv("GROQ_MODEL", defaults.model),
            chroma_dir=os.getenv("CHROMA_DIR", defaults.chroma_dir),
            collection=os.getenv("CHROMA_COLLECTION", defaults.collection),
            chunk_size=_env_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            top_k=_env_int("TOP_K", defaults.top_k),
            min_relevance=_env_float("MIN_RELEVANCE", defaults.min_relevance),
            max_file_mb=_env_int("MAX_FILE_MB", defaults.max_file_mb),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", defaults.max_question_chars),
            max_agent_steps=_env_int("MAX_AGENT_STEPS", defaults.max_agent_steps),
        )
