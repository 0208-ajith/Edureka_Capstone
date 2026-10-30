"""Input validation and output guardrails (Task 9)."""

from __future__ import annotations

import re
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS

NOT_FOUND_MESSAGE = (
    "I couldn't find information about that in the uploaded documents. "
    "Try rephrasing the question or uploading a document that covers it."
)
NO_DOCUMENTS_MESSAGE = "No documents are indexed yet. Upload and ingest a document first."
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the documents you've uploaded. "
    "Please ask something about their content."
)
UNSAFE_MESSAGE = (
    "I can't provide that response. The answer was withheld because it was "
    "flagged as unsafe by the validation step."
)

# Phrases typical of attempts to override the assistant's instructions.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|the\s+|your\s+)?(previous|prior|system)\s+(instructions|prompts?|rules)",
    r"(reveal|print|show|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+prompt|hidden\s+instructions)",
    r"you\s+are\s+now\s+(a|an|in)\s+",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CITATION_RE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")


class GuardrailViolation(ValueError):
    """Raised when user input fails validation. The message is safe to show."""


def validate_upload(filename: str, size_bytes: int, max_mb: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(SUPPORTED_EXTENSIONS)
        raise GuardrailViolation(f"{filename}: unsupported file type '{ext or 'none'}'. Allowed: {allowed}.")
    if size_bytes == 0:
        raise GuardrailViolation(f"{filename} is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise GuardrailViolation(f"{filename} is larger than the {max_mb} MB limit.")


def looks_like_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def clean_question(question: str, max_chars: int) -> str:
    """Normalise a user question and reject empty, oversized or manipulative input."""
    cleaned = _CONTROL_CHARS_RE.sub("", question or "").strip()
    if not cleaned:
        raise GuardrailViolation("Please enter a question.")
    if len(cleaned) > max_chars:
        raise GuardrailViolation(f"Questions are limited to {max_chars} characters (yours has {len(cleaned)}).")
    if looks_like_prompt_injection(cleaned):
        raise GuardrailViolation(
            "This question looks like an attempt to override the assistant's instructions, so it was blocked. "
            "Please ask about the content of your documents."
        )
    return cleaned


def strip_unknown_citations(answer: str, valid_labels: set[str]) -> str:
    """Drop citation markers like [S7] that don't refer to a retrieved passage.

    A citation to a passage that was never retrieved is a hallucination; removing it
    keeps every remaining citation traceable to real document text.
    """

    def _replace(match: re.Match) -> str:
        labels = [label.strip() for label in match.group(1).split(",")]
        kept = [label for label in labels if label in valid_labels]
        return "".join(f"[{label}]" for label in kept)

    return _CITATION_RE.sub(_replace, answer)


def cited_labels(answer: str) -> list[str]:
    """Return citation labels in the order they first appear in the answer."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        for label in match.group(1).split(","):
            label = label.strip()
            if label not in seen:
                seen.append(label)
    return seen
