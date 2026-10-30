from types import SimpleNamespace

import pytest

from docqa.chunking import chunk_sections
from docqa.config import Settings
from docqa.ingestion import Section
from docqa.vector_store import VectorStore

POLICY_TEXT = {
    "handbook.txt": [
        "Employees receive 24 days of paid annual leave per calendar year. Unused leave of up to "
        "5 days can be carried over to the next year.",
        "Remote work is allowed up to three days per week with manager approval.",
    ],
    "travel.txt": [
        "Economy class is required for flights under six hours. Hotel bookings are capped at "
        "150 USD per night in tier-two cities.",
    ],
}


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_dir=str(tmp_path / "chroma"), min_relevance=0.2, max_agent_steps=3)


@pytest.fixture
def store(settings):
    store = VectorStore(settings.chroma_dir, "test_docs")
    for source, paragraphs in POLICY_TEXT.items():
        sections = [Section(p, {"source": source}) for p in paragraphs]
        store.replace_source(source, chunk_sections(sections, settings.chunk_size, settings.chunk_overlap))
    return store


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, stop_details=None)


class FakeLLM:
    """Stands in for ClaudeLLM: returns scripted responses and records each request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        # Snapshot the message list: the agent keeps appending to the same list object.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)
