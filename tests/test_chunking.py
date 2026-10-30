import pytest

from docqa.chunking import chunk_sections, split_text
from docqa.ingestion import Section


def test_short_text_is_a_single_chunk():
    assert split_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n  ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 240 for c in chunks)
    # consecutive chunks share some text because of the overlap
    assert chunks[1].split()[0] in chunks[0]


def test_prefers_paragraph_boundaries():
    para_a = "Alpha sentence. " * 5
    para_b = "Beta sentence. " * 5
    chunks = split_text(f"{para_a}\n\n{para_b}", chunk_size=len(para_a) + 5, overlap=0)
    assert chunks[0].startswith("Alpha") and "Beta" not in chunks[0]
    assert chunks[-1].startswith("Beta")


def test_long_unbroken_text_is_hard_split():
    chunks = split_text("x" * 250, chunk_size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_chunk_sections_keeps_metadata_and_unique_ids():
    sections = [
        Section("Page one text. " * 30, {"source": "a.pdf", "page": 1}),
        Section("Page two text. " * 30, {"source": "a.pdf", "page": 2}),
    ]
    chunks = chunk_sections(sections, chunk_size=200, overlap=20)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
