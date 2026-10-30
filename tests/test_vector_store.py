from docqa.chunking import chunk_sections
from docqa.ingestion import Section
from docqa.retrieval import EvidenceSet, retrieve


def test_semantic_search_finds_relevant_passage(store):
    hits = store.search("How many vacation days do staff get?", k=3)
    assert hits and "24 days" in hits[0].text
    assert hits[0].metadata["source"] == "handbook.txt"
    assert hits[0].score > hits[-1].score or len(hits) == 1


def test_search_can_be_restricted_to_one_source(store):
    hits = store.search("annual leave", k=5, source="travel.txt")
    assert hits and all(h.metadata["source"] == "travel.txt" for h in hits)


def test_relevance_threshold_filters_unrelated_queries(store):
    assert retrieve(store, "recipe for chocolate cake with frosting", k=5, min_relevance=0.45) == []


def test_reingesting_replaces_old_chunks(store):
    store.replace_source("travel.txt", chunk_sections([Section("Taxis are reimbursed.", {"source": "travel.txt"})]))
    assert store.list_sources()["travel.txt"] == 1
    assert "Economy" not in " ".join(h.text for h in store.search("flights", k=5, source="travel.txt"))
    assert "Taxis" in store.search("taxi reimbursement", k=1, source="travel.txt")[0].text


def test_delete_and_reset(store):
    store.delete_source("travel.txt")
    assert "travel.txt" not in store.list_sources()
    store.reset()
    assert store.count() == 0
    assert store.search("anything") == []


def test_evidence_set_labels_are_stable(store):
    evidence = EvidenceSet()
    first = evidence.add(store.search("annual leave", k=2))
    second = evidence.add(store.search("annual leave carry over", k=2))
    assert first[0].label == "S1"
    assert {p.chunk_id for p in second} <= {p.chunk_id for p in evidence.passages}
    assert len(evidence.passages) == len({p.chunk_id for p in evidence.passages})
