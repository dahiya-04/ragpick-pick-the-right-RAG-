"""Chunking, keyword/dense/hybrid search, and gold-evidence matching across chunkers."""

from ragpick.chunking import make_chunker
from ragpick.embeddings import make_embedder
from ragpick.metrics import EvidenceMatcher, prepare_evidence, retrieval_metrics
from ragpick.retrieval import make_retriever

GOLD = ("Regulated entities shall maintain all necessary records of transactions between the RE and the customer, "
        "both domestic and international, for at least five years from the date of transaction.")


def _q(evidence, doc=None):
    q = {"id": "q1", "question": "x", "evidence": [{"text": evidence, **({"doc": doc} if doc else {})}]}
    prepare_evidence([q])
    return q


def test_fixed_chunker_respects_size_and_overlap(docs):
    chunks = make_chunker({"name": "f", "type": "fixed", "size": 30, "overlap": 10}, {}).chunk_corpus(docs)
    assert all(len(c.text.split()) <= 30 for c in chunks)
    first, second = [c for c in chunks if c.doc_id == "kyc_2016"][:2]
    assert first.text.split()[-10:] == second.text.split()[:10]


def test_structure_chunker_labels_numbered_paragraphs(docs):
    chunks = make_chunker({"name": "s", "type": "structure", "max_size": 80, "min_size": 5}, {}).chunk_corpus(docs)
    sections = {c.meta["section"] for c in chunks if c.doc_id == "kyc_2016"}
    assert {"2", "2.2", "3"} <= sections  # a short heading ("2. Record keeping") merges into the para after it


def test_gold_evidence_is_found_whatever_the_chunker(docs):
    q = _q(GOLD)
    for spec in ({"name": "f", "type": "fixed", "size": 40, "overlap": 10},
                 {"name": "s", "type": "structure", "max_size": 80, "min_size": 5},
                 {"name": "r", "type": "recursive", "size": 60}):
        matcher = EvidenceMatcher(make_chunker(spec, {}).chunk_corpus(docs))
        assert matcher.findable(q["evidence"][0]), spec["name"]


def test_doc_restriction_blocks_matches_in_other_documents(docs):
    chunks = make_chunker({"name": "s", "type": "structure", "max_size": 80, "min_size": 5}, {}).chunk_corpus(docs)
    matcher = EvidenceMatcher(chunks)
    assert matcher.findable(_q(GOLD, doc="kyc_2016.pdf")["evidence"][0])
    assert not matcher.findable(_q(GOLD, doc="cards_2022")["evidence"][0])


def test_recall_and_mrr():
    from ragpick.chunking import Chunk

    chunks = [Chunk(f"c{i}", "d", t, t, {}) for i, t in enumerate(
        ["unrelated text about payment systems and settlement cycles in banks", GOLD, "more unrelated words here"])]
    m = retrieval_metrics([0, 1, 2], _q(GOLD), EvidenceMatcher(chunks), [1, 3])
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["mrr"] == 0.5


def test_bm25_matches_identifiers_and_hybrid_returns_results(docs, tmp_path):
    chunks = make_chunker({"name": "s", "type": "structure", "max_size": 80, "min_size": 5}, {}).chunk_corpus(docs)
    bm25 = make_retriever({"type": "bm25"}, {})
    bm25.index(chunks)
    top = chunks[bm25.search("What does RBI/2022-23/92 say?", 1)[0][0]]
    assert "RBI/2022-23/92" in top.text

    emb = {"h": make_embedder({"type": "hashing"})}
    hybrid = make_retriever({"type": "hybrid", "embedder": "h"}, emb)
    hybrid.index(chunks, tmp_path)
    hits = hybrid.search("penalty per day for late credit card closure", 3)
    assert any("penalty" in chunks[i].text for i, _ in hits)  # hashing embedder is crude; top 3 is enough
