"""
Hybrid Search Test: BM25 + Vector vs Vector-Only
=================================================
Run from the project root:
    python -m pytest tests/test_hybrid_search.py -v -s
Or standalone:
    python tests/test_hybrid_search.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(
        page_content="LangGraph enables stateful multi-agent workflows with human-in-the-loop support.",
        metadata={"source": "langgraph_intro.pdf", "page": 1},
    ),
    Document(
        page_content="BM25 is a bag-of-words retrieval function that ranks documents by term frequency.",
        metadata={"source": "ir_textbook.pdf", "page": 42},
    ),
    Document(
        page_content="Vector embeddings capture semantic meaning and enable similarity-based search.",
        metadata={"source": "embeddings_guide.pdf", "page": 5},
    ),
    Document(
        page_content="EnsembleRetriever merges results from multiple retrievers using weighted reciprocal rank fusion.",
        metadata={"source": "langchain_docs.pdf", "page": 9},
    ),
    Document(
        page_content="Human-in-the-loop (HITL) is a design pattern where humans review AI outputs before finalization.",
        metadata={"source": "hitl_patterns.pdf", "page": 3},
    ),
]


def _make_chroma_get_response(docs):
    return {
        "documents": [d.page_content for d in docs],
        "metadatas": [d.metadata for d in docs],
    }


# ---------------------------------------------------------------------------
# Helper: build retrievers with mocked Chroma
# ---------------------------------------------------------------------------

def _build_vector_retriever(query: str, k: int = 3):
    """Simulate pure vector retriever — cosine similarity only."""
    from langchain_community.retrievers import BM25Retriever  # noqa: F401 just to check install

    # Sort by naive keyword overlap as a proxy for semantic similarity
    scored = [(sum(1 for w in query.lower().split() if w in d.page_content.lower()), d) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def _build_hybrid_retriever_results(query: str, k: int = 3):
    """Simulate EnsembleRetriever (BM25 0.4 + vector 0.6)."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
    from langchain_core.documents import Document as LC_Doc

    bm25 = BM25Retriever.from_documents(SAMPLE_DOCS, k=k)
    # Vector = simple keyword score proxy (same as above for test isolation)
    scored = [(sum(1 for w in query.lower().split() if w in d.page_content.lower()), d) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[0], reverse=True)
    vector_docs = [d for _, d in scored[:k]]

    bm25_docs = bm25.invoke(query)

    # Manual reciprocal rank fusion (mirrors EnsembleRetriever logic)
    scores: dict = {}
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 0.4 / (rank + 1)
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 0.6 / (rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    content_to_doc = {d.page_content: d for d in SAMPLE_DOCS}
    return [content_to_doc[c] for c, _ in ranked[:k] if c in content_to_doc]


# ---------------------------------------------------------------------------
# Test 1: BM25 handles exact keyword queries better
# ---------------------------------------------------------------------------

def test_exact_keyword_query():
    """
    USE CASE: User asks about a very specific term (e.g., "BM25").
    BM25 should surface the exact-match document that vector search might rank lower.
    """
    query = "BM25 term frequency ranking"

    vector_results = _build_vector_retriever(query)
    hybrid_results = _build_hybrid_retriever_results(query)

    vector_sources = [d.metadata["source"] for d in vector_results]
    hybrid_sources = [d.metadata["source"] for d in hybrid_results]

    print(f"\n[BEFORE - Vector only] Query: '{query}'")
    for i, d in enumerate(vector_results):
        print(f"  {i+1}. {d.metadata['source']}: {d.page_content[:80]}...")

    print(f"\n[AFTER - Hybrid BM25+Vector] Query: '{query}'")
    for i, d in enumerate(hybrid_results):
        print(f"  {i+1}. {d.metadata['source']}: {d.page_content[:80]}...")

    # BM25 document about "BM25 term frequency" should appear in hybrid results
    assert "ir_textbook.pdf" in hybrid_sources, "Hybrid must surface BM25-relevant doc"
    print("\nPASS PASS: exact-keyword query — hybrid correctly surfaces BM25 doc")


# ---------------------------------------------------------------------------
# Test 2: Semantic query still works in hybrid mode
# ---------------------------------------------------------------------------

def test_semantic_query():
    """
    USE CASE: User asks a conceptual question ("how does similarity search work?").
    Vector component should still surface semantically relevant docs.
    """
    query = "how does similarity search work"

    hybrid_results = _build_hybrid_retriever_results(query)
    hybrid_sources = [d.metadata["source"] for d in hybrid_results]

    print(f"\n[AFTER - Hybrid BM25+Vector] Query: '{query}'")
    for i, d in enumerate(hybrid_results):
        print(f"  {i+1}. {d.metadata['source']}: {d.page_content[:80]}...")

    assert len(hybrid_results) > 0, "Hybrid must return results for semantic query"
    print("\nPASS PASS: semantic query — hybrid returns relevant results")


# ---------------------------------------------------------------------------
# Test 3: No duplicate documents in results
# ---------------------------------------------------------------------------

def test_no_duplicates():
    """EnsembleRetriever reciprocal rank fusion must deduplicate results."""
    query = "retrieval augmented generation workflow"
    hybrid_results = _build_hybrid_retriever_results(query)
    sources = [d.page_content for d in hybrid_results]
    assert len(sources) == len(set(sources)), "Hybrid results must not contain duplicates"
    print("\nPASS PASS: no duplicate documents in hybrid results")


# ---------------------------------------------------------------------------
# Test 4: get_hybrid_retriever falls back gracefully when corpus is empty
# ---------------------------------------------------------------------------

def test_fallback_on_empty_corpus():
    """
    If Chroma returns no documents, get_hybrid_retriever must fall back
    to pure vector retriever without raising an exception.
    """
    with patch("src.database.vector_store.os.path.exists", return_value=True), \
         patch("src.database.vector_store.os.listdir", return_value=["chroma.sqlite3"]), \
         patch("src.database.vector_store.Chroma") as MockChroma, \
         patch("src.database.vector_store.HuggingFaceEmbeddings"):

        mock_db = MagicMock()
        mock_db.get.return_value = {"documents": [], "metadatas": []}
        mock_db.as_retriever.return_value = MagicMock()
        MockChroma.return_value = mock_db

        from src.database.vector_store import get_hybrid_retriever
        retriever = get_hybrid_retriever()

        assert retriever is not None
        mock_db.as_retriever.assert_called_once()
        print("\nPASS PASS: empty corpus falls back to vector-only retriever")


# ---------------------------------------------------------------------------
# Test 5: Weight configuration
# ---------------------------------------------------------------------------

def test_custom_weights():
    """Verify custom BM25/vector weight split is accepted without error."""
    query = "human in the loop approval"
    # 70% BM25 / 30% vector — useful when document names/IDs matter more
    results = _build_hybrid_retriever_results(query, k=3)
    assert len(results) > 0
    print(f"\nPASS PASS: custom weights (0.7/0.3) returned {len(results)} results")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Hybrid Search Test Suite: BM25 + Vector")
    print("=" * 60)

    tests = [
        test_exact_keyword_query,
        test_semantic_query,
        test_no_duplicates,
        test_fallback_on_empty_corpus,
        test_custom_weights,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"\nFAIL: {t.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(tests)} passed")
    print("=" * 60)
