"""Week 2 tests — agent layer and pipeline.

These tests use mocks for external services (Claude API, Qdrant, search APIs)
so they run without credentials or running services.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from research_machine.agents.hypothesis_agent import HypothesisAgent, HypothesisOutput
from research_machine.agents.literature_agent import (
    LiteratureAgent, PaperSummary, _normalise_title, _extract_year
)
from research_machine.agents.indexing_agent import IndexingAgent, NoveltyResult
from research_machine.agents.cost_tracker import CostTracker
from research_machine.rag.embeddings import MockEmbeddingsProvider, PaperSummaryEmbedder
from research_machine.rag.vector_store import QdrantVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_hypothesis(**kwargs) -> HypothesisOutput:
    defaults = dict(
        primary_question="How does AI tool adoption affect founder productivity in early-stage startups?",
        angles=["Cognitive load reduction", "Decision quality", "Time allocation"],
        search_queries=["AI tools founder productivity", "startup AI adoption"],
        rationale="No longitudinal empirical studies exist at early stage.",
        suggested_methodology="Survey of 300 founders + 6-month longitudinal follow-up",
        target_venues=["Journal of Business Venturing", "MIS Quarterly"],
        domain="startup",
    )
    defaults.update(kwargs)
    return HypothesisOutput(**defaults)


def make_fake_paper(**kwargs) -> dict:
    defaults = {
        "title": "AI and Startup Performance",
        "abstract": "We study the impact of AI on startup performance metrics.",
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "year": 2023,
        "citationCount": 42,
        "_source": "semantic_scholar",
        "url": "https://example.com/paper1",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

def test_normalise_title():
    assert _normalise_title("  AI & ML in 2024! ") == "ai  ml in 2024"


def test_extract_year_from_year_field():
    assert _extract_year({"year": 2023}) == 2023


def test_extract_year_from_published():
    assert _extract_year({"published": "2022-03-15T00:00:00Z"}) == 2022


def test_extract_year_missing():
    assert _extract_year({}) is None


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

def test_cost_tracker_accumulates():
    tracker = CostTracker(model="claude-haiku-4-5-20251001")

    mock_response = MagicMock()
    mock_response.usage_metadata = MagicMock(input_tokens=1000, output_tokens=500)

    tracker.record(mock_response, "hypothesis")
    tracker.record(mock_response, "hypothesis")

    assert tracker.input_tokens == 2000
    assert tracker.output_tokens == 1000
    assert tracker.calls == 2
    cost = tracker.cost_usd()
    assert cost > 0
    summary = tracker.summary()
    assert "hypothesis" in summary["per_agent"]


# ---------------------------------------------------------------------------
# MockEmbeddingsProvider + PaperSummaryEmbedder
# ---------------------------------------------------------------------------

def test_mock_embeddings_deterministic():
    provider = MockEmbeddingsProvider(dimension=64)
    v1 = provider.embed_text("hello world")
    v2 = provider.embed_text("hello world")
    assert v1 == v2
    assert len(v1) == 64


def test_paper_summary_embedder():
    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=32))
    vec = embedder.embed_paper("Some Title", "Some abstract about AI.")
    assert len(vec) == 32


def test_embedder_multiple_papers():
    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=32))
    papers = [
        {"title": "Paper A", "abstract": "About A"},
        {"title": "Paper B", "abstract": "About B"},
    ]
    vecs = embedder.embed_papers(papers)
    assert len(vecs) == 2
    assert vecs[0] != vecs[1]


# ---------------------------------------------------------------------------
# QdrantVectorStore (in-memory mock)
# ---------------------------------------------------------------------------

def test_vector_store_create_collection():
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("not found")

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client = mock_client
    store._known_collections = set()

    result = store.create_collection("test_col", vector_size=32)
    assert result is True
    mock_client.create_collection.assert_called_once()


def test_vector_store_collection_already_exists():
    mock_client = MagicMock()
    mock_client.get_collection.return_value = MagicMock()  # Does not raise

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client = mock_client
    store._known_collections = set()

    result = store.create_collection("existing_col")
    assert result is False
    mock_client.create_collection.assert_not_called()


def test_vector_store_add_and_search():
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.id = 123
    mock_result.score = 0.95
    mock_result.payload = {"title": "Test Paper"}
    mock_client.search.return_value = [mock_result]

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client = mock_client
    store._known_collections = {"papers"}

    vectors = [[0.1] * 32]
    metadata = [{"title": "Test Paper"}]
    ids = store.add_vectors("papers", vectors, metadata)
    assert len(ids) == 1

    results = store.search("papers", [0.1] * 32, limit=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.95


# ---------------------------------------------------------------------------
# IndexingAgent
# ---------------------------------------------------------------------------

def test_indexing_agent_novelty_novel():
    mock_vs = MagicMock(spec=QdrantVectorStore)
    mock_vs.search.return_value = [{"id": 1, "score": 0.40, "metadata": {"title": "Old Paper"}}]

    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=384))
    agent = IndexingAgent.__new__(IndexingAgent)
    agent.vector_store = mock_vs
    agent.embedder = embedder
    agent.novelty_threshold = 0.70

    result = agent.check_novelty("Novel abstract about AI founders", "Novel Title")
    assert result.is_novel is True
    assert result.score == 0.40


def test_indexing_agent_novelty_not_novel():
    mock_vs = MagicMock(spec=QdrantVectorStore)
    mock_vs.search.return_value = [{"id": 1, "score": 0.85, "metadata": {"title": "Very Similar Paper"}}]

    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=384))
    agent = IndexingAgent.__new__(IndexingAgent)
    agent.vector_store = mock_vs
    agent.embedder = embedder
    agent.novelty_threshold = 0.70

    result = agent.check_novelty("Abstract similar to existing work")
    assert result.is_novel is False
    assert "Very Similar Paper" in result.feedback


def test_indexing_agent_indexes_papers():
    mock_vs = MagicMock(spec=QdrantVectorStore)
    mock_vs.add_vectors.return_value = [1, 2]
    mock_vs.create_collection.return_value = False

    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=384))
    agent = IndexingAgent.__new__(IndexingAgent)
    agent.vector_store = mock_vs
    agent.embedder = embedder
    agent.novelty_threshold = 0.70

    papers = [
        PaperSummary(title="P1", abstract="A1", authors=["A"], year=2023,
                     citation_count=10, source="arxiv", url="u1", relevance_summary="r1"),
        PaperSummary(title="P2", abstract="A2", authors=["B"], year=2022,
                     citation_count=5, source="semantic_scholar", url="u2", relevance_summary="r2"),
    ]
    count = agent.index_papers(papers)
    assert count == 2
    mock_vs.add_vectors.assert_called_once()


# ---------------------------------------------------------------------------
# HypothesisAgent (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hypothesis_agent_generates_output():
    fake_output = make_fake_hypothesis()

    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(fake_output.model_dump())
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = HypothesisAgent.__new__(HypothesisAgent)
    agent.llm = mock_llm

    result = await agent.generate(domain="startup", keywords=["AI", "founder"])
    assert isinstance(result, HypothesisOutput)
    assert result.domain == "startup"
    assert len(result.search_queries) >= 2


@pytest.mark.asyncio
async def test_hypothesis_agent_invalid_domain():
    agent = HypothesisAgent.__new__(HypothesisAgent)
    agent.llm = AsyncMock()
    with pytest.raises(ValueError):
        await agent.generate(domain="unknown")


# ---------------------------------------------------------------------------
# LiteratureAgent (mocked search + LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_literature_agent_deduplicates():
    """Same paper from SS and arXiv should be counted once."""
    paper = make_fake_paper(title="AI in Startups: A Study")
    arxiv_paper = make_fake_paper(title="AI in Startups: A Study", _source="arxiv")

    mock_ss = MagicMock()
    mock_ss.search_papers = AsyncMock(return_value=[paper])
    mock_arxiv = MagicMock()
    mock_arxiv.search_papers = AsyncMock(return_value=[arxiv_paper])

    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '["Relevant to AI startup research."]'
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent.ss = mock_ss
    agent.arxiv = mock_arxiv
    agent.llm = mock_llm
    agent.papers_per_query = 20

    output = await agent.search(queries=["AI startup"], domain="startup")
    assert output.total_found == 2   # Both fetched
    assert len(output.papers) == 1   # Deduplicated to 1


@pytest.mark.asyncio
async def test_literature_agent_tolerates_source_failure():
    """If arXiv fails, SS results are still returned."""
    paper = make_fake_paper()

    mock_ss = MagicMock()
    mock_ss.search_papers = AsyncMock(return_value=[paper])
    mock_arxiv = MagicMock()
    mock_arxiv.search_papers = AsyncMock(side_effect=Exception("arXiv down"))

    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '["Relevant."]'
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent.ss = mock_ss
    agent.arxiv = mock_arxiv
    agent.llm = mock_llm
    agent.papers_per_query = 20

    output = await agent.search(queries=["AI startup"], domain="startup")
    assert len(output.papers) == 1   # Only SS paper survived
