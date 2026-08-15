"""Week 3 tests — Analysis, Writing, and Quality Gates.

These tests use mocks for external services (Claude API, Qdrant) so they run without credentials.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from research_machine.agents.hypothesis_agent import HypothesisOutput
from research_machine.agents.literature_agent import PaperSummary, LiteratureOutput
from research_machine.agents.analysis_agent import AnalysisAgent, AnalysisOutput
from research_machine.agents.writing_agent import WritingAgent, DraftPaper
from research_machine.quality.novelty_check import NoveltyGate, QualityGateResult
from research_machine.quality.citation_validator import CitationValidator
from research_machine.quality.peer_review import PeerReviewGate
from research_machine.rag.embeddings import MockEmbeddingsProvider, PaperSummaryEmbedder
from research_machine.rag.vector_store import QdrantVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_hypothesis(**kwargs) -> HypothesisOutput:
    defaults = dict(
        primary_question="How does AI tool adoption affect founder productivity?",
        angles=["Cognitive load", "Decision quality"],
        search_queries=["AI adoption founders"],
        rationale="Gap in longitudinal studies",
        suggested_methodology="6-month survey of 200 founders",
        target_venues=["Journal of Business Venturing"],
        domain="startup",
    )
    defaults.update(kwargs)
    return HypothesisOutput(**defaults)


def make_fake_papers(count: int = 5) -> list[PaperSummary]:
    papers = []
    for i in range(count):
        papers.append(
            PaperSummary(
                title=f"Paper {i}: AI and Productivity",
                abstract=f"Study on AI impact on productivity metric {i}.",
                authors=[f"Author{i}"],
                year=2023,
                citation_count=10 + i,
                source="semantic_scholar",
                url=f"https://example.com/{i}",
                relevance_summary=f"Relevant to founder productivity study.",
            )
        )
    return papers


def make_fake_literature(**kwargs) -> LiteratureOutput:
    defaults = dict(
        papers=make_fake_papers(10),
        total_found=50,
        search_queries_used=["AI adoption founders"],
    )
    defaults.update(kwargs)
    return LiteratureOutput(**defaults)


# ---------------------------------------------------------------------------
# Analysis Agent Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analysis_agent_identifies_gaps():
    """Test that AnalysisAgent finds gaps and trends in papers."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    analysis_data = {
        "key_findings": ["Finding 1", "Finding 2"],
        "research_gaps": ["Gap 1"],
        "methodologies_used": ["Survey"],
        "contradictions": [],
        "trend_analysis": "Trends summary",
        "contribution_angle": "Novel angle",
        "recommended_methodology": "Empirical survey",
        "key_citations": ["Paper 1"],
    }
    mock_response.content = json.dumps(analysis_data)
    mock_llm.invoke = MagicMock(return_value=mock_response)

    agent = AnalysisAgent.__new__(AnalysisAgent)
    agent.llm = mock_llm

    hypothesis = make_fake_hypothesis()
    papers = make_fake_papers()

    result = await agent.analyze(hypothesis, papers, "startup")
    assert isinstance(result, AnalysisOutput)
    assert len(result.key_findings) == 2
    assert len(result.research_gaps) >= 1


# ---------------------------------------------------------------------------
# Writing Agent Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writing_agent_produces_imrad():
    """Test that WritingAgent produces IMRAD-structured paper."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    paper_data = {
        "title": "AI Adoption and Founder Productivity",
        "abstract": "This study examines...",
        "introduction_md": "## Background\n\nThe study of AI impact...",
        "methods_md": "## Methods\n\nWe conducted a survey...",
        "results_md": "## Results\n\nOur findings show...",
        "discussion_md": "## Discussion\n\nThese results suggest...",
        "citation_count": 20,
        "content_markdown": "# Full paper",
    }
    mock_response.content = json.dumps(paper_data)
    mock_llm.invoke = MagicMock(return_value=mock_response)

    agent = WritingAgent.__new__(WritingAgent)
    agent.llm = mock_llm

    hypothesis = make_fake_hypothesis()
    analysis = AnalysisOutput(
        key_findings=["Finding 1"],
        research_gaps=["Gap 1"],
        methodologies_used=["Survey"],
        contradictions=[],
        trend_analysis="Trend summary",
        contribution_angle="Novel angle",
        recommended_methodology="Survey",
        key_citations=["Paper 1"],
    )
    papers = make_fake_papers()

    result = await agent.write(hypothesis, analysis, papers, "startup")
    assert isinstance(result, DraftPaper)
    assert "AI Adoption" in result.title
    assert len(result.introduction_md) > 0
    assert len(result.discussion_md) > 0


# ---------------------------------------------------------------------------
# Novelty Gate Tests
# ---------------------------------------------------------------------------

def test_novelty_gate_accepts_novel_paper():
    """Test that NoveltyGate accepts papers below similarity threshold."""
    mock_vs = MagicMock(spec=QdrantVectorStore)
    mock_vs.search.return_value = [{"id": 1, "score": 0.5, "metadata": {"title": "Some Paper"}}]

    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=384))
    gate = NoveltyGate(mock_vs, embedder)

    result = gate.evaluate("Novel Paper", "This is a novel abstract")
    assert result.passed is True
    assert result.score == 0.5


def test_novelty_gate_rejects_duplicate():
    """Test that NoveltyGate rejects papers above similarity threshold."""
    mock_vs = MagicMock(spec=QdrantVectorStore)
    mock_vs.search.return_value = [{"id": 1, "score": 0.85, "metadata": {"title": "Very Similar Paper"}}]

    embedder = PaperSummaryEmbedder(MockEmbeddingsProvider(dimension=384))
    gate = NoveltyGate(mock_vs, embedder)

    result = gate.evaluate("Duplicate Paper", "Very similar abstract")
    assert result.passed is False
    assert result.score == 0.85


# ---------------------------------------------------------------------------
# Citation Validator Tests
# ---------------------------------------------------------------------------

def test_citation_validator_accepts_good_citations():
    """Test that CitationValidator passes with sufficient citations."""
    validator = CitationValidator(None)
    citations = [
        {"year": 2023, "h_index": 10},
        {"year": 2023, "h_index": 8},
        {"year": 2022, "h_index": 6},
        {"year": 2021, "h_index": 5},
        {"year": 2020, "h_index": 12},
        {"year": 2023, "h_index": 7},
        {"year": 2023, "h_index": 9},
        {"year": 2022, "h_index": 15},
        {"year": 2024, "h_index": 4},
        {"year": 2023, "h_index": 11},
        {"year": 2023, "h_index": 6},
        {"year": 2022, "h_index": 8},
        {"year": 2021, "h_index": 10},
        {"year": 2020, "h_index": 7},
        {"year": 2023, "h_index": 9},
    ]
    result = validator.evaluate(citations)
    assert result.passed is True
    assert result.score > 0.5


def test_citation_validator_rejects_insufficient_citations():
    """Test that CitationValidator rejects with too few citations."""
    validator = CitationValidator(None)
    citations = [
        {"year": 2023, "h_index": 10},
        {"year": 2023, "h_index": 8},
    ]
    result = validator.evaluate(citations)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Peer Review Gate Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_peer_review_quick_review_passes():
    """Test Haiku quick review passes with no critical issues."""
    gate = PeerReviewGate.__new__(PeerReviewGate)
    gate.haiku = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"issues": []}'
    gate.haiku.invoke = MagicMock(return_value=mock_response)

    result = await gate.quick_review("This is a well-structured paper with clear methodology...")
    assert result.passed is True
    assert len(result.issues) == 0


@pytest.mark.asyncio
async def test_peer_review_detailed_accepts():
    """Test Sonnet detailed review accepts with high score."""
    gate = PeerReviewGate.__new__(PeerReviewGate)
    gate.sonnet = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": 8.0, "feedback": "Excellent paper", "recommendation": "accept"}'
    gate.sonnet.invoke = MagicMock(return_value=mock_response)

    result = await gate.detailed_review("Paper content", "startup")
    assert result.score == 8.0
    assert result.recommendation == "accept"


@pytest.mark.asyncio
async def test_peer_review_full_flow():
    """Test full two-stage peer review."""
    gate = PeerReviewGate.__new__(PeerReviewGate)

    # Mock Haiku
    haiku_response = MagicMock()
    haiku_response.content = '{"issues": []}'
    gate.haiku = MagicMock()
    gate.haiku.invoke = MagicMock(return_value=haiku_response)

    # Mock Sonnet
    sonnet_response = MagicMock()
    sonnet_response.content = '{"score": 7.5, "feedback": "Good work", "recommendation": "accept"}'
    gate.sonnet = MagicMock()
    gate.sonnet.invoke = MagicMock(return_value=sonnet_response)

    result = await gate.evaluate("Full paper content", "startup")
    assert result.passed is True
    assert result.score > 0.7
