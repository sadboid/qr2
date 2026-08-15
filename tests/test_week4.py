"""Week 4 tests — End-to-end testing, citation verification, and multi-format output.

These tests use mocks for external services so they run without credentials.
Integration tests marked with @pytest.mark.slow run against real APIs.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from research_machine.agents.writing_agent import DraftPaper
from research_machine.db.models import Citation
from research_machine.quality.citation_verifier import CitationVerifier, VerificationReport
from research_machine.output.formatter import PaperFormatter
from research_machine.metrics import PaperMetrics, MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_draft_paper(**kwargs) -> DraftPaper:
    defaults = dict(
        title="AI Adoption and Founder Productivity",
        abstract="This study examines how AI tools affect founder decision-making speed in early-stage startups.",
        introduction_md="## Introduction\n\nAI adoption is rapidly increasing in startups...",
        methods_md="## Methods\n\nWe conducted a survey of 200 founders...",
        results_md="## Results\n\nOur findings show a 30% improvement in decision speed...",
        discussion_md="## Discussion\n\nThese results suggest that AI tools are transformative...",
        citations=[],
        citation_count=20,
        content_markdown="# Full Paper\n\n...",
        content_latex="\\documentclass{article}...",
    )
    defaults.update(kwargs)
    return DraftPaper(**defaults)


def make_fake_citation(**kwargs) -> Citation:
    defaults = dict(
        id="cit_001",
        title="Machine Learning in Business",
        authors=[{"name": "Smith, J.", "authorId": "auth_001"}],
        year=2023,
        venue="Journal of Business Computing",
        h_index=None,
        external_id=None,  # Add this
        url="https://example.com/paper1",
        abstract="A study on ML applications in business...",
        paper_id=None,  # Add this
    )
    defaults.update(kwargs)

    # Create a mock Citation object with mutable attributes
    class MockCitation:
        def __init__(self, **attrs):
            for k, v in attrs.items():
                setattr(self, k, v)

    return MockCitation(**defaults)


# ---------------------------------------------------------------------------
# Citation Verifier Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_citation_verifier_finds_paper():
    """Test CitationVerifier against mock Semantic Scholar API."""
    verifier = CitationVerifier()

    # Mock the SemanticScholarClient
    mock_client = MagicMock()
    verifier.client = mock_client

    # Mock search results
    search_result = {
        "paperId": "ss_12345",
        "title": "Machine Learning in Business",
        "year": 2023,
        "authors": [{"name": "Smith, J.", "authorId": "auth_001"}],
        "citationCount": 15,
    }
    mock_client.search_papers = AsyncMock(return_value=[search_result])
    mock_client.get_author_papers = AsyncMock(return_value=[
        {"citationCount": 20},
        {"citationCount": 18},
        {"citationCount": 15},
    ])
    mock_client.calculate_h_index = MagicMock(return_value=3)

    citation = make_fake_citation()
    result = await verifier._verify_single_citation(citation)

    assert result["verified"] is True
    assert result["external_id"] == "ss_12345"
    assert result["h_index"] == 3


@pytest.mark.asyncio
async def test_citation_verifier_handles_missing_paper():
    """Test CitationVerifier when paper is not found."""
    verifier = CitationVerifier()

    mock_client = MagicMock()
    verifier.client = mock_client
    mock_client.search_papers = AsyncMock(return_value=[])

    citation = make_fake_citation()
    result = await verifier._verify_single_citation(citation)

    assert result["verified"] is False
    assert "found" in result["reason"].lower() or "no papers" in result["reason"].lower()


@pytest.mark.asyncio
async def test_citation_verifier_uses_cache():
    """Test CitationVerifier caches results."""
    verifier = CitationVerifier()

    citation1 = make_fake_citation(title="Machine Learning in Business", year=2023)
    citation2 = make_fake_citation(title="Machine Learning in Business", year=2023)

    mock_client = MagicMock()
    verifier.client = mock_client

    search_result = {
        "paperId": "ss_12345",
        "title": "Machine Learning in Business",
        "year": 2023,
        "authors": [{"name": "Smith, J.", "authorId": "auth_001"}],
    }
    mock_client.search_papers = AsyncMock(return_value=[search_result])
    mock_client.get_author_papers = AsyncMock(return_value=[{"citationCount": 20}])
    mock_client.calculate_h_index = MagicMock(return_value=3)

    # First call
    await verifier._verify_single_citation(citation1)
    call_count_1 = mock_client.search_papers.call_count

    # Second call with same citation (should use cache)
    await verifier._verify_single_citation(citation2)
    call_count_2 = mock_client.search_papers.call_count

    # Should not have called API again (cache hit)
    assert call_count_2 == call_count_1


@pytest.mark.asyncio
async def test_citation_verifier_batch_verification():
    """Test batch verification of multiple citations."""
    verifier = CitationVerifier()

    mock_client = MagicMock()
    verifier.client = mock_client

    # Create a side effect function that returns matching results
    async def mock_search(query, limit=3):
        # Return different results based on the query to match the title
        return [{
            "paperId": f"ss_{len(query)}",
            "title": query,  # Match the query title
            "year": 2023 if "Paper 1" in query else 2022,
            "authors": [{"name": "Smith, J.", "authorId": "auth_001"}],
        }]

    mock_client.search_papers = AsyncMock(side_effect=mock_search)
    mock_client.get_author_papers = AsyncMock(return_value=[{"citationCount": 20}])
    mock_client.calculate_h_index = MagicMock(return_value=3)

    citations = [
        make_fake_citation(title="Paper 1", year=2023),
        make_fake_citation(title="Paper 2", year=2022),
    ]

    verifier.request_delay = 0.01  # Speed up test

    report = await verifier.verify_citations(citations)

    assert isinstance(report, VerificationReport)
    assert report.total_citations == 2
    assert report.verified_count > 0
    assert 0 <= report.confidence <= 1


# ---------------------------------------------------------------------------
# Paper Formatter Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paper_formatter_to_latex():
    """Test conversion of paper to LaTeX."""
    formatter = PaperFormatter()
    draft = make_fake_draft_paper()

    latex_content = await formatter.to_latex(draft)

    assert isinstance(latex_content, str)
    assert "\\documentclass" in latex_content
    assert "Introduction" in latex_content


@pytest.mark.asyncio
async def test_paper_formatter_to_docx():
    """Test conversion of paper to DOCX."""
    formatter = PaperFormatter()
    draft = make_fake_draft_paper()

    docx_bytes = await formatter.to_docx(draft)

    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 0
    # DOCX files start with PK signature (ZIP format)
    assert docx_bytes[:2] == b'PK'


@pytest.mark.asyncio
async def test_paper_formatter_to_json():
    """Test conversion of paper to JSON metadata."""
    formatter = PaperFormatter()

    # Create citations as dicts for JSON serialization
    citations = [
        {
            "title": "Paper 1",
            "year": 2023,
            "authors": [{"name": "Author A"}],
        },
        {
            "title": "Paper 2",
            "year": 2022,
            "authors": [{"name": "Author B"}],
        },
    ]

    draft = make_fake_draft_paper(citations=citations)

    metadata = {
        "novelty_score": 0.65,
        "citation_count": 20,
        "status": "accepted",
    }

    json_data = await formatter.to_json(draft, metadata)

    assert isinstance(json_data, dict)
    assert json_data["metadata"]["title"] == draft.title
    assert json_data["metadata"]["novelty_score"] == 0.65
    assert json_data["sections"]["introduction"] == draft.introduction_md
    assert json_data["citations"]["count"] == draft.citation_count


@pytest.mark.asyncio
async def test_paper_formatter_save_all_formats(tmp_path):
    """Test saving paper in all formats."""
    formatter = PaperFormatter()
    draft = make_fake_draft_paper()

    output_dir = tmp_path / "papers" / "test_paper"
    metadata = {"status": "accepted", "novelty_score": 0.65}

    outputs = await formatter.save_all_formats(draft, output_dir, metadata)

    assert "markdown" in outputs
    assert "latex" in outputs
    assert "docx" in outputs

    # Check files exist
    assert outputs["markdown"].exists()
    assert outputs["latex"].exists()
    assert outputs["docx"].exists()

    # Verify markdown content
    with open(outputs["markdown"], "r") as f:
        md_content = f.read()
        assert len(md_content) > 0

    # Verify LaTeX content
    with open(outputs["latex"], "r") as f:
        latex_content = f.read()
        assert "\\documentclass" in latex_content


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

def test_metrics_collector_add_paper():
    """Test MetricsCollector adds papers correctly."""
    collector = MetricsCollector()

    metrics = PaperMetrics(
        paper_id="p_001",
        title="Test Paper",
        novelty_score=0.65,
        citation_count=20,
        avg_h_index=6.5,
        recency_ratio=0.35,
        peer_review_score=7.5,
        generation_cost_usd=5.50,
        generation_time_seconds=120,
        status="accepted",
    )

    collector.add_paper(metrics)

    assert len(collector.papers) == 1
    assert collector.papers[0].title == "Test Paper"


def test_metrics_collector_summary():
    """Test MetricsCollector generates summary."""
    collector = MetricsCollector()

    for i in range(3):
        metrics = PaperMetrics(
            paper_id=f"p_{i:03d}",
            title=f"Paper {i}",
            novelty_score=0.60 + (i * 0.05),
            citation_count=15 + (i * 2),
            avg_h_index=5.0 + (i * 0.5),
            recency_ratio=0.30 + (i * 0.05),
            peer_review_score=7.0 + (i * 0.5),
            generation_cost_usd=5.0 + (i * 0.5),
            generation_time_seconds=100 + (i * 10),
            status="accepted" if i < 2 else "revision_requested",
        )
        collector.add_paper(metrics)

    report = collector.summary_report()

    assert report["total_papers"] == 3
    assert "accepted" in report["status_breakdown"]
    assert "revision_requested" in report["status_breakdown"]
    assert report["status_breakdown"]["accepted"] == 2
    assert report["average_metrics"]["citation_count"] > 0
    assert report["total_cost_usd"] > 0
    assert report["cost_per_paper"] > 0


def test_metrics_paper_gates():
    """Test PaperMetrics gate checking."""
    metrics = PaperMetrics(
        paper_id="p_001",
        title="Good Paper",
        novelty_score=0.65,  # Passes (< 0.70)
        citation_count=20,  # Passes (>= 15)
        avg_h_index=6.0,  # Passes (>= 5)
        recency_ratio=0.35,  # Passes (>= 0.30)
        peer_review_score=7.5,  # Passes (>= 7.0)
        generation_cost_usd=5.50,
        generation_time_seconds=120,
        status="accepted",
    )

    assert metrics.passes_novelty_gate()
    assert metrics.passes_citation_gate()
    assert metrics.passes_peer_review_gate()
    assert metrics.all_gates_pass()


def test_metrics_paper_gates_failure():
    """Test PaperMetrics gate failures."""
    metrics = PaperMetrics(
        paper_id="p_002",
        title="Poor Paper",
        novelty_score=0.75,  # Fails (>= 0.70)
        citation_count=10,  # Fails (< 15)
        avg_h_index=3.0,  # Fails (< 5)
        recency_ratio=0.20,  # Fails (< 0.30)
        peer_review_score=6.0,  # Fails (< 7.0)
        generation_cost_usd=5.50,
        generation_time_seconds=120,
        status="rejected",
    )

    assert not metrics.passes_novelty_gate()
    assert not metrics.passes_citation_gate()
    assert not metrics.passes_peer_review_gate()
    assert not metrics.all_gates_pass()


def test_metrics_get_high_quality_papers():
    """Test MetricsCollector filtering for high-quality papers."""
    collector = MetricsCollector()

    # Good paper
    collector.add_paper(PaperMetrics(
        paper_id="p_001",
        title="Good Paper",
        novelty_score=0.65,
        citation_count=20,
        avg_h_index=6.0,
        recency_ratio=0.35,
        peer_review_score=7.5,
        generation_cost_usd=5.50,
        generation_time_seconds=120,
        status="accepted",
    ))

    # Poor paper
    collector.add_paper(PaperMetrics(
        paper_id="p_002",
        title="Poor Paper",
        novelty_score=0.75,
        citation_count=10,
        avg_h_index=3.0,
        recency_ratio=0.20,
        peer_review_score=6.0,
        generation_cost_usd=5.50,
        generation_time_seconds=120,
        status="rejected",
    ))

    high_quality = collector.get_high_quality_papers()

    assert len(high_quality) == 1
    assert high_quality[0].title == "Good Paper"


# ---------------------------------------------------------------------------
# Integration Tests (slow, may require real APIs)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_pipeline_integration():
    """
    End-to-end integration test.
    Requires: working FastAPI server, mocked LLMs
    """
    # This would integrate with the actual pipeline
    # For now, we test the components separately
    pass
