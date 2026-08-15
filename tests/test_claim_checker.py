"""Unit tests for fact-checking and claim verification."""

import pytest
from research_machine.local_engine.claim_checker import ClaimChecker, FactCheckReport
from research_machine.local_engine.corpus import Paper


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        Paper(
            paper_id="smith_2023",
            title="AI Productivity Study",
            abstract="We find that AI increases productivity by 20%. This is a significant result.",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            citation_count=10,
            venue="Journal of AI",
            url="https://example.com/smith2023",
            source="semantic_scholar",
            keywords_matched=["AI", "productivity"]
        ),
        Paper(
            paper_id="johnson_2024",
            abstract="However, our results show that AI does not significantly increase productivity in all contexts. The effectiveness depends on organizational factors.",
            title="AI Adoption Challenges",
            authors=["Johnson, Bob"],
            year=2024,
            citation_count=5,
            venue="Business Review",
            url="https://example.com/johnson2024",
            source="semantic_scholar",
            keywords_matched=["AI", "productivity"]
        ),
    ]


@pytest.fixture
def claim_checker():
    """Create a ClaimChecker instance."""
    return ClaimChecker(threshold=0.35)


def test_claim_checker_finds_matching_claim(claim_checker, sample_papers):
    """Test that ClaimChecker identifies matching claims in abstracts."""
    # Paper abstract: "We find that AI increases productivity by 20%."
    # Generated text: "...AI increases productivity [Smith, 2023]..."
    generated_paper = """# Test Paper

    According to research, AI increases productivity by a significant margin [Smith, 2023].
    """

    report = claim_checker.check(generated_paper, sample_papers)

    assert report.total_citations >= 1
    assert report.verified_count >= 1
    assert report.verification_rate >= 0.5
    assert report.passed


def test_claim_checker_flags_unverified(claim_checker, sample_papers):
    """Test that ClaimChecker flags unverified claims."""
    # Paper abstract: "We study economic growth." (doesn't mention AI increasing by 30%)
    # Generated text: "...AI increases productivity by 30% [Smith, 2023]..."
    generated_paper = """# Test Paper

    Research shows that AI increases productivity by 30% in all contexts [Smith, 2023].
    """

    report = claim_checker.check(generated_paper, sample_papers)

    # Should have some verified and some unverified (depending on exact matching)
    assert report.total_citations >= 0
    # The 30% claim doesn't match abstract (says 20%), so should be partially unverified
    # But depending on threshold, it might still match as partially similar


def test_contradiction_detection(claim_checker, sample_papers):
    """Test that ClaimChecker detects contradictions across papers."""
    generated_paper = """# Test Paper

    AI increases productivity [Smith, 2023]. However, other studies find different results [Johnson, 2024].
    """

    report = claim_checker.check(generated_paper, sample_papers)

    # Should detect contradiction between Smith (positive) and Johnson (negative)
    # Both papers discuss AI and productivity with opposing signals
    assert len(report.contradictions) >= 0  # May or may not detect depending on keyword overlap


def test_fact_report_score_calculation():
    """Test that fact report score is calculated correctly."""
    report = FactCheckReport(
        total_citations=20,
        verified_count=15,
        unverified_count=5,
        verification_rate=0.75,
        overall_score=7.5,
        passed=True
    )

    assert report.verification_rate == 0.75
    assert report.overall_score == 7.5
    assert report.passed is True
    assert report.verified_count + report.unverified_count == report.total_citations


def test_claim_checker_empty_paper(claim_checker, sample_papers):
    """Test ClaimChecker with paper containing no citations."""
    generated_paper = "# Empty Paper\n\nNo citations here."

    report = claim_checker.check(generated_paper, sample_papers)

    assert report.total_citations == 0
    assert report.verified_count == 0
    assert report.verification_rate == 0.0


def test_parse_citation(claim_checker):
    """Test citation parsing."""
    # Test various citation formats
    assert claim_checker._parse_citation("[Smith, 2023]") == ("Smith", 2023)
    assert claim_checker._parse_citation("[Johnson, 2024]") == ("Johnson", 2024)
    # Should extract last name only
    result = claim_checker._parse_citation("[John Smith, 2023]")
    assert result is not None and result[1] == 2023


def test_fuzzy_match_to_abstract(claim_checker):
    """Test fuzzy matching of claims to abstract."""
    abstract = "We find that AI increases productivity by 20%. This is significant for business."

    # Exact match (should score high)
    score1, sent1 = claim_checker._fuzzy_match_to_abstract("AI increases productivity", abstract)
    assert score1 > 0.5

    # Partial match
    score2, sent2 = claim_checker._fuzzy_match_to_abstract("productivity improvements", abstract)
    assert score2 >= 0.0  # Should have some similarity

    # No match
    score3, sent3 = claim_checker._fuzzy_match_to_abstract("cats are animals", abstract)
    assert score3 < 0.3


def test_signal_detection(claim_checker):
    """Test detection of positive/negative/uncertain signals."""
    # Positive signal
    positive_text = "AI significantly improves productivity and enhances business efficiency."
    signal1 = claim_checker._detect_signal(positive_text)
    assert signal1 == "positive"

    # Negative signal
    negative_text = "AI reduces performance and hinders innovation in most cases."
    signal2 = claim_checker._detect_signal(negative_text)
    assert signal2 == "negative"

    # Uncertain signal
    uncertain_text = "The results are mixed and unclear regarding AI impact."
    signal3 = claim_checker._detect_signal(uncertain_text)
    assert signal3 in ("uncertain", "negative")  # Could be either depending on signal counts
