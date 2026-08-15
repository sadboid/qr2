#!/usr/bin/env python3
"""Script to generate test papers end-to-end.

Usage:
    # Real mode (needs ANTHROPIC_API_KEY + running Qdrant/PostgreSQL):
    python scripts/generate_test_papers.py --questions 3 --output-dir ./papers

    # Mock mode (no API keys needed — generates real output files with fake content):
    python scripts/generate_test_papers.py --questions 3 --mock --output-dir ./papers_demo
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

from research_machine.agents.writing_agent import DraftPaper
from research_machine.agents.analysis_agent import AnalysisOutput
from research_machine.agents.hypothesis_agent import HypothesisOutput
from research_machine.agents.literature_agent import PaperSummary, LiteratureOutput
from research_machine.output.formatter import PaperFormatter
from research_machine.metrics import PaperMetrics, MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test research questions for seeding
TEST_QUESTIONS = [
    {
        "domain": "startup",
        "keywords": ["founder productivity", "AI tools", "decision-making"],
        "description": "How do AI tools affect founder decision-making speed in early-stage startups?",
    },
    {
        "domain": "enterprise",
        "keywords": ["LLM deployment", "governance", "risk management"],
        "description": "What organizational structures best support LLM governance and responsible AI adoption?",
    },
    {
        "domain": "startup",
        "keywords": ["startup failure prediction", "AI", "early warning"],
        "description": "Can machine learning models predict startup failure with early warning signals?",
    },
    {
        "domain": "enterprise",
        "keywords": ["MLOps", "data quality", "operational excellence"],
        "description": "How do leading enterprises balance MLOps efficiency with data quality requirements?",
    },
    {
        "domain": "startup",
        "keywords": ["business model innovation", "AI", "market disruption"],
        "description": "How can AI-enabled business model innovation create sustainable competitive advantages for startups?",
    },
]

# ---------------------------------------------------------------------------
# Mock content templates — realistic enough to test the formatter properly
# ---------------------------------------------------------------------------

_MOCK_PAPERS = [
    {
        "title": "AI Tools and Productivity in Early-Stage Ventures",
        "authors": ["Chen, L.", "Patel, R."],
        "year": 2023,
        "citations": 42,
    },
    {
        "title": "Decision Velocity and Cognitive Load in Startup Founders",
        "authors": ["Kim, J.", "Schmidt, M."],
        "year": 2022,
        "citations": 38,
    },
    {
        "title": "Machine Learning Applications in Business Operations",
        "authors": ["Liu, X.", "Hoffman, D."],
        "year": 2023,
        "citations": 55,
    },
    {
        "title": "Organizational Readiness for AI Adoption: A Framework",
        "authors": ["Wang, Y.", "Thompson, E."],
        "year": 2022,
        "citations": 67,
    },
    {
        "title": "Startup Failure Rates and Predictive Analytics",
        "authors": ["Martinez, A.", "Lee, S."],
        "year": 2021,
        "citations": 29,
    },
    {
        "title": "LLM Governance in Fortune 500 Companies",
        "authors": ["Sharma, P.", "Brown, K."],
        "year": 2024,
        "citations": 18,
    },
    {
        "title": "AI-Enabled Business Model Innovation: Case Studies",
        "authors": ["Johnson, T.", "Garcia, C."],
        "year": 2023,
        "citations": 34,
    },
    {
        "title": "Human-AI Collaboration in Strategic Decision Making",
        "authors": ["Anderson, R.", "Nakamura, H."],
        "year": 2023,
        "citations": 51,
    },
]

_MOCK_INTRO = """\
The rapid proliferation of artificial intelligence (AI) tools in the business landscape has fundamentally
altered how organizations process information, allocate resources, and execute decisions. For early-stage
startups operating under resource constraints and uncertainty, the adoption of AI-powered tools presents
both opportunities and risks that remain understudied in the entrepreneurship literature.

Prior research has demonstrated that cognitive load significantly affects founder decision quality
[Decision Velocity and Cognitive Load in Startup Founders, 2022]. However, the specific mechanisms
through which AI tools modulate this relationship — particularly in time-pressured, high-stakes startup
environments — have not been systematically examined. This gap is especially pronounced in the context
of Series A fundraising cycles, where decision speed and accuracy have documented implications for
survival outcomes [Startup Failure Rates and Predictive Analytics, 2021].

This paper addresses the research question: **How does AI tool adoption affect founder decision-making
speed and quality in early-stage startups?** Drawing on a longitudinal survey of 200 founders across
six industry verticals, we propose that AI tools mediate the relationship between information volume
and decision velocity through three mechanisms: (1) automated synthesis, (2) scenario simulation, and
(3) real-time benchmarking against market data [AI Tools and Productivity in Early-Stage Ventures, 2023].

Our contribution is threefold. First, we provide the first longitudinal evidence of AI tool effects on
founder cognitive load in naturalistic settings. Second, we develop a typology of AI adoption patterns
that distinguishes between augmentative and substitutive tool usage. Third, we offer actionable
guidelines for founders seeking to optimize AI tool portfolios for their specific decision context.
"""

_MOCK_METHODS = """\
This study employs a mixed-methods longitudinal design spanning six months, combining quantitative
survey data with qualitative interview insights. We recruited 200 founders of early-stage startups
(Seed to Series A) from three major startup ecosystems: San Francisco, New York, and London.

Participants completed bi-weekly surveys measuring: (a) AI tool usage frequency and type using the
validated AI Adoption Scale [Machine Learning Applications in Business Operations, 2023]; (b) decision
velocity via a novel Decision Timing Log; and (c) decision outcome quality rated retrospectively at
30-day intervals. Cognitive load was assessed using the NASA Task Load Index (NASA-TLX), adapted for
entrepreneurial contexts.

Semi-structured interviews with a stratified subsample of 40 founders were conducted at months one,
three, and six. Interviews explored subjective experiences of AI tool integration and perceived impacts
on decision processes. All interviews were transcribed and analyzed using thematic analysis following
Braun and Clarke's (2019) framework [Human-AI Collaboration in Strategic Decision Making, 2023].

Statistical analysis employed multilevel mixed-effects models to account for within-person variation
over time, with firm age, industry, and funding stage as covariates. Propensity score matching was used
to address selection bias in AI tool adoption. All analyses were conducted in R 4.3.0 using the lme4
and MatchIt packages.
"""

_MOCK_RESULTS = """\
Our analysis of 200 founders over six months reveals three primary findings regarding AI tool adoption
and decision-making efficiency.

**Finding 1: AI tool adoption is associated with 31% faster decision cycles.** Founders using AI tools
for information synthesis completed the core decision cycle (information gathering to commitment) in
an average of 4.2 days, compared to 6.1 days for non-AI-adopting founders (p < 0.001, Cohen's d = 0.74).
This effect was strongest in market analysis decisions (42% faster) and weakest in personnel decisions
(18% faster), suggesting domain-specific mediating factors [AI Tools and Productivity in Early-Stage
Ventures, 2023].

**Finding 2: Augmentative AI use outperforms substitutive use in decision quality.** Founders who used
AI tools to augment their analysis (verify hypotheses, generate alternatives) achieved higher ex-post
decision quality ratings (M = 7.8/10) than those who delegated decisions to AI recommendations
(M = 6.2/10). This 26% difference persisted after controlling for decision complexity and domain expertise.

**Finding 3: Cognitive load mediation is confirmed.** Path analysis confirmed that AI tools reduced
founder cognitive load (β = -0.43, SE = 0.06, p < 0.001), which in turn increased decision velocity
(β = -0.38, SE = 0.07, p < 0.001) and quality (β = 0.29, SE = 0.05, p < 0.001). The indirect effect
(cognitive load as mediator) accounted for 61% of the total effect of AI adoption on decision velocity
[Decision Velocity and Cognitive Load in Startup Founders, 2022].
"""

_MOCK_DISCUSSION = """\
Our findings make several contributions to the entrepreneurship and AI adoption literatures. The
31% improvement in decision velocity confirms and extends prior work on AI productivity effects in
corporate settings [Organizational Readiness for AI Adoption: A Framework, 2022] to the entrepreneurial
context, where resource constraints and uncertainty create distinct boundary conditions.

The distinction between augmentative and substitutive AI use has important theoretical implications.
Our results suggest that AI tools function most effectively as cognitive prosthetics — amplifying founder
judgment rather than replacing it. This aligns with theoretical frameworks of human-AI complementarity
[Human-AI Collaboration in Strategic Decision Making, 2023] and challenges narratives of AI as
decision-making autonomous agents in high-uncertainty contexts.

**Practical implications** are significant for entrepreneurs and investors. Founders should invest
in AI tools that surface information and generate options, rather than those that issue recommendations.
Investors evaluating startups may wish to assess AI tool portfolio composition as a proxy for
organizational information processing capability.

**Limitations** must be acknowledged. Self-reported AI tool usage may be subject to social desirability
bias. The sample is limited to three English-speaking startup ecosystems, potentially limiting
generalizability to emerging markets. Decision quality ratings at 30 days may not capture long-run
outcomes. Future research should extend the time horizon and employ objective decision outcome metrics
(e.g., funding round success, customer acquisition cost).

**Future directions** include examining AI tool adoption at later startup stages, investigating the
role of team composition in moderating AI tool effects, and developing validated instruments for
measuring AI-mediated cognition in entrepreneurial settings [AI-Enabled Business Model Innovation:
Case Studies, 2023].
"""

_MOCK_ABSTRACT_TEMPLATE = """\
**Background**: The integration of artificial intelligence tools in startup operations has accelerated
significantly, yet their effects on founder decision-making remain empirically underexplored.

**Gap**: Prior research has examined AI productivity effects in corporate settings but has not
investigated the specific mechanisms through which AI tools influence decision velocity and quality
in resource-constrained entrepreneurial contexts.

**Research Question**: How does AI tool adoption affect decision-making speed and quality among
early-stage startup founders?

**Approach**: A six-month longitudinal mixed-methods study with 200 founders (Seed to Series A)
across three startup ecosystems, combining bi-weekly surveys, decision timing logs, and semi-structured
interviews.

**Contribution**: We find that AI tool adoption is associated with 31% faster decision cycles,
mediated by reduced cognitive load. Augmentative AI use outperforms substitutive use on decision
quality metrics by 26%. Results have implications for founder tool portfolio design and investor
evaluation of organizational information processing capability.
"""


def _build_mock_draft(question: dict, paper_idx: int) -> DraftPaper:
    """Build a realistic mock DraftPaper for the given question."""
    domain = question["domain"]
    domain_adj = "startup" if domain == "startup" else "enterprise"

    title = f"{question['description'].rstrip('?')}: A Longitudinal Study"

    intro = _MOCK_INTRO
    methods = _MOCK_METHODS
    results = _MOCK_RESULTS
    discussion = _MOCK_DISCUSSION
    abstract = _MOCK_ABSTRACT_TEMPLATE

    # Assemble full markdown
    content_md = f"""# {title}

## Abstract

{abstract}

## Introduction

{intro}

## Methods

{methods}

## Results

{results}

## Discussion

{discussion}

## References

"""
    for i, p in enumerate(_MOCK_PAPERS, 1):
        authors_str = ", ".join(p["authors"])
        content_md += f"{i}. {authors_str} ({p['year']}). {p['title']}. *Journal of Business and AI*, {10 + i}(2), {100 + i*12}–{120 + i*12}.\n"

    return DraftPaper(
        title=title,
        abstract=abstract,
        introduction_md=intro,
        methods_md=methods,
        results_md=results,
        discussion_md=discussion,
        citation_count=len(_MOCK_PAPERS),
        content_markdown=content_md,
    )


def _build_mock_quality_results() -> dict:
    """Build mock quality gate results."""
    return {
        "novelty": {
            "passed": True,
            "score": 0.62,
            "feedback": "Paper is sufficiently novel (similarity 0.62 < threshold 0.70)",
        },
        "citation": {
            "passed": True,
            "score": 0.88,
            "feedback": "15+ citations with adequate h-index and recency",
        },
        "peer_review": {
            "passed": True,
            "score": 7.5,
            "recommendation": "accept",
            "feedback": (
                "The paper presents a well-structured longitudinal study with clear methodology. "
                "The distinction between augmentative and substitutive AI use is a useful theoretical "
                "contribution. Minor revisions suggested: expand the limitations section and provide "
                "more detail on the thematic analysis process."
            ),
        },
    }


async def generate_mock_papers(num_questions: int, output_dir: Path) -> MetricsCollector:
    """Generate papers using mock data — no API keys required.

    Bypasses all LLM calls. Uses pre-baked fixture content but runs the
    real PaperFormatter to produce actual .md, .tex, .docx, metadata.json files.

    Args:
        num_questions: Number of test questions to generate papers for
        output_dir: Output directory

    Returns:
        MetricsCollector with results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    collector = MetricsCollector()

    questions = TEST_QUESTIONS[:min(num_questions, len(TEST_QUESTIONS))]

    logger.info("[MOCK MODE] Bypassing all LLM/API calls — using fixture data")

    for idx, question in enumerate(questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[MOCK] Paper {idx}/{len(questions)}: {question['description']}")
        logger.info(f"{'='*80}")

        start_time = datetime.utcnow()

        try:
            draft_paper = _build_mock_draft(question, idx)
            quality_results = _build_mock_quality_results()

            elapsed_time = (datetime.utcnow() - start_time).total_seconds()

            novelty_result = quality_results["novelty"]
            citation_result = quality_results["citation"]
            peer_review_result = quality_results["peer_review"]

            status = "accepted"  # All mock gates pass

            metrics = PaperMetrics(
                paper_id=f"mock_{idx:03d}",
                title=draft_paper.title,
                novelty_score=novelty_result["score"],
                citation_count=draft_paper.citation_count,
                avg_h_index=6.5,  # Mock average h-index
                recency_ratio=0.50,  # 50% of papers from last 3 years
                peer_review_score=peer_review_result["score"],
                generation_cost_usd=0.0,  # No API cost in mock mode
                generation_time_seconds=elapsed_time,
                status=status,
            )
            collector.add_paper(metrics)

            # Run the real formatter — produces actual files
            paper_output_dir = output_dir / f"paper_{idx:03d}"
            formatter = PaperFormatter()

            metadata = {
                "mode": "mock",
                "novelty": novelty_result,
                "citations": citation_result,
                "peer_review": peer_review_result,
                "domain": question["domain"],
                "keywords": question["keywords"],
                "generated_at": datetime.utcnow().isoformat(),
            }

            outputs = await formatter.save_all_formats(draft_paper, paper_output_dir, metadata)

            logger.info(f"✓ Paper saved to {paper_output_dir}")
            logger.info(f"  Title:       {draft_paper.title}")
            logger.info(f"  Status:      {status}")
            logger.info(f"  Novelty:     {novelty_result['score']:.2f} ({'PASS' if novelty_result['passed'] else 'FAIL'})")
            logger.info(f"  Citations:   {draft_paper.citation_count}")
            logger.info(f"  Peer Review: {peer_review_result['score']:.1f}/10 ({peer_review_result['recommendation']})")
            logger.info(f"  Files:       {', '.join(outputs.keys())}")

        except Exception as e:
            logger.error(f"Error generating mock paper {idx}: {e}", exc_info=True)
            continue

    return collector


async def generate_real_papers(num_questions: int, output_dir: Path) -> MetricsCollector:
    """Generate papers using the full pipeline (requires API keys and services)."""
    from research_machine.pipeline import run_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    collector = MetricsCollector()

    questions = TEST_QUESTIONS[:min(num_questions, len(TEST_QUESTIONS))]

    for idx, question in enumerate(questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Generating paper {idx}/{len(questions)}: {question['description']}")
        logger.info(f"{'='*80}")

        try:
            start_time = datetime.utcnow()
            state = await run_pipeline(
                domain=question["domain"],
                keywords=question["keywords"],
            )
            elapsed_time = (datetime.utcnow() - start_time).total_seconds()

            draft_paper = state.get("draft_paper")
            if not draft_paper:
                logger.warning(f"No draft paper generated for question {idx}")
                continue

            quality_results = state.get("quality_results") or {}
            novelty_result = quality_results.get("novelty", {})
            citation_result = quality_results.get("citation", {})
            peer_review_result = quality_results.get("peer_review", {})

            novelty_pass = novelty_result.get("passed", False)
            citation_pass = citation_result.get("passed", False)
            peer_review_pass = peer_review_result.get("passed", False)

            if novelty_pass and citation_pass and peer_review_pass:
                status = "accepted"
            elif novelty_pass or citation_pass or peer_review_pass:
                status = "revision_requested"
            else:
                status = "rejected"

            estimated_cost = 0.08 + 0.15 + 2.50 + 1.80 + 0.60

            metrics = PaperMetrics(
                paper_id=f"test_{idx:03d}",
                title=draft_paper.title,
                novelty_score=novelty_result.get("score", 0.5),
                citation_count=draft_paper.citation_count or 0,
                avg_h_index=citation_result.get("score", 0),
                recency_ratio=citation_result.get("score", 0),
                peer_review_score=peer_review_result.get("score", 0),
                generation_cost_usd=estimated_cost,
                generation_time_seconds=elapsed_time,
                status=status,
            )
            collector.add_paper(metrics)

            paper_output_dir = output_dir / f"paper_{idx:03d}"
            formatter = PaperFormatter()

            metadata = {
                "novelty": novelty_result,
                "citations": citation_result,
                "peer_review": peer_review_result,
                "domain": question["domain"],
                "keywords": question["keywords"],
                "generated_at": datetime.utcnow().isoformat(),
            }

            outputs = await formatter.save_all_formats(draft_paper, paper_output_dir, metadata)

            logger.info(f"✓ Paper saved to {paper_output_dir}")
            logger.info(f"  Title:       {draft_paper.title}")
            logger.info(f"  Status:      {status}")
            logger.info(f"  Novelty:     {novelty_result.get('score', 'N/A'):.2f}")
            logger.info(f"  Citations:   {draft_paper.citation_count}")
            logger.info(f"  Peer Review: {peer_review_result.get('score', 'N/A'):.1f}/10")
            logger.info(f"  Time: {elapsed_time:.1f}s, Cost: ${estimated_cost:.2f}")

        except Exception as e:
            logger.error(f"Error generating paper {idx}: {e}", exc_info=True)
            continue

    return collector


def _print_report(collector: MetricsCollector) -> None:
    """Print summary and detailed report."""
    summary = collector.summary_report()

    logger.info(f"\n{'='*80}")
    logger.info("GENERATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total papers: {summary['total_papers']}")
    logger.info(f"Status:       {summary['status_breakdown']}")
    logger.info(f"Avg metrics:")
    for metric, value in summary["average_metrics"].items():
        logger.info(f"  - {metric}: {value}")
    logger.info(f"Total cost:   ${summary['total_cost_usd']:.2f}")
    logger.info(f"Cost/paper:   ${summary['cost_per_paper']:.2f}")

    detailed = collector.detailed_report()
    print("\nDetailed Report:")
    print("=" * 80)
    for paper in detailed["papers"]:
        print(f"Paper: {paper['title']}")
        print(f"  ID:     {paper['id']}")
        print(f"  Status: {paper['status']}")
        print(f"  Metrics:")
        for k, v in paper["metrics"].items():
            print(f"    - {k}: {v}")
        print(f"  Gates:")
        for gate, passed in paper["gates"].items():
            icon = "✓ PASS" if passed else "✗ FAIL"
            print(f"    - {gate}: {icon}")
        print()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate test papers using the research pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock mode — no API keys needed, generates real output files:
  python scripts/generate_test_papers.py --mock --questions 3 --output-dir ./papers_demo

  # Real mode — needs ANTHROPIC_API_KEY + running Qdrant/PostgreSQL:
  python scripts/generate_test_papers.py --questions 3 --output-dir ./papers
        """,
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=3,
        help="Number of test questions to generate papers for (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./papers"),
        help="Output directory for generated papers (default: ./papers)",
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["all", "md", "tex", "pdf", "docx", "json"],
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode — bypass all LLM/API calls, use fixture data instead. "
             "No API keys or external services required. Useful for testing output formats.",
    )

    args = parser.parse_args()

    mode_label = "[MOCK]" if args.mock else "[REAL]"
    logger.info(f"{mode_label} Starting paper generation")
    logger.info(f"  Questions:  {args.questions}")
    logger.info(f"  Output dir: {args.output_dir}")
    if args.mock:
        logger.info("  Mode: MOCK — no API calls will be made")
    else:
        logger.info("  Mode: REAL — requires ANTHROPIC_API_KEY + Qdrant + PostgreSQL")

    if args.mock:
        collector = await generate_mock_papers(args.questions, args.output_dir)
    else:
        collector = await generate_real_papers(args.questions, args.output_dir)

    _print_report(collector)


if __name__ == "__main__":
    asyncio.run(main())
