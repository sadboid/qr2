#!/usr/bin/env python3
"""Script to generate test papers end-to-end.

Usage:
    python scripts/generate_test_papers.py --questions 3 --output-dir ./papers
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

from research_machine.pipeline import run_pipeline
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


async def generate_test_papers(num_questions: int, output_dir: Path) -> MetricsCollector:
    """Generate test papers using the full pipeline.

    Args:
        num_questions: Number of test questions to run
        output_dir: Output directory for papers

    Returns:
        MetricsCollector with results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    collector = MetricsCollector()

    # Select test questions
    questions = TEST_QUESTIONS[:min(num_questions, len(TEST_QUESTIONS))]

    for idx, question in enumerate(questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Generating paper {idx}/{len(questions)}: {question['description']}")
        logger.info(f"{'='*80}")

        paper_start_time = datetime.utcnow()

        try:
            # Run the full pipeline
            start_time = datetime.utcnow()
            state = await run_pipeline(
                domain=question["domain"],
                keywords=question["keywords"],
            )
            elapsed_time = (datetime.utcnow() - start_time).total_seconds()

            # Check if paper was generated
            draft_paper = state.get("draft_paper")
            if not draft_paper:
                logger.warning(f"No draft paper generated for question {idx}")
                continue

            # Extract quality results
            quality_results = state.get("quality_results", {})
            novelty_result = quality_results.get("novelty", {})
            citation_result = quality_results.get("citation", {})
            peer_review_result = quality_results.get("peer_review", {})

            # Determine status based on gate results
            novelty_pass = novelty_result.get("passed", False)
            citation_pass = citation_result.get("passed", False)
            peer_review_pass = peer_review_result.get("passed", False)

            if novelty_pass and citation_pass and peer_review_pass:
                status = "accepted"
            elif novelty_pass or citation_pass or peer_review_pass:
                status = "revision_requested"
            else:
                status = "rejected"

            # Calculate estimated cost (rough)
            # Hypothesis: $0.08, Literature: $0.15, Analysis: $2.50, Writing: $1.80, Peer Review: $0.60
            estimated_cost = 0.08 + 0.15 + 2.50 + 1.80 + 0.60

            # Create metrics
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

            # Save paper in all formats
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

            outputs = await formatter.save_all_formats(
                draft_paper,
                paper_output_dir,
                metadata,
            )

            logger.info(f"✓ Paper saved to {paper_output_dir}")
            logger.info(f"  - Title: {draft_paper.title}")
            logger.info(f"  - Status: {status}")
            logger.info(f"  - Novelty: {novelty_result.get('score', 'N/A'):.2f}")
            logger.info(f"  - Citations: {draft_paper.citation_count}")
            logger.info(f"  - Peer Review: {peer_review_result.get('score', 'N/A'):.1f}/10")
            logger.info(f"  - Time: {elapsed_time:.1f}s, Cost: ${estimated_cost:.2f}")

        except Exception as e:
            logger.error(f"Error generating paper {idx}: {e}", exc_info=True)
            continue

    return collector


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test papers using the research pipeline.")
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
        help="Output format(s) (default: all)",
    )

    args = parser.parse_args()

    logger.info(f"Starting test paper generation")
    logger.info(f"Questions: {args.questions}")
    logger.info(f"Output directory: {args.output_dir}")

    # Generate papers
    collector = await generate_test_papers(args.questions, args.output_dir)

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("GENERATION SUMMARY")
    logger.info(f"{'='*80}")

    summary = collector.summary_report()
    logger.info(f"Total papers generated: {summary['total_papers']}")
    logger.info(f"Status breakdown: {summary['status_breakdown']}")
    logger.info(f"Average metrics:")
    for metric, value in summary['average_metrics'].items():
        logger.info(f"  - {metric}: {value}")
    logger.info(f"Total cost: ${summary['total_cost_usd']:.2f}")
    logger.info(f"Cost per paper: ${summary['cost_per_paper']:.2f}")

    # Detailed report
    detailed = collector.detailed_report()
    print("\nDetailed Report:")
    print("=" * 80)
    for paper in detailed["papers"]:
        print(f"Paper: {paper['title']}")
        print(f"  ID: {paper['id']}")
        print(f"  Status: {paper['status']}")
        print(f"  Metrics:")
        for k, v in paper['metrics'].items():
            print(f"    - {k}: {v}")
        print(f"  Gates:")
        for gate, passed in paper['gates'].items():
            print(f"    - {gate}: {'✓ PASS' if passed else '✗ FAIL'}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
