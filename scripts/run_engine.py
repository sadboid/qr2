#!/usr/bin/env python3
"""
Local Research Engine — no LLM API needed.

Uses real Semantic Scholar + arXiv data (free) with extractive synthesis
to generate full IMRAD research papers.

Usage:
    python scripts/run_engine.py --topic "AI tools founder productivity" --domain startup
    python scripts/run_engine.py --topic "LLM governance enterprise" --domain enterprise
    python scripts/run_engine.py  # uses default AI + entrepreneurship topic
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_machine.local_engine import LocalResearchEngine
from research_machine.output.formatter import PaperFormatter
from research_machine.agents.writing_agent import DraftPaper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy sub-loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Pre-defined topics
# ---------------------------------------------------------------------------

TOPICS = {
    "ai_startup": {
        "question": "How do AI tools affect founder decision-making speed and quality in early-stage startups?",
        "keywords": ["AI tools", "founder productivity", "startup decision making", "artificial intelligence", "entrepreneurship"],
        "domain": "startup",
    },
    "ml_failure": {
        "question": "Can machine learning models predict startup failure with early warning signals?",
        "keywords": ["startup failure prediction", "machine learning", "early warning", "entrepreneurship analytics"],
        "domain": "startup",
    },
    "llm_enterprise": {
        "question": "What organizational structures best support LLM governance and responsible AI adoption?",
        "keywords": ["LLM governance", "enterprise AI", "responsible AI", "organizational structure"],
        "domain": "enterprise",
    },
    "ai_roi": {
        "question": "How do enterprises measure ROI from AI adoption and what factors drive value creation?",
        "keywords": ["AI ROI", "enterprise value", "machine learning business impact", "digital transformation"],
        "domain": "enterprise",
    },
}


async def run(args):
    # Determine research question
    if args.topic_key and args.topic_key in TOPICS:
        topic = TOPICS[args.topic_key]
        question = topic["question"]
        keywords = topic["keywords"]
        domain = topic["domain"]
    else:
        question = args.question or TOPICS["ai_startup"]["question"]
        keywords = [k.strip() for k in (args.keywords or "AI tools,startup,entrepreneurship,decision making").split(",")]
        domain = args.domain

    print("\n" + "=" * 72)
    print("  LOCAL RESEARCH ENGINE  (no LLM API required)")
    print("=" * 72)
    print(f"  Question : {question}")
    print(f"  Domain   : {domain}")
    print(f"  Keywords : {', '.join(keywords[:4])}")
    print(f"  Output   : {args.output_dir}")
    print("=" * 72 + "\n")

    # Run engine
    engine = LocalResearchEngine(target_corpus_size=35)
    result = await engine.run(
        research_question=question,
        keywords=keywords,
        domain=domain,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    paper_dir = output_dir / "paper_001"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Build a DraftPaper-compatible object for formatter
    class _FakeDraft:
        def __init__(self, r):
            self.title = r.title
            self.abstract = r.paper_data.get("abstract", "")
            self.introduction_md = r.paper_data.get("introduction_md", "")
            self.methods_md = r.paper_data.get("methods_md", "")
            self.results_md = r.paper_data.get("results_md", "")
            self.discussion_md = r.paper_data.get("discussion_md", "")
            self.citations = r.synthesis.top_papers
            self.citation_count = r.paper_data.get("citation_count", 0)
            self.content_markdown = r.paper_data.get("content_markdown", "")
            self.content_latex = ""

    draft = _FakeDraft(result)
    formatter = PaperFormatter()

    metadata = {
        **result.to_metadata()["metadata"],
        "overall_status": result.status,
        "generation_metrics": result.to_metadata()["generation_metrics"],
    }

    outputs = await formatter.save_all_formats(draft, paper_dir, metadata)

    # Save metadata.json (rich version)
    metadata_full = result.to_metadata()
    meta_path = paper_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_full, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    qr = result.quality_results
    print("\n" + "=" * 72)
    print("  PAPER GENERATED")
    print("=" * 72)
    print(f"  Title        : {result.title}")
    print(f"  Status       : {result.status.upper()}")
    print(f"  Words        : {result.word_count:,}")
    print(f"  Corpus       : {result.corpus_size} real papers from Semantic Scholar + arXiv")
    print(f"  Elapsed      : {result.elapsed_seconds:.1f}s")
    print(f"  Cost         : $0.00  (no LLM API used)")
    print()
    print("  Quality Gates:")
    nv = qr["novelty"]
    print(f"    Novelty     : {nv['score']:.2f}  {'✓ PASS' if nv['passed'] else '✗ FAIL'}  ({nv['feedback'][:60]})")
    ct = qr["citation"]
    print(f"    Citations   : {draft.citation_count}     {'✓ PASS' if ct['passed'] else '✗ FAIL'}  (recency {ct['metrics']['recency_ratio']:.0%}, h≈{ct['metrics']['avg_h_index']:.1f})")
    pr = qr["peer_review"]
    print(f"    Peer Review : {pr['score']:.1f}/10 {'✓ PASS' if pr['passed'] else '✗ FAIL'}  ({pr['recommendation']})")
    print()
    print("  Output Files:")
    for fmt, path in outputs.items():
        size_kb = Path(path).stat().st_size // 1024
        print(f"    {fmt:10s}: {path}  ({size_kb} KB)")
    print(f"    {'metadata':10s}: {meta_path}")
    print("=" * 72 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Local Research Engine — generates papers from real academic data without LLM API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default AI + startup topic:
  python scripts/run_engine.py

  # Use a pre-defined topic key:
  python scripts/run_engine.py --topic-key ai_startup
  python scripts/run_engine.py --topic-key llm_enterprise
  python scripts/run_engine.py --topic-key ml_failure

  # Custom question:
  python scripts/run_engine.py \\
    --question "How does AI adoption affect startup survival rates?" \\
    --keywords "AI adoption,startup survival,entrepreneurship" \\
    --domain startup \\
    --output-dir ./papers_engine

Available topic keys: ai_startup, ml_failure, llm_enterprise, ai_roi
        """,
    )
    parser.add_argument("--topic-key", help="Pre-defined topic key (ai_startup, ml_failure, llm_enterprise, ai_roi)")
    parser.add_argument("--question", help="Custom research question")
    parser.add_argument("--keywords", help="Comma-separated keywords")
    parser.add_argument("--domain", default="startup", choices=["startup", "enterprise"], help="Domain context")
    parser.add_argument("--output-dir", default="./papers_engine", help="Output directory")

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
