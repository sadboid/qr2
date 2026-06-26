#!/usr/bin/env python3
"""
Batch paper generation — run multiple topics sequentially.

Usage:
    python scripts/batch_generate.py --domain startup
    python scripts/batch_generate.py --domain enterprise --max-papers 4
    python scripts/batch_generate.py --topics ai_startup ml_failure
    python scripts/batch_generate.py --all
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path as _P
_env = _P(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _v.strip():
                os.environ.setdefault(_k.strip(), _v.strip())

from research_machine.local_engine import LocalResearchEngine
from research_machine.output.formatter import PaperFormatter
from research_machine.agents.writing_agent import DraftPaper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _load_domain_topics(domain: str) -> list:
    base = Path(__file__).parent.parent / "research_machine" / "domain_knowledge"
    path = base / f"{domain}_topics.json"
    if not path.exists():
        logger.warning(f"No topic file for domain '{domain}': {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("topics", [])


def _filter_topics(topics: list, ids: list = None, max_papers: int = None, priority: str = None) -> list:
    if ids:
        topics = [t for t in topics if t["id"] in ids]
    if priority:
        topics = [t for t in topics if t.get("priority") == priority]
    if max_papers:
        topics = topics[:max_papers]
    return topics


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


async def generate_one(topic: dict, output_dir: Path, paper_num: int, target_corpus_size: int = 50) -> dict:
    """Run engine for a single topic; return summary dict."""
    paper_dir = output_dir / f"paper_{paper_num:03d}"
    paper_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        engine = LocalResearchEngine(target_corpus_size=target_corpus_size)
        result = await engine.run(
            research_question=topic["question"],
            keywords=topic["keywords"],
            domain=topic.get("domain_key", topic.get("id", "startup").split("_")[0]),
        )

        draft = _FakeDraft(result)
        formatter = PaperFormatter()
        metadata = {
            **result.to_metadata()["metadata"],
            "overall_status": result.status,
            "generation_metrics": result.to_metadata()["generation_metrics"],
            "topic_id": topic["id"],
        }
        await formatter.save_all_formats(draft, paper_dir, metadata)

        meta_full = result.to_metadata()
        meta_full["topic_id"] = topic["id"]
        with open(paper_dir / "metadata.json", "w") as f:
            json.dump(meta_full, f, indent=2, ensure_ascii=False, default=str)

        elapsed = time.time() - t0
        qr = result.quality_results
        fc = qr.get("fact_check", {})
        return {
            "topic_id": topic["id"],
            "title": result.title,
            "status": result.status,
            "word_count": result.word_count,
            "corpus_size": result.corpus_size,
            "citations": result.paper_data.get("citation_count", 0),
            "novelty_score": qr["novelty"]["score"],
            "peer_review_score": qr["peer_review"]["score"],
            "fact_check_score": fc.get("score", 0.0),
            "fact_check_passed": fc.get("passed", True),
            "elapsed_s": round(elapsed, 1),
            "output_dir": str(paper_dir),
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Failed on topic {topic['id']}: {e}", exc_info=True)
        return {
            "topic_id": topic["id"],
            "title": topic.get("title", "?"),
            "status": "error",
            "word_count": 0,
            "corpus_size": 0,
            "citations": 0,
            "novelty_score": 0.0,
            "peer_review_score": 0.0,
            "fact_check_score": 0.0,
            "fact_check_passed": False,
            "elapsed_s": round(elapsed, 1),
            "output_dir": str(paper_dir),
            "error": str(e),
        }


def _print_report(results: list, output_dir: Path):
    print("\n" + "=" * 80)
    print("  BATCH GENERATION REPORT")
    print("=" * 80)
    print(f"  Output directory : {output_dir}")
    print(f"  Papers attempted : {len(results)}")
    accepted = [r for r in results if r["status"] == "accepted"]
    errors = [r for r in results if r["error"]]
    print(f"  Accepted         : {len(accepted)}")
    print(f"  Errors           : {len(errors)}")
    print()

    for i, r in enumerate(results, 1):
        status_icon = "✓" if r["status"] == "accepted" else ("✗" if r["error"] else "~")
        print(f"  {i}. [{status_icon}] {r['title'][:65]}")
        if r["error"]:
            print(f"       ERROR: {r['error'][:70]}")
        else:
            fc_icon = "✓" if r["fact_check_passed"] else "✗"
            print(
                f"       Status: {r['status']:20s}  Words: {r['word_count']:,}  "
                f"Corpus: {r['corpus_size']}  Elapsed: {r['elapsed_s']}s"
            )
            print(
                f"       Novelty: {r['novelty_score']:.2f}  "
                f"PeerReview: {r['peer_review_score']:.1f}/10  "
                f"FactCheck: {r['fact_check_score']:.1f}/10 [{fc_icon}]"
            )
        print()

    if accepted:
        avg_words = sum(r["word_count"] for r in accepted) / len(accepted)
        avg_pr = sum(r["peer_review_score"] for r in accepted) / len(accepted)
        avg_fc = sum(r["fact_check_score"] for r in accepted) / len(accepted)
        total_time = sum(r["elapsed_s"] for r in results)
        print(f"  Averages (accepted papers):")
        print(f"    Word count   : {avg_words:,.0f}")
        print(f"    Peer Review  : {avg_pr:.1f}/10")
        print(f"    Fact Check   : {avg_fc:.1f}/10")
        print(f"  Total elapsed  : {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 80 + "\n")

    # Save batch report
    report_path = output_dir / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump({"papers": results, "summary": {
            "total": len(results),
            "accepted": len(accepted),
            "errors": len(errors),
        }}, f, indent=2, ensure_ascii=False)
    print(f"  Report saved: {report_path}\n")


async def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect topics
    topics = []
    if args.all:
        for domain in ["startup", "enterprise"]:
            for t in _load_domain_topics(domain):
                t["domain_key"] = domain
                topics.append(t)
    elif args.domain:
        for t in _load_domain_topics(args.domain):
            t["domain_key"] = args.domain
            topics.append(t)
    elif args.topics:
        for domain in ["startup", "enterprise"]:
            for t in _load_domain_topics(domain):
                if t["id"] in args.topics:
                    t["domain_key"] = domain
                    topics.append(t)

    topics = _filter_topics(topics, max_papers=args.max_papers, priority=args.priority)

    if not topics:
        print("No topics matched. Use --domain startup|enterprise, --all, or --topics <id>...")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BATCH GENERATION: {len(topics)} topic(s)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    results = []
    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Generating: {topic['title'][:60]}...")
        r = await generate_one(topic, output_dir, i, target_corpus_size=args.target_corpus_size)
        results.append(r)
        status = "✓ ACCEPTED" if r["status"] == "accepted" else f"✗ {r['status'].upper()}"
        print(f"  → {status}  ({r['word_count']:,} words, {r['elapsed_s']}s)")

    _print_report(results, output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch paper generation from domain knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all startup topics:
  python scripts/batch_generate.py --domain startup

  # Generate top 3 enterprise topics:
  python scripts/batch_generate.py --domain enterprise --max-papers 3

  # Generate high-priority topics from both domains:
  python scripts/batch_generate.py --all --priority high

  # Generate specific topic IDs:
  python scripts/batch_generate.py --topics startup_ai_founder startup_failure_ml

  # Full batch run (all topics, both domains):
  python scripts/batch_generate.py --all --output-dir ./papers_batch
        """,
    )
    parser.add_argument("--domain", choices=["startup", "enterprise"], help="Generate all topics for a domain")
    parser.add_argument("--all", action="store_true", help="Generate all topics (both domains)")
    parser.add_argument("--topics", nargs="+", help="Specific topic IDs to generate")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to generate")
    parser.add_argument("--priority", choices=["high", "medium", "low"], help="Filter by topic priority")
    parser.add_argument("--output-dir", default="./papers_batch", help="Output directory")
    parser.add_argument("--target-corpus-size", type=int, default=50, help="Papers per corpus (default: 50)")

    args = parser.parse_args()
    if not any([args.domain, args.all, args.topics]):
        parser.error("Specify --domain, --all, or --topics")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
