#!/usr/bin/env python3
"""
Monitor paper generation metrics across all runs.

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --log-file ./logs/paper_metrics.jsonl
    python scripts/monitor.py --tail 10
    python scripts/monitor.py --filter-status accepted
    python scripts/monitor.py --summary
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


DEFAULT_LOG = Path(__file__).parent.parent / "logs" / "paper_metrics.jsonl"


def _load_metrics(log_file: Path) -> list:
    if not log_file.exists():
        return []
    records = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _print_summary(records: list):
    if not records:
        print("  No records found.")
        return

    total = len(records)
    by_status = defaultdict(int)
    for r in records:
        by_status[r.get("status", "unknown")] += 1

    accepted = [r for r in records if r.get("status") == "accepted"]
    domains = defaultdict(int)
    for r in records:
        domains[r.get("domain", "?")] += 1

    print(f"  Total papers generated : {total}")
    print(f"  By status:")
    for status, count in sorted(by_status.items()):
        pct = count / total * 100
        print(f"    {status:25s}: {count} ({pct:.0f}%)")
    print(f"  By domain:")
    for domain, count in sorted(domains.items()):
        print(f"    {domain:25s}: {count}")

    if accepted:
        avg_words = sum(r.get("word_count", 0) for r in accepted) / len(accepted)
        avg_corpus = sum(r.get("corpus_size", 0) for r in accepted) / len(accepted)
        avg_pr = sum(r.get("peer_review_score", 0) for r in accepted) / len(accepted)
        avg_fc = sum(r.get("fact_check_score", 0) for r in accepted) / len(accepted)
        avg_time = sum(r.get("elapsed_seconds", 0) for r in accepted) / len(accepted)
        total_cost = sum(r.get("cost_usd", 0) for r in records)
        print(f"\n  Accepted papers averages:")
        print(f"    Word count     : {avg_words:,.0f}")
        print(f"    Corpus size    : {avg_corpus:.1f} papers")
        print(f"    Peer review    : {avg_pr:.1f}/10")
        print(f"    Fact check     : {avg_fc:.1f}/10")
        print(f"    Generation time: {avg_time:.1f}s")
        print(f"\n  Total API cost   : ${total_cost:.4f}")


def _print_table(records: list, tail: int = None):
    if tail:
        records = records[-tail:]
    if not records:
        print("  No records.")
        return

    header = f"{'#':>4}  {'Date':10}  {'Domain':10}  {'Status':20}  {'Words':>6}  {'PR':>5}  {'FC':>5}  {'Time':>6}  Title"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(records, 1):
        date = (r.get("generated_at", "") or "")[:10]
        domain = r.get("domain", "?")[:10]
        status = r.get("status", "?")[:20]
        words = r.get("word_count", 0)
        pr = r.get("peer_review_score", 0)
        fc = r.get("fact_check_score", 0)
        elapsed = r.get("elapsed_seconds", 0)
        title = (r.get("title", "?") or "")[:45]
        print(f"{i:>4}  {date:10}  {domain:10}  {status:20}  {words:>6,}  {pr:>5.1f}  {fc:>5.1f}  {elapsed:>5.1f}s  {title}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor paper generation metrics from logs/paper_metrics.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all papers:
  python scripts/monitor.py

  # Show summary stats only:
  python scripts/monitor.py --summary

  # Show last 10 papers:
  python scripts/monitor.py --tail 10

  # Show only accepted papers:
  python scripts/monitor.py --filter-status accepted

  # Custom log file:
  python scripts/monitor.py --log-file ./logs/paper_metrics.jsonl
        """,
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG, help="Path to JSONL metrics log")
    parser.add_argument("--tail", type=int, help="Show only last N records")
    parser.add_argument("--filter-status", help="Filter by status (accepted, revision_requested, rejected)")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics only")

    args = parser.parse_args()

    records = _load_metrics(args.log_file)

    print(f"\n{'='*70}")
    print(f"  PAPER GENERATION MONITOR")
    print(f"  Log: {args.log_file}")
    print(f"{'='*70}\n")

    if not records:
        print("  No papers logged yet. Run the engine to generate papers first.\n")
        print(f"  Expected log location: {args.log_file}\n")
        sys.exit(0)

    if args.filter_status:
        records = [r for r in records if r.get("status") == args.filter_status]
        print(f"  Filtered to status='{args.filter_status}': {len(records)} papers\n")

    if args.summary:
        _print_summary(records)
    else:
        _print_table(records, tail=args.tail)
        print()
        _print_summary(records)

    print()


if __name__ == "__main__":
    main()
