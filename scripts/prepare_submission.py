#!/usr/bin/env python3
"""
Submission preparation — check a paper directory against venue requirements
and generate venue-formatted LaTeX output.

Usage:
    python scripts/prepare_submission.py --paper-dir ./papers_batch/paper_001 --venue elsevier
    python scripts/prepare_submission.py --paper-dir ./papers_batch/paper_001 --venue ieee --output-dir ./submission
    python scripts/prepare_submission.py --paper-dir ./papers_batch/paper_001 --list-venues
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research_machine.output.venue_formatter import (
    VENUES,
    generate_venue_checklist,
    format_ieee_latex,
    format_acm_latex,
    format_elsevier_latex,
    build_references_block,
)


def _load_metadata(paper_dir: Path) -> dict:
    meta_path = paper_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata.json not found in {paper_dir}")
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def _load_markdown(paper_dir: Path) -> str:
    md_path = paper_dir / "paper.md"
    if not md_path.exists():
        print(f"ERROR: paper.md not found in {paper_dir}")
        sys.exit(1)
    return md_path.read_text(encoding="utf-8")


def _count_words(text: str) -> int:
    return len(text.split())


def _extract_abstract_words(metadata: dict) -> int:
    abstract = metadata.get("abstract", "")
    return _count_words(abstract)


def _print_checklist(checklist: dict):
    print(f"\n  Venue: {checklist['venue']}")
    print(f"  Submission-ready: {'✓ YES' if checklist['ready_for_submission'] else '✗ NO'}")
    print(f"  Checks passed: {checklist['passed_count']}/{checklist['total_checks']}")
    print()
    for key, check in checklist["checks"].items():
        icon = "✓" if check["passed"] else "✗"
        print(f"    {icon} {check['label']:25s}: {check['value']} (req: {check['requirement']})")
    if checklist.get("double_blind"):
        print()
        print("  ⚠  This venue uses DOUBLE-BLIND review — remove author names before submission.")
    print()
    print(f"  Submission notes: {checklist['submission_notes']}")


def _build_fake_papers(metadata: dict) -> list:
    """Build a list of paper-like objects for reference building."""
    class _FakePaper:
        def __init__(self, d):
            self.authors = d.get("authors", ["Unknown"])
            self.title = d.get("title", "Unknown")
            self.year = d.get("year", 2024)
            self.venue = d.get("venue", "")

    refs = metadata.get("citations", {}).get("references", [])
    return [_FakePaper(r) for r in refs]


def run(args):
    if args.list_venues:
        print("\n  Available venues:")
        for key, spec in VENUES.items():
            db = " (double-blind)" if spec.double_blind else ""
            print(f"    {key:12s}: {spec.name}{db}")
            print(f"              Max words: {spec.max_words:,}  Min citations: {spec.min_citations}")
        print()
        return

    if not args.paper_dir:
        print("ERROR: --paper-dir is required when not using --list-venues")
        sys.exit(1)

    paper_dir = Path(args.paper_dir)
    if not paper_dir.exists():
        print(f"ERROR: Paper directory not found: {paper_dir}")
        sys.exit(1)

    venue_key = args.venue
    if venue_key not in VENUES:
        print(f"ERROR: Unknown venue '{venue_key}'. Use --list-venues to see options.")
        sys.exit(1)

    metadata = _load_metadata(paper_dir)
    content_md = _load_markdown(paper_dir)

    title = metadata.get("metadata", {}).get("title") or metadata.get("abstract", "")[:60]
    abstract = metadata.get("abstract", "")
    keywords = metadata.get("metadata", {}).get("keywords", [])
    quality_results = {
        "novelty": metadata.get("metadata", {}).get("novelty", {}),
        "peer_review": metadata.get("metadata", {}).get("peer_review", {}),
        "fact_check": metadata.get("metadata", {}).get("fact_check", {}),
        "citation": metadata.get("metadata", {}).get("citation", {}),
    }

    word_count = _count_words(content_md)
    abstract_words = _extract_abstract_words(metadata)
    citation_count = metadata.get("citations", {}).get("count", 0)

    print("\n" + "=" * 70)
    print("  SUBMISSION PREPARATION REPORT")
    print("=" * 70)
    print(f"  Paper    : {title[:60]}")
    print(f"  Location : {paper_dir}")
    print(f"  Words    : {word_count:,}")
    print(f"  Abstract : {abstract_words} words")
    print(f"  Citations: {citation_count}")

    checklist = generate_venue_checklist(
        word_count=word_count,
        citation_count=citation_count,
        abstract_words=abstract_words,
        quality_results=quality_results,
        venue_key=venue_key,
    )
    _print_checklist(checklist)

    # Generate formatted LaTeX
    output_dir = Path(args.output_dir) if args.output_dir else paper_dir / "submission"
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = _build_fake_papers(metadata)
    refs_block = build_references_block(papers)

    if venue_key == "ieee":
        latex = format_ieee_latex(title, abstract, content_md, keywords, refs_block)
    elif venue_key == "acm":
        latex = format_acm_latex(title, abstract, content_md, keywords, refs_block)
    else:
        latex = format_elsevier_latex(title, abstract, content_md, keywords, refs_block)

    latex_out = output_dir / f"paper_{venue_key}.tex"
    latex_out.write_text(latex, encoding="utf-8")

    # Save checklist JSON
    checklist_out = output_dir / f"checklist_{venue_key}.json"
    with open(checklist_out, "w") as f:
        json.dump(checklist, f, indent=2)

    print(f"  Output files:")
    print(f"    LaTeX    : {latex_out}")
    print(f"    Checklist: {checklist_out}")

    if checklist["ready_for_submission"]:
        print("\n  ✓ Paper is ready for submission to this venue.")
    else:
        failed = [c["label"] for c in checklist["checks"].values() if not c["passed"]]
        print(f"\n  ✗ Not ready. Fix these issues: {', '.join(failed)}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check submission readiness and generate venue-formatted output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check against Elsevier requirements:
  python scripts/prepare_submission.py --paper-dir ./papers_batch/paper_001 --venue elsevier

  # Generate IEEE-formatted LaTeX:
  python scripts/prepare_submission.py --paper-dir ./papers_batch/paper_001 --venue ieee

  # Save to a specific submission folder:
  python scripts/prepare_submission.py --paper-dir ./papers_batch/paper_001 --venue acm --output-dir ./submission/paper1_acm

  # List all supported venues:
  python scripts/prepare_submission.py --list-venues
        """,
    )
    parser.add_argument("--paper-dir", help="Path to the paper directory (containing paper.md + metadata.json)")
    parser.add_argument("--venue", default="elsevier", choices=list(VENUES.keys()), help="Target venue style")
    parser.add_argument("--output-dir", help="Output directory for formatted files (default: <paper-dir>/submission/)")
    parser.add_argument("--list-venues", action="store_true", help="List all supported venues and exit")

    args = parser.parse_args()
    if not args.list_venues and not args.paper_dir:
        parser.error("--paper-dir is required unless using --list-venues")

    run(args)


if __name__ == "__main__":
    main()
