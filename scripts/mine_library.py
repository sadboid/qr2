#!/usr/bin/env python3
"""Learn from the harvested library.

Runs the three analyses that turn a folder of PDFs into things the engine can
use, and prints what changed about its beliefs:

  style_profile.json  how Q1 papers write   -> injected into writer prompts
  style_norms.json    how Q1 prose lints    -> calibrates the style gate
  research_graph.json what has been studied -> gap detection

    python scripts/mine_library.py                # all three
    python scripts/mine_library.py --only norms   # just one
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_machine.scholar.research_graph import build_from_library  # noqa: E402
from research_machine.scholar.rhetoric_miner import mine_library  # noqa: E402
from research_machine.scholar.style_norms import build_norms  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default="library")
    ap.add_argument("--only", choices=["style", "norms", "graph"],
                    help="run a single analysis instead of all three")
    ap.add_argument("--min-papers", type=int, default=3,
                    help="a pattern must appear in this many DIFFERENT papers to be reported")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lib = Path(args.library)
    if not (lib / "manifest.json").exists():
        sys.exit(f"No library at {lib} — run scripts/harvest_library.py first.")

    manifest = json.loads((lib / "manifest.json").read_text())
    print(f"Library: {manifest['count']} works indexed, {manifest['with_pdf']} full texts\n")

    run = lambda name: args.only in (None, name)  # noqa: E731

    if run("style"):
        prof = mine_library(str(lib), str(lib / "style_profile.json"),
                            min_papers=args.min_papers)
        c = prof["corpus"]
        print(f"\nSTYLE PROFILE — {c['n_papers']} papers mined, genres {c['genres']}")
        for genre, secs in prof["sections"].items():
            if genre == "any":
                for sec, d in secs.items():
                    print(f"  {sec:13} n={d['n_papers']:>2} | {d['words_median']:>5} words "
                          f"| {d['citations_per_1k']:>5} cites/1k "
                          f"| {d['words_per_sentence']:>4} w/sent "
                          f"| {d['sentences_per_paragraph']:>4} sent/para")
        print(f"  gap sentences mined: {len(prof['moves']['gap_statements'])}, "
              f"contribution: {len(prof['moves']['contribution_statements'])}")

    if run("norms"):
        norms = build_norms(str(lib), str(lib / "style_norms.json"))
        if norms:
            c = norms["corpus"]
            print(f"\nSTYLE NORMS — {c['n_papers']} papers, {c['words_analysed']:,} words "
                  f"({c['style_outliers_trimmed']} style outliers trimmed)")
            print("  what published Q1 prose actually contains, per 1,000 words:")
            for cat, v in list(norms["norms"].items())[:8]:
                print(f"    {cat:46} median {v['median_per_1k']:>5} | p90 {v['p90_per_1k']:>5} "
                      f"| {v['papers_triggering']} papers")

    if run("graph"):
        report = build_from_library(str(lib), str(lib / "research_graph.json"))
        c = report["corpus"]
        print(f"\nRESEARCH GRAPH — {c['n_coded_papers']} papers coded")
        for dim in ("theories", "methods", "populations", "outcomes"):
            top = ", ".join(f"{k} ({n})" for k, n in c[dim][:5]) or "none detected"
            print(f"  {dim:12}: {top}")
        gaps = report["gaps"]
        total = sum(len(v) for v in gaps.values())
        print(f"  gaps found: {total}")
        for kind, items in gaps.items():
            for g in items[:2]:
                print(f"    [{kind}] {g['claim']}")
        if total == 0:
            print("    (none — the corpus is too thin to support a gap claim; harvest more)")

    print("\nThe engine picks these up automatically on the next run.")


if __name__ == "__main__":
    main()
