#!/usr/bin/env python3
"""Build the local Q1 paper library the engine learns from.

Harvests legally available full texts (Semantic Scholar OA -> Unpaywall ->
open-licence publisher links) from a list of Q1 journals, or from an explicit
DOI list. Paywalled works are queued in needs_library_access.csv for a human
to fetch through their institution — nothing is scraped past a paywall.

    # walk Q1 entrepreneurship/management journals
    python scripts/harvest_library.py --journals q1-entrepreneurship --from-year 2019

    # only papers about a topic, across those journals
    python scripts/harvest_library.py --journals q1-entrepreneurship --topic "artificial intelligence"

    # rebuild a specific seed set
    python scripts/harvest_library.py --dois-from q1_ai_entrepreneurship/README.md
"""

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_machine.scholar.harvester import OALibrary  # noqa: E402

# Journal sets. Names are resolved against Crossref at runtime, so a rename or
# a near-miss shows up as "could not resolve" rather than silently harvesting
# the wrong publication.
JOURNAL_SETS = {
    "q1-entrepreneurship": [
        "Small Business Economics",
        "Journal of Business Venturing",
        "Journal of Business Venturing Insights",
        "Entrepreneurship Theory and Practice",
        "International Journal of Entrepreneurial Behaviour & Research",
        "International Small Business Journal",
        "Strategic Entrepreneurship Journal",
        "Journal of Small Business Management",
        "Review of Managerial Science",
    ],
    "q1-innovation": [
        "Technovation",
        "Research Policy",
        "Technological Forecasting and Social Change",
        "Journal of Product Innovation Management",
        "Industrial Marketing Management",
    ],
    "q1-management": [
        "Journal of Business Research",
        "British Journal of Management",
        "Journal of Management Studies",
        "Long Range Planning",
        "Journal of Business Ethics",
    ],
}
JOURNAL_SETS["all"] = sum(JOURNAL_SETS.values(), [])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default="library", help="library root (default: ./library)")
    ap.add_argument("--email", default="duy.bui@eiu.edu.vn",
                    help="contact email for the Crossref/Unpaywall polite pools")
    ap.add_argument("--journals", choices=sorted(JOURNAL_SETS),
                    help="journal set to walk")
    ap.add_argument("--dois-from", help="file containing DOIs (any text; DOIs are extracted)")
    ap.add_argument("--topic", help="restrict to works matching this topic")
    ap.add_argument("--from-year", type=int, default=2019)
    ap.add_argument("--per-journal", type=int, default=15,
                    help="max full texts to keep per journal")
    ap.add_argument("--scan-cap", type=int, default=60,
                    help="max works to examine per journal")
    args = ap.parse_args()

    if not args.journals and not args.dois_from:
        ap.error("give --journals or --dois-from")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    lib = OALibrary(args.library, args.email)
    try:
        if args.dois_from:
            text = Path(args.dois_from).read_text()
            dois = re.findall(r"10\.\d{4,9}/[^\s\"'<>)\]]+", text)
            dois = list(dict.fromkeys(d.rstrip(".,;") for d in dois))
            print(f"Found {len(dois)} DOIs in {args.dois_from}")
            lib.harvest_dois(dois)
        if args.journals:
            lib.harvest_journals(
                JOURNAL_SETS[args.journals], from_year=args.from_year,
                topic=args.topic, per_journal=args.per_journal, scan_cap=args.scan_cap,
            )
    finally:
        lib.save()
        print(f"\nLibrary: {len(lib.entries)} works indexed | "
              f"{lib.n_with_pdf} full texts | {len(lib.paywalled)} need library access")
        print(f"  manifest : {lib.manifest_path}")
        print(f"  paywalled: {lib.paywalled_path}")
        lib.close()


if __name__ == "__main__":
    main()
