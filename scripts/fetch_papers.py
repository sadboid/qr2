#!/usr/bin/env python3
"""Academic search machine: find open-access FULL-TEXT papers on a topic and
download them into a study folder with a metadata index.

Sources: OpenAlex (metadata + OA locations) → Unpaywall fallback (DOI → OA PDF).
Prioritizes highly-cited papers in reputable (Q1-tier) venues.

Usage:
  python scripts/fetch_papers.py \
    --query "entrepreneurship artificial intelligence" \
    --out ./q1_ai_entrepreneurship \
    --n 15 --email duy.bui@eiu.edu.vn
"""
import argparse
import re
import sys
import time
from pathlib import Path

import httpx

OPENALEX = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/"

# Reputable venues (substring match, case-insensitive) — proxy for Q1/WoS quality
Q1_VENUES = [
    "journal of business venturing", "entrepreneurship theory and practice",
    "small business economics", "strategic entrepreneurship journal",
    "technovation", "research policy", "journal of business research",
    "international journal of entrepreneurial behav", "international small business journal",
    "journal of management studies", "academy of management", "british journal of management",
    "journal of finance", "journal of financial economics", "management science",
    "mis quarterly", "journal of product innovation management", "technological forecasting",
    "journal of business ethics", "long range planning", "california management review",
    "harvard business review", "journal of economic surveys", "journal of economics",
    "entrepreneurship education", "review of managerial science", "electronic markets",
]


def reconstruct_abstract(inv):
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def sanitize(name, maxlen=80):
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:maxlen]


def is_q1_venue(venue):
    if not venue:
        return False
    v = venue.lower()
    return any(q in v for q in Q1_VENUES)


def search_openalex(query, email, per_page=60):
    params = {
        "filter": f"title_and_abstract.search:{query},is_oa:true,type:article",
        "sort": "cited_by_count:desc",
        "per_page": per_page,
        "mailto": email,
    }
    r = httpx.get(OPENALEX, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["results"]


def unpaywall_pdf(doi, email):
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "")
    try:
        r = httpx.get(f"{UNPAYWALL}{doi}", params={"email": email}, timeout=30)
        if r.status_code != 200:
            return None
        loc = r.json().get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def download_pdf(url, dest, email):
    """Download and verify a PDF. Returns True on success."""
    try:
        with httpx.Client(follow_redirects=True, timeout=60,
                          headers={"User-Agent": f"academic-search-machine (mailto:{email})"}) as c:
            r = c.get(url)
            if r.status_code != 200:
                return False
            data = r.content
            # Verify it's actually a PDF, not an HTML landing page
            if not data[:5].startswith(b"%PDF"):
                return False
            if len(data) < 20_000:  # <20KB is almost certainly not a real paper
                return False
            dest.write_bytes(data)
            return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15, help="target number of papers to download")
    ap.add_argument("--email", default="duy.bui@eiu.edu.vn")
    ap.add_argument("--require-q1", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[search] OpenAlex: '{args.query}' (OA articles, by citations)...")
    candidates = search_openalex(args.query, args.email)
    print(f"[search] {len(candidates)} candidates returned")

    downloaded = []
    for w in candidates:
        if len(downloaded) >= args.n:
            break
        title = (w.get("title") or "").strip()
        if not title:
            continue
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        if args.require_q1 and not is_q1_venue(venue):
            continue
        # must plausibly be about the topic
        t_low = title.lower()
        if "entrepreneur" not in t_low and "startup" not in t_low and "business" not in t_low and "firm" not in t_low and "venture" not in t_low:
            # allow if abstract mentions entrepreneurship strongly
            abs = reconstruct_abstract(w.get("abstract_inverted_index")).lower()
            if "entrepreneur" not in abs:
                continue

        doi = w.get("doi")
        year = w.get("publication_year")
        cites = w.get("cited_by_count", 0)
        authors = [a["author"]["display_name"] for a in (w.get("authorships") or [])[:6]]

        # Build an ordered list of candidate PDF urls and try each until one works:
        # every OpenAlex OA location, then Unpaywall's resolved PDF.
        pdf_candidates = []
        for loc in (w.get("locations") or []):
            if loc.get("pdf_url"):
                pdf_candidates.append(loc["pdf_url"])
        best = (w.get("best_oa_location") or {}).get("pdf_url")
        if best:
            pdf_candidates.insert(0, best)
        up = unpaywall_pdf(doi, args.email)
        if up:
            pdf_candidates.append(up)
        # dedupe preserving order
        seen_u = set()
        pdf_candidates = [u for u in pdf_candidates if not (u in seen_u or seen_u.add(u))]
        if not pdf_candidates:
            continue

        idx = len(downloaded) + 1
        fname = f"{idx:02d}_{year}_{sanitize(title, 60)}.pdf"
        dest = out / fname
        print(f"[{idx:02d}] {cites:>4} cites | {venue[:30]:30} | trying {len(pdf_candidates)} url(s)...")
        ok = False
        used_url = None
        for u in pdf_candidates:
            if download_pdf(u, dest, args.email):
                ok = True
                used_url = u
                break
        if not ok:
            print(f"     ✗ PDF unavailable/invalid, skipping")
            continue
        pdf_url = used_url
        kb = dest.stat().st_size // 1024
        print(f"     ✓ {fname} ({kb} KB)")
        downloaded.append({
            "idx": idx, "title": title, "authors": authors, "year": year,
            "venue": venue, "cites": cites, "doi": doi, "file": fname,
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index"))[:600],
            "pdf_url": pdf_url,
        })
        time.sleep(0.5)  # be polite

    # Write index
    readme = out / "README.md"
    lines = [
        f"# Q1 Full-Text Papers — {args.query.title()}",
        "",
        f"Curated set of **{len(downloaded)}** open-access, full-text papers from reputable "
        "(Q1/WoS-tier) journals, sorted by citation count. Fetched via OpenAlex + Unpaywall.",
        "",
        "> Study material. All papers are open access; PDFs are the publishers'/authors' OA versions.",
        "",
        "| # | Year | Citations | Venue | Title | File |",
        "|---|------|-----------|-------|-------|------|",
    ]
    for p in downloaded:
        lines.append(
            f"| {p['idx']} | {p['year']} | {p['cites']} | {p['venue']} | "
            f"{p['title']} | [`{p['file']}`](./{p['file']}) |"
        )
    lines += ["", "---", "", "## Details", ""]
    for p in downloaded:
        lines += [
            f"### {p['idx']}. {p['title']}",
            "",
            f"- **Authors**: {', '.join(p['authors'])}",
            f"- **Year**: {p['year']}  |  **Venue**: {p['venue']}  |  **Citations**: {p['cites']}",
            f"- **DOI**: {p['doi']}",
            f"- **File**: `{p['file']}`",
            "",
            f"**Abstract**: {p['abstract']}...",
            "",
        ]
    readme.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[done] {len(downloaded)} papers → {out}/")
    print(f"[done] index: {readme}")
    return 0 if downloaded else 1


if __name__ == "__main__":
    sys.exit(main())
