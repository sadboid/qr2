"""Venue-specific paper formatting for IEEE, ACM, and Elsevier submission."""

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VenueSpec:
    name: str
    style: str          # "ieee", "acm", "elsevier", "apa"
    max_words: int
    min_citations: int
    abstract_words: int
    keywords_required: bool
    double_blind: bool
    latex_class: str
    submission_notes: str


VENUES = {
    "ieee": VenueSpec(
        name="IEEE Transactions",
        style="ieee",
        max_words=8000,
        min_citations=25,
        abstract_words=250,
        keywords_required=True,
        double_blind=False,
        latex_class="IEEEtran",
        submission_notes="Use \\documentclass[journal]{IEEEtran}. Max 8 pages double-column. "
                         "Index terms required (4-6 keywords from IEEE taxonomy). "
                         "Submission via ScholarOne Manuscripts.",
    ),
    "acm": VenueSpec(
        name="ACM Computing Surveys",
        style="acm",
        max_words=15000,
        min_citations=40,
        abstract_words=300,
        keywords_required=True,
        double_blind=True,
        latex_class="acmart",
        submission_notes="Use \\documentclass[manuscript,screen,review]{acmart}. "
                         "CCS concepts and keywords required. "
                         "Double-blind review — remove author info. "
                         "Submit via ACM submission system.",
    ),
    "elsevier": VenueSpec(
        name="Elsevier Journal (e.g., Int. J. Information Management)",
        style="elsevier",
        max_words=10000,
        min_citations=30,
        abstract_words=250,
        keywords_required=True,
        double_blind=True,
        latex_class="elsarticle",
        submission_notes="Use \\documentclass[review,12pt]{elsarticle}. "
                         "5-8 keywords required. "
                         "Double-blind: use \\blankline for author info. "
                         "Submit via Elsevier Editorial System (EES).",
    ),
    "apa": VenueSpec(
        name="APA Journal (e.g., Journal of Applied Psychology)",
        style="apa",
        max_words=8000,
        min_citations=30,
        abstract_words=250,
        keywords_required=True,
        double_blind=True,
        latex_class="apa7",
        submission_notes="APA 7th edition formatting. "
                         "Running head required on every page. "
                         "Double-blind review. Submit via Manuscript Central.",
    ),
}


def _strip_md_formatting(text: str) -> str:
    """Remove Markdown-specific formatting for venue output."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)         # italic
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"\[(\d+)\]", r"[\1]", text)      # keep citation refs
    return text.strip()


def _count_words(text: str) -> int:
    return len(text.split())


def _truncate_abstract(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def format_ieee_latex(
    title: str,
    abstract: str,
    content_md: str,
    keywords: list,
    references_block: str,
    authors: Optional[list] = None,
) -> str:
    """Generate IEEE Transactions-style LaTeX."""
    kw_str = "; ".join(keywords[:6])
    author_block = "\\author{Anonymous}" if not authors else "\n".join(
        f"\\author{{{a}}}" for a in authors
    )
    abstract_clean = _strip_md_formatting(_truncate_abstract(abstract, 250))
    body = _strip_md_formatting(content_md)

    return f"""\\documentclass[journal]{{IEEEtran}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{hyperref}}

\\begin{{document}}

\\title{{{title}}}

{author_block}

\\maketitle

\\begin{{abstract}}
{abstract_clean}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{kw_str}
\\end{{IEEEkeywords}}

{body}

\\begin{{thebibliography}}{{99}}
{references_block}
\\end{{thebibliography}}

\\end{{document}}
"""


def format_acm_latex(
    title: str,
    abstract: str,
    content_md: str,
    keywords: list,
    references_block: str,
    authors: Optional[list] = None,
) -> str:
    """Generate ACM-style LaTeX (double-blind)."""
    kw_str = ", ".join(keywords[:8])
    abstract_clean = _strip_md_formatting(_truncate_abstract(abstract, 300))
    body = _strip_md_formatting(content_md)

    return f"""\\documentclass[manuscript,screen,review,anonymous]{{acmart}}
\\usepackage[utf8]{{inputenc}}

\\begin{{document}}

\\title{{{title}}}

\\begin{{abstract}}
{abstract_clean}
\\end{{abstract}}

\\keywords{{{kw_str}}}

\\maketitle

{body}

\\bibliographystyle{{ACM-Reference-Format}}
\\begin{{thebibliography}}{{99}}
{references_block}
\\end{{thebibliography}}

\\end{{document}}
"""


def format_elsevier_latex(
    title: str,
    abstract: str,
    content_md: str,
    keywords: list,
    references_block: str,
    authors: Optional[list] = None,
) -> str:
    """Generate Elsevier-style LaTeX."""
    kw_str = "\\sep ".join(keywords[:8])
    abstract_clean = _strip_md_formatting(_truncate_abstract(abstract, 250))
    body = _strip_md_formatting(content_md)

    return f"""\\documentclass[review,12pt]{{elsarticle}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{hyperref}}
\\journal{{International Journal of Information Management}}

\\begin{{document}}

\\begin{{frontmatter}}
\\title{{{title}}}

\\begin{{abstract}}
{abstract_clean}
\\end{{abstract}}

\\begin{{keyword}}
{kw_str}
\\end{{keyword}}
\\end{{frontmatter}}

{body}

\\bibliographystyle{{elsarticle-num}}
\\begin{{thebibliography}}{{99}}
{references_block}
\\end{{thebibliography}}

\\end{{document}}
"""


def build_references_block(papers: list) -> str:
    """Build a LaTeX bibliography block from Paper objects."""
    lines = []
    for i, p in enumerate(papers, 1):
        authors_str = " and ".join(p.authors[:3]) if p.authors else "Unknown"
        if len(p.authors) > 3:
            authors_str += " et al."
        venue = p.venue or "Preprint"
        lines.append(
            f"\\bibitem{{ref{i}}} {authors_str}. {p.title}. "
            f"\\textit{{{venue}}}, {p.year}."
        )
    return "\n".join(lines)


def generate_venue_checklist(
    word_count: int,
    citation_count: int,
    abstract_words: int,
    quality_results: dict,
    venue_key: str = "elsevier",
) -> dict:
    """Return a submission-readiness checklist for a specific venue."""
    spec = VENUES.get(venue_key, VENUES["elsevier"])

    checks = {
        "word_count_ok": {
            "passed": word_count <= spec.max_words,
            "value": word_count,
            "requirement": f"≤ {spec.max_words}",
            "label": "Word count",
        },
        "citation_count_ok": {
            "passed": citation_count >= spec.min_citations,
            "value": citation_count,
            "requirement": f"≥ {spec.min_citations}",
            "label": "Citation count",
        },
        "abstract_length_ok": {
            "passed": abstract_words <= spec.abstract_words,
            "value": abstract_words,
            "requirement": f"≤ {spec.abstract_words} words",
            "label": "Abstract length",
        },
        "novelty_gate": {
            "passed": quality_results.get("novelty", {}).get("passed", False),
            "value": quality_results.get("novelty", {}).get("score", 0),
            "requirement": "< 0.70",
            "label": "Novelty gate",
        },
        "peer_review_gate": {
            "passed": quality_results.get("peer_review", {}).get("passed", False),
            "value": quality_results.get("peer_review", {}).get("score", 0),
            "requirement": "≥ 7.0/10",
            "label": "Peer review gate",
        },
        "fact_check_gate": {
            "passed": quality_results.get("fact_check", {}).get("passed", True),
            "value": quality_results.get("fact_check", {}).get("score", 10.0),
            "requirement": "≥ 6.0/10",
            "label": "Fact-check gate",
        },
    }

    all_passed = all(c["passed"] for c in checks.values())
    passed_count = sum(1 for c in checks.values() if c["passed"])

    return {
        "venue": spec.name,
        "venue_key": venue_key,
        "ready_for_submission": all_passed,
        "passed_count": passed_count,
        "total_checks": len(checks),
        "checks": checks,
        "submission_notes": spec.submission_notes,
        "double_blind": spec.double_blind,
    }
