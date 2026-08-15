"""PDF -> structured paper: sections, paragraphs, sentences, citations.

Built for real journal PDFs, which are messier than they look: running
headers, page numbers, hyphenated line breaks, two-column flows, and headings
that may be numbered, capitalised, or neither. The parser aims for a clean
IMRaD-ish segmentation and drops the reference list, since mining prose
statistics over a bibliography poisons every downstream count.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical section -> heading patterns actually seen in management journals
_SECTION_PATTERNS = [
    ("abstract",     r"^\s*abstract\b"),
    ("introduction", r"^\s*(?:\d+\.?\s*)?introduction\b"),
    ("literature",   r"^\s*(?:\d+\.?\s*)?(?:literature\s+review|theoretical\s+(?:background|framework)"
                     r"|conceptual\s+(?:background|framework)|theory\s+and\s+hypothes|background)\b"),
    ("methods",      r"^\s*(?:\d+\.?\s*)?(?:method(?:s|ology)?|research\s+(?:method|design)"
                     r"|data\s+and\s+method|empirical\s+(?:strategy|approach)|materials\s+and\s+methods)\b"),
    ("results",      r"^\s*(?:\d+\.?\s*)?(?:results?|findings?|analysis\s+and\s+results)\b"),
    ("discussion",   r"^\s*(?:\d+\.?\s*)?(?:discussion|general\s+discussion)\b"),
    ("conclusion",   r"^\s*(?:\d+\.?\s*)?(?:conclusions?|concluding\s+remarks)\b"),
    ("references",   r"^\s*(?:\d+\.?\s*)?(?:references|bibliography|works\s+cited)\s*$"),
]
_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _SECTION_PATTERNS]

# (Author, 2020) / (Author & Author, 2020) / (Author et al., 2020).
# The author part is required: a bare "(2020)" is the tail of an INTEGRAL
# citation ("Chalmers et al. (2020)"), and counting it here both inflates the
# parenthetical count and hides the integral one.
_CITE_PAREN = re.compile(
    r"\(([^()]{0,120}?[A-Za-zÀ-ɏ]{2,}[^()]{0,40}?\b(?:19|20)\d{2}[a-z]?)\)")
# Author (2020) / Author et al. (2020) / Author and Author (2020).
# "et al." ends the name — nothing follows it before the year.
_CITE_INTEGRAL = re.compile(
    r"\b([A-Z][A-Za-zÀ-ɏ'’-]+"
    r"(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][A-Za-zÀ-ɏ'’-]+))?)"
    r"\s*\((?:19|20)\d{2}[a-z]?\)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'“])")


@dataclass
class ParsedPaper:
    doi: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    n_pages: int = 0
    ok: bool = False
    genre: str = "unknown"        # empirical | review | conceptual | editorial
    reason: str = ""

    def section_sentences(self, name: str) -> List[str]:
        return split_sentences(self.sections.get(name, ""))

    def paragraphs(self, name: str) -> List[str]:
        body = self.sections.get(name, "")
        return [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 80]


def split_sentences(text: str) -> List[str]:
    """Sentence split that resists the abbreviations common in citations."""
    if not text:
        return []
    guard = text
    for abbr in ("et al.", "e.g.", "i.e.", "cf.", "vs.", "Fig.", "Eq.", "No.", "pp.", "Dr.", "Prof."):
        guard = guard.replace(abbr, abbr.replace(".", "․"))
    out = []
    for s in _SENT_SPLIT.split(guard):
        s = s.replace("․", ".").strip()
        if len(s) > 25:
            out.append(s)
    return out


def _extract_raw(pdf_path: Path, max_pages: int = 40) -> Optional[tuple]:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for pg in reader.pages[:max_pages]:
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages), len(reader.pages)
    except Exception as e:
        logger.debug(f"[PDFParser] {pdf_path.name}: {e!r}")
        return None


def _strip_running_heads(text: str) -> str:
    """Remove lines that repeat on most pages (journal name, author strip) and
    bare page numbers — they otherwise show up as high-frequency 'phrases'."""
    lines = text.split("\n")
    counts = Counter(l.strip() for l in lines if 8 < len(l.strip()) < 90)
    noisy = {l for l, c in counts.items() if c >= 4}
    out = []
    for l in lines:
        s = l.strip()
        if not s or s in noisy:
            continue
        if re.fullmatch(r"[\divxlcIVXLC \-–—.]{1,12}", s):     # page numbers
            continue
        out.append(l)
    return "\n".join(out)


_REF_HEAD = re.compile(r"^\s*(?:references|bibliography|works\s+cited)\b\s*$", re.IGNORECASE)
_REF_LINE = re.compile(
    r"^[A-Z][A-Za-zÀ-ɏ'’-]+,\s*[A-Z]\.|"          # Surname, I.
    r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)|"             # (2020)
    r"\bdoi\.org/|\bhttps?://|"
    r",\s*\d+\s*\(\d+\),\s*\d+")                   # , 36(4), 1165
_MIN_WORDS_PER_SECTION = 200


def _cut_references(text: str) -> str:
    """Truncate the bibliography.

    A bare 'References' heading is the clean signal, but many extractions lose
    the line break, so fall back to the last stretch where most lines look like
    reference entries. Mining prose statistics over a bibliography inflates
    every citation and phrase count, so it is worth being aggressive here.
    """
    lines = text.split("\n")
    for i in range(len(lines) - 1, max(0, len(lines) - 4000), -1):
        if _REF_HEAD.match(lines[i]):
            return "\n".join(lines[:i])

    # No clean heading: scan from the back for where reference-like density
    # takes over and stays high.
    window, best = 25, None
    for i in range(len(lines) - window, 0, -1):
        chunk = [l for l in lines[i:i + window] if len(l.strip()) > 20]
        if not chunk:
            continue
        density = sum(1 for l in chunk if _REF_LINE.search(l.strip())) / len(chunk)
        if density >= 0.55:
            best = i
        elif best is not None and i < best - window:
            break
    return "\n".join(lines[:best]) if best else text


def _classify_genre(sections: Dict[str, str]) -> str:
    have = set(sections)
    blob = " ".join(sections.values()).lower()[:20000]
    if "methods" in have and "results" in have:
        if re.search(r"\b(systematic (literature )?review|prisma|we (?:reviewed|screened)"
                     r"|inclusion criteria|bibliometric)\b", blob):
            return "review"
        return "empirical"
    if re.search(r"\b(systematic (literature )?review|prisma|bibliometric analysis)\b", blob):
        return "review"
    total = sum(len(v.split()) for v in sections.values())
    if total < 4000 and "methods" not in have:
        return "editorial"
    return "conceptual"


def _dehyphenate(text: str) -> str:
    """Rejoin words split across lines and unwrap soft line breaks, while
    PRESERVING paragraph boundaries.

    PDF extraction gives one line per typeset line, so naively unwrapping every
    newline fuses a whole section into a single paragraph — and every
    paragraph-level statistic downstream becomes meaningless. A paragraph ends
    where a short line (the ragged last line) closes a sentence.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # word split across lines
    lines = [l.rstrip() for l in text.split("\n")]
    lengths = [len(l) for l in lines if len(l) > 20]
    if not lengths:
        return re.sub(r"[ \t]{2,}", " ", text)
    typical = sorted(lengths)[len(lengths) // 2]
    out: List[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        stripped = line.strip()
        if not stripped:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        ends_sentence = bool(re.search(r"[.!?][\"')\]]?$", stripped))
        short_line = len(stripped) < 0.82 * typical
        starts_new = bool(re.match(r"[A-Z“\"(]", nxt))
        if ends_sentence and short_line and starts_new:
            out.append("")                            # paragraph break marker
    joined = "\n".join(out)
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", joined)  # unwrap within paragraphs
    return re.sub(r"[ \t]{2,}", " ", joined)


def parse_pdf(pdf_path: str, doi: str = "") -> ParsedPaper:
    """Parse a journal PDF into canonical sections."""
    path = Path(pdf_path)
    raw = _extract_raw(path)
    if raw is None:
        return ParsedPaper(doi=doi, reason="unreadable")
    text, n_pages = raw
    if len(text) < 4000:
        return ParsedPaper(doi=doi, n_pages=n_pages, reason="too little extractable text (scanned?)")

    text = _cut_references(_strip_running_heads(text))

    # Walk lines; a line that matches a heading pattern and is short enough to
    # BE a heading opens a new section.
    sections: Dict[str, List[str]] = {}
    current = "front"
    for line in text.split("\n"):
        stripped = line.strip()
        if 3 < len(stripped) < 80:
            for name, pat in _COMPILED:
                if pat.match(stripped):
                    current = name
                    sections.setdefault(name, [])
                    break
            else:
                sections.setdefault(current, []).append(line)
                continue
            continue
        sections.setdefault(current, []).append(line)

    merged = {k: _dehyphenate("\n".join(v)).strip() for k, v in sections.items()}
    merged.pop("references", None)          # never mine the bibliography
    merged.pop("front", None)
    merged = {k: v for k, v in merged.items()
              if len(v.split()) > (30 if k == "abstract" else _MIN_WORDS_PER_SECTION)}

    p = ParsedPaper(doi=doi, sections=merged, n_pages=n_pages)
    p.genre = _classify_genre(merged)
    # Mineable = an introduction plus at least one body section. Papers that
    # legitimately lack Methods/Results (editorials, conceptual pieces) are not
    # failures; they are a different genre and are mined as such.
    body = {"results", "discussion", "methods", "literature", "conclusion"}
    p.ok = "introduction" in merged and bool(body & set(merged))
    if not p.ok:
        p.reason = f"sections found: {sorted(merged)}"
    return p


def citations_in(text: str) -> Dict[str, List[str]]:
    """Split citations into integral (Author (2020) claims...) and
    non-integral ((Author, 2020)) — the ratio is a real style signal."""
    return {
        "integral": [m.group(1) for m in _CITE_INTEGRAL.finditer(text)],
        "parenthetical": [m.group(1) for m in _CITE_PAREN.finditer(text)],
    }


def citation_density(text: str) -> float:
    """Citations per 1,000 words."""
    words = len(text.split())
    if not words:
        return 0.0
    c = citations_in(text)
    return round(1000 * (len(c["integral"]) + len(c["parenthetical"])) / words, 1)
