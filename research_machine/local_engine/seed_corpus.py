"""Seed-corpus mode: build the corpus from a hand-curated set of exemplar
papers + citation chaining, instead of keyword search.

Rationale (from the v9 audit): a keyword-retrieved corpus for a founder-
decision-making review was dominated by adjacent-domain AI papers (education,
HR, clinical) — the single biggest quality ceiling. A corpus grown from
human-vetted Q1 seeds via their citation neighborhood is, by construction,
about the actual topic: papers a seed cites, and papers citing a seed, live
in the same scholarly conversation.

Seeds come from a study folder produced by scripts/fetch_papers.py
(README.md with structured entries + the OA PDFs alongside). PDFs on disk are
mined for full text, so seeds also carry the richest synthesis signal.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .corpus import Paper, _to_paper, _deduplicate_papers
from .citation_network import _get_references, _get_citations
from .synthesizer import _is_off_domain

logger = logging.getLogger(__name__)

_ENTRY_RE = re.compile(
    r"### \d+\.\s*(?P<title>.+?)\n\n"
    r"- \*\*Authors\*\*:\s*(?P<authors>.+?)\n"
    r"- \*\*Year\*\*:\s*(?P<year>\d{4})\s*\|\s*\*\*Venue\*\*:\s*(?P<venue>.+?)\s*\|\s*\*\*Citations\*\*:\s*(?P<cites>\d+)\n"
    r"- \*\*DOI\*\*:\s*(?P<doi>\S+)\n"
    r"- \*\*File\*\*:\s*`(?P<file>[^`]+)`\n\n"
    r"\*\*Abstract\*\*:\s*(?P<abstract>.+?)(?:\.\.\.)?\n",
    re.DOTALL,
)


def _pdf_text(path: Path, max_chars: int = 12000) -> Optional[str]:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = []
        for pg in reader.pages[:15]:
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pass
        text = "\n".join(pages)
        return text[:max_chars] if len(text) > 200 else None
    except Exception as e:
        logger.debug(f"[SeedCorpus] PDF extraction failed for {path.name}: {e!r}")
        return None


def load_seeds(seed_dir: str, keywords: List[str]) -> List[Paper]:
    """Parse a fetch_papers.py study folder into seed Paper objects with
    full text mined from the PDFs on disk."""
    root = Path(seed_dir)
    readme = root / "README.md"
    if not readme.exists():
        raise FileNotFoundError(f"No README.md in seed dir {seed_dir}")

    seeds: List[Paper] = []
    for m in _ENTRY_RE.finditer(readme.read_text()):
        doi = m["doi"].replace("https://doi.org/", "").strip()
        pdf = root / m["file"]
        full_text = _pdf_text(pdf) if pdf.exists() else None
        p = Paper(
            paper_id=f"DOI:{doi}",
            title=m["title"].strip(),
            abstract=m["abstract"].strip(),
            authors=[a.strip() for a in m["authors"].split(",")],
            year=int(m["year"]),
            citation_count=int(m["cites"]),
            venue=m["venue"].strip(),
            url=f"https://doi.org/{doi}",
            source="seed",
            keywords_matched=[k for k in keywords if k.lower() in
                              (m["title"] + " " + m["abstract"]).lower()],
            doi=doi,
            full_text=full_text,
        )
        seeds.append(p)
    logger.info(
        f"[SeedCorpus] Loaded {len(seeds)} seeds "
        f"({sum(1 for s in seeds if s.full_text)} with full text from PDFs)"
    )
    return seeds


async def expand_seeds(
    seeds: List[Paper],
    keywords: List[str],
    domain: str,
    target_size: int = 50,
) -> Tuple[List[Paper], int]:
    """Grow the corpus from the seeds' citation neighborhood.

    For every seed, fetch its references (backward) and citations (forward)
    from Semantic Scholar (DOI-addressed). Candidates are ranked by
    CONNECTIVITY — how many distinct seeds they are linked to — then by
    citation count; connectivity is the Research-Rabbit signal that a paper
    belongs to the same conversation. The domain filter still guards the
    expansion (seeds themselves are exempt: they are human-vetted).

    Returns (corpus, n_raw_candidates).
    """
    connectivity: Dict[str, int] = {}
    candidates: Dict[str, dict] = {}

    for seed in seeds:
        for fetch, direction in ((_get_references, "ref"), (_get_citations, "cite")):
            try:
                results = await fetch(seed.paper_id, limit=20)
            except Exception as e:
                logger.debug(f"[SeedCorpus] {direction} fetch failed for {seed.title[:40]}: {e!r}")
                continue
            for raw in results:
                pid = raw.get("paperId")
                if not pid:
                    continue
                connectivity[pid] = connectivity.get(pid, 0) + 1
                candidates.setdefault(pid, raw)

    n_raw = len(candidates)
    logger.info(f"[SeedCorpus] Citation neighborhood: {n_raw} unique candidates from {len(seeds)} seeds")

    seed_keys = {re.sub(r"\W+", "", s.title.lower())[:60] for s in seeds}
    scored: List[Tuple[int, int, Paper]] = []
    for pid, raw in candidates.items():
        p = _to_paper(raw, keywords)
        if p is None:
            continue
        if re.sub(r"\W+", "", p.title.lower())[:60] in seed_keys:
            continue  # already a seed
        if _is_off_domain(p, domain, keywords):
            continue
        scored.append((connectivity[pid], p.citation_count, p))

    scored.sort(key=lambda t: (-t[0], -t[1]))
    room = max(0, target_size - len(seeds))
    expansion = [p for _, _, p in scored[:room]]
    kept = _deduplicate_papers(list(seeds) + expansion)

    logger.info(
        f"[SeedCorpus] Corpus: {len(seeds)} seeds + {len(kept) - len(seeds)} "
        f"chained papers = {len(kept)} (top connectivity "
        f"{scored[0][0] if scored else 0} seed-links)"
    )
    return kept, n_raw + len(seeds)
