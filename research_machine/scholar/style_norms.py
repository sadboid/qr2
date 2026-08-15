"""What published Q1 prose actually scores on our own style linters.

The style gate has been failing every generated paper, and part of that is
real (AI tells, filler) while part is an artefact: academic prose legitimately
contains "significantly", "crucial", "demonstrates", and a review that quotes
its sources inherits their wording. Judging a manuscript against zero is
judging it against a standard no published paper meets.

So measure the standard. Run the same linters over real Q1 papers from the
harvested library, record how often each finding category fires per 1,000
words, and let the gate flag only what exceeds what the field actually
publishes. This is the method-norms discipline applied to prose: the norm is
what appears in print, not what I assume good writing looks like.
"""

import json
import logging
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_NORMS_PATH = "library/style_norms.json"
# Prose sections only. Methods and Results are heavy with numbers and
# instrument names, which trip different rules than argumentative prose.
_PROSE_SECTIONS = ("introduction", "literature", "discussion", "conclusion")


def _normalize_category(raw: str) -> str:
    """'AI vocabulary ('utilizing')' -> 'AI vocabulary'."""
    return re.sub(r"\s*\(.*", "", raw).strip().lower()


def _lint_rates(text: str) -> Dict[str, float]:
    """Findings per 1,000 words, by normalized category, for one text."""
    from ..local_engine import deslop, prose_check
    words = max(1, len(text.split()))
    rates: Dict[str, float] = defaultdict(float)
    for module in (deslop, prose_check):
        try:
            findings = module.lint(text=text)[0]
        except Exception as e:
            logger.debug(f"[StyleNorms] lint failed: {e!r}")
            continue
        for f in findings:
            severity, category = f[0], _normalize_category(f[1])
            rates[f"{severity}:{category}"] += 1000 / words
    return dict(rates)


def build_norms(library_root: str = "library", out_path: str = _NORMS_PATH,
                min_papers: int = 5) -> dict:
    """Measure linter behaviour on real Q1 prose and write the norms file."""
    from .pdf_parser import parse_pdf
    root = Path(library_root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [e for e in manifest["entries"] if e.get("pdf_path")]

    per_paper: List[Dict[str, float]] = []
    words_seen = 0
    for e in entries:
        parsed = parse_pdf(str(root / e["pdf_path"]), e["doi"])
        if not parsed.ok:
            continue
        prose = "\n\n".join(parsed.sections.get(s, "") for s in _PROSE_SECTIONS)
        if len(prose.split()) < 800:
            continue
        words_seen += len(prose.split())
        per_paper.append(_lint_rates(prose))

    if len(per_paper) < min_papers:
        logger.warning(
            f"[StyleNorms] only {len(per_paper)} usable papers (need {min_papers}) — "
            f"norms not written; the gate keeps its default thresholds"
        )
        return {}

    # Trim papers that are themselves outliers before setting the norm.
    # Published does not mean clean: the first run of this found a Scopus paper
    # opening "In today's rapidly evolving business landscape" with 4.6x the
    # AI-vocabulary rate of its peers. Learning norms from such papers would
    # drift the standard toward the style the gate exists to catch.
    totals = sorted(sum(r.values()) for r in per_paper)
    # Trim the noisiest decile. A fixed multiple of the median stops trimming
    # as the corpus grows (more papers raise the median, so the outlier slips
    # back under the bar); a percentile keeps the same standard at any size.
    cut = totals[int(0.9 * (len(totals) - 1))]
    kept, trimmed = [], []
    for e_rates in per_paper:
        (trimmed if sum(e_rates.values()) >= cut and len(per_paper) > 10
         else kept).append(e_rates)
    if len(kept) >= min_papers and trimmed:
        logger.info(
            f"[StyleNorms] trimmed {len(trimmed)} style-outlier paper(s) "
            f"before computing norms ({len(kept)} retained)"
        )
        per_paper = kept

    categories = {c for rates in per_paper for c in rates}
    norms = {}
    for cat in categories:
        values = [rates.get(cat, 0.0) for rates in per_paper]
        values.sort()
        p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
        norms[cat] = {
            "median_per_1k": round(statistics.median(values), 2),
            "p90_per_1k": round(p90, 2),
            "papers_triggering": sum(1 for v in values if v > 0),
        }

    out = {
        "corpus": {
            "n_papers": len(per_paper),
            "words_analysed": words_seen,
            "sections": list(_PROSE_SECTIONS),
            "style_outliers_trimmed": len(trimmed),
        },
        "norms": dict(sorted(norms.items(), key=lambda kv: -kv[1]["p90_per_1k"])),
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    logger.info(
        f"[StyleNorms] measured {len(per_paper)} Q1 papers "
        f"({words_seen:,} words) -> {out_path}"
    )
    return out


class StyleNorms:
    """Judges a manuscript's linter output against published practice."""

    def __init__(self, norms: Optional[dict] = None):
        data = norms or {}
        self.corpus = data.get("corpus", {})
        self.norms: Dict[str, dict] = data.get("norms", {})

    @classmethod
    def load(cls, path: str = _NORMS_PATH) -> "StyleNorms":
        p = Path(path)
        if not p.exists():
            return cls({})
        try:
            return cls(json.loads(p.read_text()))
        except Exception as e:
            logger.warning(f"[StyleNorms] could not read {path}: {e!r}")
            return cls({})

    @property
    def available(self) -> bool:
        return bool(self.norms)

    def assess(self, findings: List[tuple], word_count: int) -> dict:
        """Compare a manuscript's findings against the published p90.

        HIGH-severity deslop findings (chatbot artefacts, emoji, placeholders)
        are never normalised — no published paper contains them, and a norm
        built from papers that lack them would have nothing to say.
        """
        words = max(1, word_count)
        rates: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for f in findings:
            key = f"{f[0]}:{_normalize_category(f[1])}"
            rates[key] += 1000 / words
            counts[key] += 1

        excesses, within = [], []
        for key, rate in rates.items():
            norm = self.norms.get(key)
            if norm is None:
                # Unseen in published prose — treat as a genuine finding.
                excesses.append({
                    "category": key, "rate_per_1k": round(rate, 2),
                    "published_p90": None, "ratio": None,
                    "note": "not observed in the Q1 sample",
                })
                continue
            p90 = norm["p90_per_1k"]
            if p90 <= 0 or rate > p90:
                excesses.append({
                    "category": key, "rate_per_1k": round(rate, 2),
                    "published_p90": p90,
                    "ratio": round(rate / p90, 2) if p90 else None,
                    "note": "above what Q1 papers publish",
                })
            else:
                within.append({
                    "category": key, "rate_per_1k": round(rate, 2),
                    "published_p90": p90,
                })
        excesses.sort(key=lambda d: -(d["ratio"] or 99))
        return {
            "excesses": excesses,
            "within_norms": within,
            "n_categories_flagged": len(excesses),
            "n_categories_normal": len(within),
            "benchmark_papers": self.corpus.get("n_papers", 0),
        }
