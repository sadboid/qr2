"""Turn a mined style profile into writing constraints the engine can follow.

The writer's per-section guidance used to be hand-written priors. This renders
the same guidance from measurements taken off real Q1 papers, and states the
support behind each number so a reader can tell evidence from assertion.

Where a mined value and a prior disagree, the mined value wins — that is the
entire point of harvesting the library.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE = "library/style_profile.json"

# Map the engine's section names onto the miner's canonical names.
_SECTION_ALIAS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "literature_review": "literature",
    "literature": "literature",
    "theoretical_framework": "literature",
    "methods": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "future_directions": "conclusion",
}

# Which genre's profile to write against. A systematic review should imitate
# reviews, not experiments.
_GENRE_FOR_PAPER_TYPE = {"imrad": "review", "bibliometric": "review"}


class StyleGuide:
    """Renders mined, source-counted writing constraints for a section."""

    def __init__(self, profile: Optional[dict] = None):
        self.profile = profile or {}

    @classmethod
    def load(cls, path: str = _DEFAULT_PROFILE) -> "StyleGuide":
        p = Path(path)
        if not p.exists():
            logger.info(f"[StyleGuide] no mined profile at {path} — using priors only")
            return cls({})
        try:
            prof = json.loads(p.read_text())
            n = prof.get("corpus", {}).get("n_papers", 0)
            logger.info(f"[StyleGuide] loaded profile mined from {n} Q1 papers")
            return cls(prof)
        except Exception as e:
            logger.warning(f"[StyleGuide] could not read {path}: {e!r}")
            return cls({})

    @property
    def available(self) -> bool:
        return bool(self.profile.get("sections"))

    def _section_stats(self, section: str, genre: str) -> Optional[dict]:
        sections = self.profile.get("sections", {})
        canon = _SECTION_ALIAS.get(section, section)
        for g in (genre, "any"):
            stats = (sections.get(g) or {}).get(canon)
            if stats and stats.get("n_papers", 0) >= 3:
                return {**stats, "_genre": g}
        return None

    def constraints_for(self, section: str, paper_type: str = "imrad") -> str:
        """Render the mined constraints for one section as prompt text.

        Returns "" when the library cannot support the section — the writer
        then falls back to its priors rather than to invented numbers.
        """
        if not self.available:
            return ""
        genre = _GENRE_FOR_PAPER_TYPE.get(paper_type, "review")
        s = self._section_stats(section, genre)
        if not s:
            return ""

        n = s["n_papers"]
        src = f"{n} real Q1 {s['_genre']} papers" if s["_genre"] != "any" else f"{n} real Q1 papers"
        lines = [
            f"MEASURED HOUSE STYLE (from {src} in the harvested library — "
            f"these numbers are measurements, not preferences; match them):",
            f"- Length: about {s['words_median']} words "
            f"(observed range {s['words_range'][0]}–{s['words_range'][1]}).",
        ]
        if s.get("citations_per_1k"):
            per_section = round(s["citations_per_1k"] * s["words_median"] / 1000)
            lines.append(
                f"- Citation density: {s['citations_per_1k']} citations per 1,000 words "
                f"— roughly {per_section} citations in a section this length."
            )
        if s.get("integral_share") is not None and s.get("citations_per_1k"):
            pct = int(round(s["integral_share"] * 100))
            lines.append(
                f"- Citation form: only ~{pct}% integral (\"Smith (2020) argues…\"); "
                f"the rest parenthetical (\"(Smith, 2020)\"). Do not overuse integral citations."
            )
        if s.get("words_per_sentence"):
            lines.append(
                f"- Sentence length: median {s['words_per_sentence']} words. "
                f"Vary around it; do not write uniformly long sentences."
            )
        if s.get("sentences_per_paragraph"):
            lines.append(
                f"- Paragraph size: about {s['sentences_per_paragraph']} sentences."
            )
        if s.get("hedges_per_1k") or s.get("boosters_per_1k"):
            h, b = s.get("hedges_per_1k", 0.0), s.get("boosters_per_1k", 0.0)
            # State the measured balance; never prescribe a direction the data
            # does not show. Sections such as Results legitimately boost as
            # much as they hedge.
            if h >= b * 1.3:
                shape = "these papers hedge noticeably more than they boost — match that caution"
            elif b >= h * 1.3:
                shape = "these papers assert more than they hedge here — do not over-hedge"
            else:
                shape = "hedging and boosting are roughly balanced here"
            lines.append(
                f"- Calibration: hedges {h}/1k words vs boosters {b}/1k — {shape}."
            )
        conns = [c["text"] for c in (s.get("connectors") or [])[:10]]
        if conns:
            lines.append(
                "- Discourse connectors actually used in this section: "
                + ", ".join(f"'{c}'" for c in conns) + "."
            )
        verbs = [v["text"] for v in (s.get("reporting_verbs") or [])[:10]]
        if verbs:
            lines.append(
                "- Reporting verbs these authors use when attributing a claim to a "
                "source: " + ", ".join(verbs) + ". Vary them; do not attribute "
                "everything with the same verb."
            )
        openers = [o["text"] for o in (s.get("paragraph_openers") or [])[:8]]
        if openers:
            lines.append(
                "- Paragraph openings attested in this section: "
                + "; ".join(f"'{o}…'" for o in openers) + "."
            )
        phrases = [p["text"] for p in (s.get("phrases") or [])[:12]]
        if phrases:
            lines.append(
                "- Field-conventional phrasings (each attested in ≥3 papers): "
                + "; ".join(phrases) + "."
            )
        return "\n".join(lines)

    def move_exemplars(self, move: str, limit: int = 6) -> str:
        """Real sentences performing a rhetorical move, with their DOIs.

        Shown to the writer as evidence of how the move is made in print — not
        as text to copy. Each carries its source so any borrowed wording can be
        traced.
        """
        items = (self.profile.get("moves") or {}).get(move) or []
        if not items:
            return ""
        label = move.replace("_", " ")
        out = [f"HOW PUBLISHED PAPERS MAKE THIS MOVE ({label}) — real sentences from "
               f"the library, for structure only. Never copy wording:"]
        for it in items[:limit]:
            out.append(f"  · \"{it['sentence']}\" [{it['doi']}]")
        return "\n".join(out)
