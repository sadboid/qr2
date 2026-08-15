"""Consensus analysis engine — like Consensus.app but fully extractive (no LLM API).

For a given research question and paper corpus:
- Classifies each paper's stance: SUPPORT / OPPOSE / NEUTRAL / MIXED
- Calculates consensus direction (YES / NO / MIXED) and confidence %
- Extracts key evidence sentences from each stance group
- Generates a consensus section for inclusion in the generated paper

Scoring formula:
  support_hits — oppose_hits → net direction
  mixed_hits   → nuance modifier
  keyword_hits → relevance weight
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple
from .corpus import Paper

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Signal dictionaries
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORT_SIGNALS = [
    "significantly improves", "significantly increases", "significantly reduces costs",
    "positive effect", "positive impact", "positive relationship", "positive association",
    "enhances", "boosts", "accelerates", "facilitates", "enables", "promotes",
    "higher performance", "better outcome", "greater efficiency", "better results",
    "effective", "beneficial", "advantageous", "promising", "strong evidence",
    "confirms", "validates", "consistent with", "supports the hypothesis",
    "we find that", "results show that", "results indicate that",
    "evidence suggests", "demonstrates that", "proves that",
    "increases productivity", "improves", "leads to higher", "leads to better",
    "associated with improved", "associated with higher", "significant positive",
    "statistically significant", "outperforms",
]

_OPPOSE_SIGNALS = [
    "no significant", "not significant", "no effect", "no relationship",
    "no association", "no evidence", "null result", "null results",
    "negative effect", "negative impact", "negative association",
    "decreases performance", "reduces quality", "hinders", "impedes", "inhibits",
    "challenges", "risks", "barriers", "concerns",
    "fails to", "does not improve", "does not support",
    "contradicts", "refute", "reject the hypothesis", "inconsistent",
    "no significant difference", "no evidence of", "not associated",
    "lack of evidence", "limited evidence", "weak evidence",
]

_MIXED_SIGNALS = [
    "however", "although", "while", "but", "on the other hand",
    "mixed results", "mixed findings", "context-dependent", "boundary condition",
    "moderating", "contingent", "depends on", "varies by",
    "under certain conditions", "in some cases",
    "partially support", "partially confirms", "nuanced",
    "inconsistent findings",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperStance:
    paper: Paper
    stance: str        # "SUPPORT" | "OPPOSE" | "NEUTRAL" | "MIXED"
    confidence: float  # 0.0–1.0
    key_sentence: str  # Most representative sentence from abstract


@dataclass
class ConsensusResult:
    research_question: str
    total_papers: int
    support_count: int
    oppose_count: int
    neutral_count: int
    mixed_count: int
    stances: List[PaperStance] = field(default_factory=list)
    support_evidence: List[str] = field(default_factory=list)
    oppose_evidence: List[str] = field(default_factory=list)
    consensus_paragraph: str = ""
    reading_list: List[dict] = field(default_factory=list)

    @property
    def consensus_pct(self) -> float:
        if self.total_papers == 0:
            return 0.0
        dominant = max(self.support_count, self.oppose_count)
        return round(dominant / self.total_papers * 100, 1)

    @property
    def consensus_direction(self) -> str:
        if self.support_count == 0 and self.oppose_count == 0:
            return "INSUFFICIENT"
        if self.support_count > self.oppose_count * 2:
            return "YES"
        if self.oppose_count > self.support_count * 2:
            return "NO"
        return "MIXED"

    @property
    def consensus_label(self) -> str:
        direction = self.consensus_direction
        pct = self.consensus_pct
        if direction == "YES":
            return "Strong Evidence For" if pct >= 70 else "Moderate Evidence For"
        if direction == "NO":
            return "Strong Evidence Against" if pct >= 70 else "Moderate Evidence Against"
        if direction == "INSUFFICIENT":
            return "Insufficient Evidence"
        return "Mixed / Contested Evidence"

    def to_dict(self) -> dict:
        return {
            "direction": self.consensus_direction,
            "label": self.consensus_label,
            "pct": self.consensus_pct,
            "total_papers": self.total_papers,
            "support_count": self.support_count,
            "oppose_count": self.oppose_count,
            "neutral_count": self.neutral_count,
            "mixed_count": self.mixed_count,
            "support_evidence": self.support_evidence,
            "oppose_evidence": self.oppose_evidence,
            "consensus_paragraph": self.consensus_paragraph,
            "reading_list": self.reading_list,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if 30 < len(s.strip()) < 350]


def _classify_stance(
    abstract: str,
    keywords: List[str],
) -> Tuple[str, float, str]:
    """Classify paper stance based on signal-phrase counting."""
    text = abstract.lower()

    support_hits = sum(1 for sig in _SUPPORT_SIGNALS if sig in text)
    oppose_hits = sum(1 for sig in _OPPOSE_SIGNALS if sig in text)
    mixed_hits = sum(1 for sig in _MIXED_SIGNALS if sig in text)

    # Find the most representative sentence (highest net signal weight)
    sentences = _extract_sentences(abstract)
    best_sent = ""
    best_abs = -1
    for sent in sentences:
        s = sent.lower()
        s_sup = sum(1 for sig in _SUPPORT_SIGNALS if sig in s)
        s_opp = sum(1 for sig in _OPPOSE_SIGNALS if sig in s)
        kw_bonus = sum(0.3 for kw in keywords if kw.lower() in s)
        score = abs(s_sup - s_opp) + kw_bonus
        if score > best_abs:
            best_abs = score
            best_sent = sent

    if not best_sent and sentences:
        best_sent = sentences[0]

    total_directional = support_hits + oppose_hits

    if total_directional == 0:
        return "NEUTRAL", 0.3, best_sent

    # Mixed: explicit mixed signals AND both directions present
    if mixed_hits >= 2 and support_hits > 0 and oppose_hits > 0:
        conf = min(0.5 + mixed_hits * 0.05, 0.85)
        return "MIXED", conf, best_sent

    if support_hits >= oppose_hits * 1.5 and support_hits > 0:
        conf = min(0.50 + (support_hits - oppose_hits) * 0.08, 0.95)
        return "SUPPORT", conf, best_sent

    if oppose_hits >= support_hits * 1.5 and oppose_hits > 0:
        conf = min(0.50 + (oppose_hits - support_hits) * 0.08, 0.95)
        return "OPPOSE", conf, best_sent

    # Roughly equal support + oppose → mixed
    return "MIXED", 0.45, best_sent


# ─────────────────────────────────────────────────────────────────────────────
# Reading list builder (Research Rabbit style)
# ─────────────────────────────────────────────────────────────────────────────

def _build_reading_list(
    support_ps: List[PaperStance],
    oppose_ps: List[PaperStance],
    mixed_ps: List[PaperStance],
) -> List[dict]:
    """
    Prioritised reading path (Research Rabbit style):
      1. Foundational — high-citation SUPPORT papers (read first)
      2. Contrarian  — high-citation OPPOSE papers (critical lens)
      3. Nuanced     — MIXED / boundary-condition papers (deepens understanding)
    """
    entries = []

    # Foundational: top-3 most-cited SUPPORT papers
    for ps in sorted(support_ps, key=lambda x: -x.paper.citation_count)[:3]:
        entries.append({
            "type": "FOUNDATIONAL",
            "emoji": "📌",
            "title": ps.paper.title,
            "year": ps.paper.year,
            "ref": ps.paper.short_ref(),
            "citations": ps.paper.citation_count,
            "venue": ps.paper.venue or "arXiv",
            "key_finding": (ps.key_sentence[:150] + "…") if len(ps.key_sentence) > 150 else ps.key_sentence,
            "url": ps.paper.url or "",
        })

    # Contrarian: top-2 most-cited OPPOSE papers
    for ps in sorted(oppose_ps, key=lambda x: -x.paper.citation_count)[:2]:
        entries.append({
            "type": "CONTRARIAN",
            "emoji": "⚠️",
            "title": ps.paper.title,
            "year": ps.paper.year,
            "ref": ps.paper.short_ref(),
            "citations": ps.paper.citation_count,
            "venue": ps.paper.venue or "arXiv",
            "key_finding": (ps.key_sentence[:150] + "…") if len(ps.key_sentence) > 150 else ps.key_sentence,
            "url": ps.paper.url or "",
        })

    # Nuanced: top-2 most-cited MIXED papers
    for ps in sorted(mixed_ps, key=lambda x: -x.paper.citation_count)[:2]:
        entries.append({
            "type": "NUANCED",
            "emoji": "🔍",
            "title": ps.paper.title,
            "year": ps.paper.year,
            "ref": ps.paper.short_ref(),
            "citations": ps.paper.citation_count,
            "venue": ps.paper.venue or "arXiv",
            "key_finding": (ps.key_sentence[:150] + "…") if len(ps.key_sentence) > 150 else ps.key_sentence,
            "url": ps.paper.url or "",
        })

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Consensus paragraph builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_consensus_paragraph(
    result: "ConsensusResult",
    research_question: str,
) -> str:
    direction = result.consensus_direction
    pct = result.consensus_pct
    n = result.total_papers

    if direction == "YES":
        opening = (
            f"Across the {n}-paper corpus, the weight of evidence **supports** the proposition "
            f"that {research_question.rstrip('?').lower()} "
            f"({pct:.0f}% of papers, n={result.support_count}/{n}). "
            f"This constitutes {result.consensus_label.lower()} within the literature."
        )
        evid = "\n".join(f"  - {e}" for e in result.support_evidence[:3])
        middle = f"\n\nKey supporting evidence:\n{evid}" if evid else ""
        close = (
            f"\n\nA minority of studies (n={result.oppose_count}) report null or negative effects, "
            f"suggesting boundary conditions related to context, implementation quality, and "
            f"organisational readiness."
            if result.oppose_count
            else ""
        )

    elif direction == "NO":
        opening = (
            f"Across the {n}-paper corpus, evidence is **mixed-to-negative** regarding "
            f"the proposition that {research_question.rstrip('?').lower()} "
            f"({pct:.0f}% of papers find null or negative effects, n={result.oppose_count}/{n})."
        )
        evid = "\n".join(f"  - {e}" for e in result.oppose_evidence[:3])
        middle = f"\n\nKey contrarian evidence:\n{evid}" if evid else ""
        close = (
            f"\n\nHowever, {result.support_count} studies identify positive effects under "
            f"specific conditions, indicating outcome heterogeneity warranting further investigation."
            if result.support_count
            else ""
        )

    elif direction == "INSUFFICIENT":
        opening = (
            f"The {n}-paper corpus provides insufficient evidence to draw a clear directional "
            f"consensus on {research_question.rstrip('?').lower()}. "
            f"Most retrieved papers address related but tangential sub-questions."
        )
        middle = ""
        close = "\n\nThis constitutes a clear empirical gap warranting dedicated primary research."

    else:  # MIXED
        opening = (
            f"The research corpus exhibits **mixed findings** regarding "
            f"{research_question.rstrip('?').lower()} "
            f"({result.support_count} studies support, {result.oppose_count} oppose, "
            f"{result.mixed_count} report context-dependent effects out of {n} papers)."
        )
        all_evid = result.support_evidence[:2] + result.oppose_evidence[:2]
        evid = "\n".join(f"  - {e}" for e in all_evid)
        middle = f"\n\nRepresentative evidence:\n{evid}" if evid else ""
        close = (
            "\n\nThis heterogeneity suggests important moderating variables — "
            "such as firm size, technology maturity, industry context, and implementation quality — "
            "that future research should explicitly model and test."
        )

    return (opening + middle + close).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_consensus(
    corpus: List[Paper],
    research_question: str,
    keywords: List[str],
) -> ConsensusResult:
    """
    Analyse paper corpus and compute consensus on the research question.

    Args:
        corpus: List of Paper objects (with .abstract populated)
        research_question: The focal research question
        keywords: Topic keywords for relevance weighting

    Returns:
        ConsensusResult with stance counts, evidence quotes, reading list,
        and a ready-to-embed consensus paragraph
    """
    stances: List[PaperStance] = []

    for paper in corpus:
        if not paper.abstract or len(paper.abstract) < 50:
            continue
        stance, confidence, key_sent = _classify_stance(paper.abstract, keywords)
        stances.append(PaperStance(
            paper=paper,
            stance=stance,
            confidence=confidence,
            key_sentence=key_sent,
        ))

    support_ps = sorted([s for s in stances if s.stance == "SUPPORT"], key=lambda x: -x.confidence)
    oppose_ps = sorted([s for s in stances if s.stance == "OPPOSE"], key=lambda x: -x.confidence)
    neutral_ps = [s for s in stances if s.stance == "NEUTRAL"]
    mixed_ps = [s for s in stances if s.stance == "MIXED"]

    support_evidence = [
        f'"{s.key_sentence[:180].strip()}" [{s.paper.short_ref()}]'
        for s in support_ps[:5]
        if s.key_sentence
    ]
    oppose_evidence = [
        f'"{s.key_sentence[:180].strip()}" [{s.paper.short_ref()}]'
        for s in oppose_ps[:5]
        if s.key_sentence
    ]

    result = ConsensusResult(
        research_question=research_question,
        total_papers=len(stances),
        support_count=len(support_ps),
        oppose_count=len(oppose_ps),
        neutral_count=len(neutral_ps),
        mixed_count=len(mixed_ps),
        stances=stances,
        support_evidence=support_evidence,
        oppose_evidence=oppose_evidence,
    )

    result.consensus_paragraph = _build_consensus_paragraph(result, research_question)
    result.reading_list = _build_reading_list(support_ps, oppose_ps, mixed_ps)

    logger.info(
        f"[Consensus] {result.total_papers} papers analysed: "
        f"{result.support_count} SUPPORT / {result.oppose_count} OPPOSE / "
        f"{result.mixed_count} MIXED / {result.neutral_count} NEUTRAL "
        f"→ {result.consensus_direction} ({result.consensus_pct:.0f}%)"
    )
    return result
