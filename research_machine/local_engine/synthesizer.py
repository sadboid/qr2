"""Extractive synthesis: derive findings, gaps, and structure from a paper corpus."""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

from .corpus import Paper

logger = logging.getLogger(__name__)

_CURRENT_YEAR = 2026

# Phrases that signal a research finding
_FINDING_SIGNALS = [
    "we find", "we found", "results show", "results indicate", "results suggest",
    "our results", "findings show", "findings indicate", "findings suggest",
    "data show", "analysis reveals", "analysis indicates", "we demonstrate",
    "we show", "evidence suggests", "study shows", "study finds", "study reveals",
    "this study", "our study", "we observe", "we report", "significant",
    "positively associated", "negatively associated", "correlated", "predicts",
    "improves", "increases", "decreases", "reduces",
]

# Phrases that signal a research gap
_GAP_SIGNALS = [
    "however", "despite", "little is known", "limited research", "gap in",
    "future research", "future work", "remains unclear", "underexplored",
    "understudied", "lacks", "need for", "we call for", "we suggest future",
    "further research", "limited understanding", "no study", "few studies",
    "scarce literature", "to our knowledge", "to date", "this paper addresses",
]

# Phrases for methodology detection
_METHOD_SIGNALS = {
    "survey": ["survey", "questionnaire", "respondents"],
    "experiment": ["experiment", "randomized", "controlled trial", "rct"],
    "case study": ["case study", "case studies", "qualitative"],
    "longitudinal": ["longitudinal", "panel data", "time series"],
    "meta-analysis": ["meta-analysis", "systematic review", "literature review"],
    "machine learning": ["machine learning", "deep learning", "neural network", "classifier"],
    "regression": ["regression", "ols", "logistic", "probit"],
    "interview": ["interview", "thematic analysis", "grounded theory"],
}

# Terms that strongly indicate a paper is outside business/startup/enterprise scope
_DOMAIN_EXCLUSION_TERMS = {
    "startup": frozenset({
        "hospital", "clinical", "patient", "healthcare", "physician",
        "nursing", "surgery", "disease", "diagnosis", "treatment",
        "therapeutic", "epidemiology", "radiology", "pharmacology",
    }),
    "enterprise": frozenset({
        "hospital", "clinical", "patient", "nursing", "surgery",
        "disease", "diagnosis", "therapeutic", "epidemiology",
    }),
}


def _is_off_domain(paper: "Paper", domain: str, keywords: List[str]) -> bool:
    """Return True if a paper is clearly outside the target domain and should not be cited."""
    exclusion = _DOMAIN_EXCLUSION_TERMS.get(domain, frozenset())
    if not exclusion:
        return False
    text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    has_exclusion = any(term in text for term in exclusion)
    if not has_exclusion:
        return False
    # Only exclude if none of the core keywords are present (avoids over-filtering)
    has_required = any(kw.lower() in text for kw in keywords[:4])
    return not has_required


@dataclass
class SynthesisResult:
    key_findings: List[str]
    research_gaps: List[str]
    methodologies: List[str]
    trends: str
    contribution_angle: str
    top_papers: List[Paper]       # Top 20 by relevance for citation
    all_papers: List[Paper]
    recency_ratio: float
    avg_citation_count: float


def _extract_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def _score_sentence(sentence: str, signals: List[str]) -> float:
    s = sentence.lower()
    return sum(1 for sig in signals if sig in s)


def _top_sentences(corpus: List[Paper], signals: List[str], n: int = 8) -> List[str]:
    scored: List[Tuple[float, str, Paper]] = []
    for paper in corpus:
        for sent in _extract_sentences(paper.abstract):
            score = _score_sentence(sent, signals)
            if score > 0:
                scored.append((score + paper.relevance_score, sent, paper))

    scored.sort(key=lambda x: -x[0])
    seen: set = set()
    result = []
    for _, sent, paper in scored:
        # Deduplicate near-identical sentences
        key = re.sub(r"\W+", "", sent.lower())[:80]
        if key not in seen:
            seen.add(key)
            result.append(f"{sent} [{paper.short_ref()}]")
            if len(result) >= n:
                break
    return result


def _detect_methodologies(corpus: List[Paper]) -> List[str]:
    method_counts: Counter = Counter()
    for paper in corpus:
        text = (paper.title + " " + paper.abstract).lower()
        for method, signals in _METHOD_SIGNALS.items():
            if any(sig in text for sig in signals):
                method_counts[method] += 1
    return [m for m, _ in method_counts.most_common(5)]


def _build_trends_paragraph(
    corpus: List[Paper],
    keywords: List[str],
    research_question: str,
) -> str:
    recent = [p for p in corpus if p.is_recent]
    older = [p for p in corpus if not p.is_recent]
    total = len(corpus)
    recency_pct = round(len(recent) / total * 100) if total else 0

    top3 = corpus[:3]
    top3_titles = "; ".join(f'"{p.title}" ({p.year})' for p in top3)

    venue_counter: Counter = Counter(p.venue for p in corpus if p.venue and p.venue != "arXiv")
    top_venues = ", ".join(v for v, _ in venue_counter.most_common(3)) or "various venues"

    keyword_str = ", ".join(keywords[:4])

    para = (
        f"The literature on {keyword_str} has grown substantially in recent years. "
        f"Of the {total} papers retrieved, {recency_pct}% were published within the last three years, "
        f"indicating an active and expanding research front. "
        f"Prominent recent contributions include {top3_titles}. "
        f"Research appears primarily in {top_venues}. "
    )

    method_list = _detect_methodologies(corpus)
    if method_list:
        para += (
            f"Methodologically, the field employs diverse approaches including "
            f"{', '.join(method_list[:3])}, reflecting its interdisciplinary character. "
        )

    para += (
        f"Despite this growth, the specific question of {research_question.lower()} "
        f"remains underexplored, presenting a clear opportunity for contribution."
    )
    return para


def synthesize(
    corpus: List[Paper],
    research_question: str,
    keywords: List[str],
    domain: str = "startup",
) -> SynthesisResult:
    if not corpus:
        raise ValueError("Corpus is empty — cannot synthesize")

    logger.info(f"Synthesizing {len(corpus)} papers...")

    findings = _top_sentences(corpus, _FINDING_SIGNALS, n=15)
    gaps = _top_sentences(corpus, _GAP_SIGNALS, n=5)
    methodologies = _detect_methodologies(corpus)

    trends = _build_trends_paragraph(corpus, keywords, research_question)

    # Contribution angle: synthesize from gap sentences
    if gaps:
        contribution = (
            f"This paper addresses the identified gap by examining {research_question.lower()}. "
            f"By drawing on {len(corpus)} recent papers, we provide a systematic synthesis "
            f"of existing evidence and identify actionable directions for researchers and practitioners."
        )
    else:
        contribution = (
            f"This paper contributes to the literature on {', '.join(keywords[:2])} "
            f"by synthesizing findings across {len(corpus)} papers and identifying "
            f"convergent themes relevant to {research_question.lower()}."
        )

    recent_count = sum(1 for p in corpus if p.is_recent)
    recency_ratio = recent_count / len(corpus) if corpus else 0.0
    avg_cites = sum(p.citation_count for p in corpus) / len(corpus) if corpus else 0.0

    sorted_corpus = sorted(corpus, key=lambda p: p.relevance_score, reverse=True)

    # Filter off-domain papers from the citation list (all_papers stays intact for stats)
    in_domain = [p for p in sorted_corpus if not _is_off_domain(p, domain, keywords)]
    off_domain_count = len(sorted_corpus) - len(in_domain)
    if off_domain_count:
        logger.info(f"[Synthesizer] Filtered {off_domain_count} off-domain papers from citation list")
    top_papers = (in_domain if len(in_domain) >= 10 else sorted_corpus)[:20]

    return SynthesisResult(
        key_findings=findings,
        research_gaps=gaps,
        methodologies=methodologies,
        trends=trends,
        contribution_angle=contribution,
        top_papers=top_papers,
        all_papers=corpus,
        recency_ratio=round(recency_ratio, 2),
        avg_citation_count=round(avg_cites, 1),
    )
