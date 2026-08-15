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

# Theory signals — maps canonical theory name to search terms.
# External Enabler + Effectuation lead: they are the NATIVE theories of the
# AI-entrepreneurship literature (per our 14-paper Q1 exemplar corpus — ETP,
# SBE, IJEBR, JBVI all anchor on Davidsson's EE framework, not RBV/TAM).
_THEORY_SIGNALS: dict = {
    "External Enabler Framework": [
        "external enabler", "external enablement", "ee framework",
        "davidsson", "von briel", "enabling mechanism", "venture idea",
        "opportunity structure", "environmental change",
    ],
    "Effectuation": [
        "effectuation", "effectual", "sarasvathy", "causation logic",
        "affordable loss", "bird in hand", "means-driven", "pilot-in-the-plane",
    ],
    "Resource-Based View": [
        "resource-based view", "rbv", "resource based view",
        "dynamic capabilities", "competitive resources", "vrin", "barney",
        "firm resources", "resource heterogeneity",
    ],
    "Technology Acceptance Model": [
        "technology acceptance model", "tam", "perceived usefulness",
        "perceived ease of use", "behavioral intention", "davis 1989",
    ],
    "Social Exchange Theory": [
        "social exchange theory", "social exchange", "reciprocity norm",
        "blau", "trust and commitment", "relational exchange",
    ],
    "Institutional Theory": [
        "institutional theory", "institutional logic", "isomorphism",
        "legitimacy", "neo-institutional", "dimaggio", "powell",
    ],
    "Dynamic Capabilities": [
        "dynamic capabilities", "sensing", "seizing capabilities",
        "reconfiguring", "teece", "knowledge recombination",
    ],
    "Knowledge-Based View": [
        "knowledge-based view", "kbv", "tacit knowledge", "knowledge creation",
        "absorptive capacity", "organizational learning", "cohen and levinthal",
    ],
    "Cognitive Theory": [
        "cognitive theory", "cognitive load", "bounded rationality",
        "heuristics", "cognitive bias", "mental model", "sensemaking",
    ],
    "Agency Theory": [
        "agency theory", "principal-agent", "information asymmetry",
        "moral hazard", "adverse selection", "jensen and meckling",
    ],
    "Human Capital Theory": [
        "human capital theory", "human capital", "education and training",
        "skill accumulation", "becker", "returns to education",
    ],
    "Upper Echelons Theory": [
        "upper echelons", "managerial cognition", "ceo characteristics",
        "top management team", "hambrick and mason",
    ],
}


def _detect_primary_theory(corpus: List[Paper]) -> tuple:
    """Return (theory_name, count) for the most-cited theory in the corpus.
    Falls back to the External Enabler framework — the native theoretical lens
    of the AI-entrepreneurship literature — if no signals found."""
    counts: Counter = Counter()
    for paper in corpus:
        text = _scan_text(paper).lower()
        for theory, signals in _THEORY_SIGNALS.items():
            if any(sig in text for sig in signals):
                counts[theory] += 1
    if counts:
        top_theory, top_n = counts.most_common(1)[0]
        if top_n >= 3:
            return top_theory, top_n
        # A 1-2 paper signal is too thin to anchor the paper's entire
        # Theoretical Framework on (it whipsawed between TAM/KBV/RBV across
        # runs). Below the threshold, use the field's native lens instead.
        return "External Enabler Framework", counts.get("External Enabler Framework", 0)
    return "External Enabler Framework", 0


# Terms that strongly indicate a paper is outside business/startup/enterprise scope
_DOMAIN_EXCLUSION_TERMS = {
    "startup": frozenset({
        "hospital", "clinical", "patient", "healthcare", "physician",
        "nursing", "surgery", "surgical", "disease", "diagnosis", "treatment",
        "therapeutic", "epidemiology", "radiology", "pharmacology", "oncology",
        "forensic", "crime scene", "criminal justice", "courtroom",
        "metaverse", "6g network", "digital twin",
    }),
    "enterprise": frozenset({
        "hospital", "clinical", "patient", "nursing", "surgery", "surgical",
        "disease", "diagnosis", "therapeutic", "epidemiology", "oncology",
        "forensic", "crime scene", "criminal justice", "courtroom",
    }),
}

# Positive vocabulary a paper must show to count as in-domain. This is the
# INCLUSION gate the old filter lacked: previously a glioblastoma-surgery paper
# survived because it mentioned "decision-making" (a query keyword). Now a paper
# must demonstrate business/venture context, not merely echo a generic keyword.
_DOMAIN_VOCAB = {
    "startup": frozenset({
        "startup", "start-up", "entrepreneur", "entrepreneurship", "venture",
        "founder", "sme", "small business", "new firm", "incubator",
        "accelerator", "venture capital", "crowdfunding", "business model",
        "firm performance", "small firm", "self-employ", "spin-off", "spinoff",
    }),
    "enterprise": frozenset({
        "enterprise", "organization", "organisation", "firm", "corporate",
        "management", "business", "workplace", "employee", "manager",
        "industry", "company", "governance",
    }),
}


def _is_off_domain(paper: "Paper", domain: str, keywords: List[str]) -> bool:
    """Return True if a paper is outside the target domain and should not be cited.

    Logic (inclusion-first):
    - Paper shows domain vocabulary and no exclusion signal  → in-domain.
    - Paper shows domain vocabulary but ALSO an exclusion signal (e.g. a
      healthcare-startup paper) → keep only if ≥2 distinct query keywords match.
    - Paper shows NO domain vocabulary at all → off-domain unless ≥2 distinct
      query keywords match (a single generic hit like "AI" no longer rescues it).
    Unknown domains keep the old permissive behaviour (no filtering).
    """
    vocab = _DOMAIN_VOCAB.get(domain)
    if not vocab:
        return False
    text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    exclusion = _DOMAIN_EXCLUSION_TERMS.get(domain, frozenset())

    has_vocab = any(term in text for term in vocab)
    has_exclusion = any(term in text for term in exclusion)
    kw_hits = sum(1 for kw in keywords[:5] if kw.lower() in text)

    if has_vocab and not has_exclusion:
        return False
    return kw_hits < 2


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
    primary_theory: str = "External Enabler Framework"  # dominant theory detected from corpus
    theory_paper_count: int = 0                          # how many papers signal this theory


def _extract_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def _scan_text(paper: Paper) -> str:
    """Text used for signal detection: abstract plus full text when available.

    Full text (arXiv HTML / OA PDF, capped ~12k chars) gives methodology and
    theory detection far more signal than the abstract alone.
    """
    parts = [paper.title or "", paper.abstract or ""]
    if getattr(paper, "full_text", None):
        parts.append(paper.full_text)
    return " ".join(parts)


def _score_sentence(sentence: str, signals: List[str]) -> float:
    s = sentence.lower()
    return sum(1 for sig in signals if sig in s)


def _top_sentences(corpus: List[Paper], signals: List[str], n: int = 8) -> List[str]:
    scored: List[Tuple[float, str, Paper]] = []
    for paper in corpus:
        # Abstract sentences first (most reliable), then full-text sentences when present.
        texts = [paper.abstract]
        if getattr(paper, "full_text", None):
            texts.append(paper.full_text)
        for text in texts:
            for sent in _extract_sentences(text):
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
        text = _scan_text(paper).lower()
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
    primary_theory, theory_count = _detect_primary_theory(corpus)
    logger.info(f"[Synthesizer] Primary theory detected: {primary_theory} (n={theory_count} papers)")

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
    # Q1 empirical/review papers cite 40-60 works; the old [:20] cap threw away
    # 30 papers that had already passed quality gates. Cite up to 50 in-domain.
    top_papers = (in_domain if len(in_domain) >= 10 else sorted_corpus)[:50]

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
        primary_theory=primary_theory,
        theory_paper_count=theory_count,
    )
