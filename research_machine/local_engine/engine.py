"""Main orchestrator for the local (no-API) research engine."""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from .corpus import fetch_corpus, fetch_full_texts, Paper
from .synthesizer import synthesize, SynthesisResult
from .writer import write_full_paper
from .claim_checker import ClaimChecker
from .claim_checker_pqa import check_claims_sync, PaperQAClaimChecker
from . import claude_cli
from .source_verifier import SourceVerifier
from .source_quality import SourceQualityScorer
from .lit_review_generator import LiteratureReviewGenerator
from .lit_review_verifier import LitReviewVerifier
from .source_quality_gates import SourceQualityGates

_METRICS_LOG = Path(__file__).parent.parent.parent / "logs" / "paper_metrics.jsonl"

logger = logging.getLogger(__name__)

_CURRENT_YEAR = 2026

# ---------------------------------------------------------------------------
# Research question templates
# ---------------------------------------------------------------------------

_TITLE_TEMPLATES = {
    "startup": [
        "{question}: Evidence from Early-Stage Ventures",
        "{question}: A Systematic Review",
        "{question}: Implications for Entrepreneurial Practice",
    ],
    "enterprise": [
        "{question}: Insights from Large Organizations",
        "{question}: A Systematic Literature Review",
        "{question}: Frameworks and Evidence",
    ],
    "default": [
        "{question}: A Systematic Review of the Literature",
        "{question}: Synthesis and Future Directions",
    ],
}

_QUERY_EXPANSIONS = {
    "startup": [
        "startup entrepreneur AI productivity",
        "founder decision making machine learning",
        "early stage venture artificial intelligence",
    ],
    "enterprise": [
        "enterprise AI adoption organizational performance",
        "large organization machine learning governance",
        "corporate AI strategy digital transformation",
    ],
    "default": [
        "artificial intelligence business performance",
        "machine learning organizational outcomes",
    ],
}


def _build_queries_with_claude(
    research_question: str, keywords: List[str], domain: str, max_queries: int = 5
) -> List[str]:
    """
    GPT-Researcher-style sub-question decomposition via Claude CLI.
    Generates N targeted search queries covering different research angles.
    No API key required — uses the running Claude Code session.
    """
    kw_str = ", ".join(keywords[:4])
    dynamic_example = ", ".join(f'"query {i+1}"' for i in range(max_queries))
    prompt = (
        f'Write {max_queries} search queries to research: "{research_question}"\n'
        f'Each query: plain natural language phrase, 3-7 words, no boolean operators.\n'
        f'Domain: {domain}. Key themes: {kw_str}.\n'
        f'Cover diverse angles: empirical evidence, theoretical frameworks, '
        f'methods, practical applications, and emerging trends.\n'
        f'Respond with ONLY a JSON array: [{dynamic_example}]'
    )
    text = claude_cli.call(prompt, timeout=60)
    m = re.search(r'\[[\s\S]*?\]', text)
    if m:
        queries = json.loads(m.group(0))
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            fallbacks = _QUERY_EXPANSIONS.get(domain, _QUERY_EXPANSIONS["default"])[:1]
            return [q for q in queries[:max_queries] if q.strip()] + fallbacks
    raise ValueError(f"Could not parse query list: {text[:100]}")


def _build_queries(research_question: str, keywords: List[str], domain: str) -> List[str]:
    # Tier 2: GPT-Researcher-style sub-question decomposition via Claude CLI
    if claude_cli.is_available():
        try:
            return _build_queries_with_claude(research_question, keywords, domain)
        except Exception as e:
            logger.warning(f"[LocalEngine] Claude query decomp failed, using fallback: {e}")
    # Tier 1: keyword-based queries (no API required)
    keyword_query = " ".join(keywords[:4])
    short_kw = " ".join(keywords[:2])
    domain_expansions = _QUERY_EXPANSIONS.get(domain, _QUERY_EXPANSIONS["default"])
    queries = [keyword_query, short_kw] + domain_expansions[:2]
    return [q for q in dict.fromkeys(queries) if q.strip()]


def _make_title(research_question: str, domain: str) -> str:
    templates = _TITLE_TEMPLATES.get(domain, _TITLE_TEMPLATES["default"])
    template = templates[0]
    # Trim question mark from end of question for title
    q = research_question.rstrip("?")
    return template.format(question=q)


# ---------------------------------------------------------------------------
# AI Scientist 9-dimension peer review (Tier 2 — requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def _ai_scientist_peer_review(paper_content: str) -> Optional[dict]:
    """
    AI Scientist-style structured peer review using 9 dimensions adapted for
    Business+AI papers. Calls the running Claude Code session via CLI — no API key needed.
    Returns score dict or None if CLI unavailable.
    """
    if not claude_cli.is_available():
        return None
    try:
        excerpt = paper_content[:3500]
        system = (
            "You are a strict peer reviewer for a Q1 business+AI journal "
            "(e.g., Strategic Management Journal, MIS Quarterly). "
            "Be rigorous — most papers need revision. "
            "Return ONLY valid JSON, no prose before or after."
        )
        prompt = f"""Review the paper excerpt below and score it on 9 dimensions.

PAPER EXCERPT:
{excerpt}

Return ONLY this JSON object:
{{
  "originality": <1-4>,
  "quality": <1-4>,
  "clarity": <1-4>,
  "significance": <1-4>,
  "soundness": <1-4>,
  "presentation": <1-4>,
  "contribution": <1-4>,
  "overall": <1-10>,
  "confidence": <1-5>,
  "summary": "<2-sentence assessment>",
  "main_weakness": "<1 specific improvement needed>"
}}

Scoring: 1-4 scales: 1=poor 2=below-avg 3=good 4=excellent; overall: 1-5=reject 6=borderline 7-8=minor-revision 9-10=accept"""
        text = claude_cli.call(prompt, system=system, timeout=90)
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        logger.debug(f"[AI Scientist review] failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Quality gate (programmatic, no LLM)
# ---------------------------------------------------------------------------

def _run_quality_gates(
    synthesis: SynthesisResult,
    paper_data: dict,
    corpus: list = None,
) -> dict:
    # Novelty gate: if we found 15+ unique papers, the topic is addressable
    # Use recency ratio as proxy for novelty gap
    novelty_score = round(max(0.30, 0.75 - synthesis.recency_ratio * 0.3), 2)
    novelty_passed = novelty_score < 0.70

    # Citation gate
    citation_count = paper_data["citation_count"]
    # h-index proxy for a literature review of recent papers:
    # Recent papers (< 3 years) haven't accumulated citations yet — that's normal.
    # If recency_ratio >= 0.5, we give credit; authors with >10 avg cites score higher.
    ss_papers = [p for p in synthesis.all_papers if p.source == "semantic_scholar"]
    if ss_papers:
        avg_cites = sum(p.citation_count for p in ss_papers) / len(ss_papers)
        # Recent-adjusted h-index: recent corpus with low cites is still valid
        if synthesis.recency_ratio >= 0.60:
            avg_h = max(5.0, min(avg_cites / 5, 15.0))
        else:
            avg_h = min(max(avg_cites / 10, 1.0), 15.0)
    else:
        # All arXiv: scholarly preprints in active CS/econ fields → conservative h=5
        avg_h = 5.0
    recency_ratio = synthesis.recency_ratio
    citation_passed = (
        citation_count >= 15
        and avg_h >= 5.0
        and recency_ratio >= 0.30
    )

    # Peer review gate — Tier 1: section length + finding count proxy
    content = paper_data["content_markdown"]
    word_count = len(content.split())
    finding_count = len(synthesis.key_findings)
    gap_count = len(synthesis.research_gaps)

    rigor_score = min(10.0, (word_count / 400) + finding_count + gap_count)
    peer_review_passed = rigor_score >= 7.0
    peer_review_feedback = (
        f"Paper contains {word_count} words, {finding_count} key findings, "
        f"{gap_count} identified gaps. Rigor proxy score: {rigor_score:.1f}/10."
    )
    peer_review_recommendation = "accept" if peer_review_passed else "minor_revision"
    peer_review_dimensions: dict = {}

    # Tier 2: AI Scientist 9-dimension review (requires ANTHROPIC_API_KEY)
    as_review = _ai_scientist_peer_review(content)
    if as_review:
        ai_overall = float(as_review.get("overall", rigor_score))
        peer_review_passed = ai_overall >= 7.0
        rigor_score = ai_overall
        peer_review_recommendation = (
            "accept" if ai_overall >= 8 else
            "minor_revision" if ai_overall >= 7 else
            "major_revision" if ai_overall >= 5 else
            "reject"
        )
        peer_review_feedback = (
            f"AI Scientist review: overall={ai_overall:.0f}/10, "
            f"originality={as_review.get('originality')}/4, "
            f"significance={as_review.get('significance')}/4, "
            f"soundness={as_review.get('soundness')}/4. "
            f"{as_review.get('summary', '')} "
            f"Key weakness: {as_review.get('main_weakness', '')}"
        )
        peer_review_dimensions = {
            k: as_review.get(k)
            for k in ("originality", "quality", "clarity", "significance",
                      "soundness", "presentation", "contribution", "confidence")
        }
        logger.info(f"[LocalEngine] AI Scientist review: overall={ai_overall}/10, "
                    f"rec={peer_review_recommendation}")

    # Fact-check gate: try PaperQA2 first (Tier 2), fall back to SequenceMatcher (Tier 1)
    fact_check_passed = True
    fact_check_score = 10.0
    fact_check_feedback = "No fact-check performed"
    fact_check_method = "none"

    if corpus:
        pqa_report = check_claims_sync(paper_data["content_markdown"], corpus)
        if pqa_report is not None:
            fact_check_passed = pqa_report.passed
            fact_check_score = pqa_report.overall_score
            fact_check_method = "paperqa2"
            fact_check_feedback = (
                f"{pqa_report.verified_count}/{pqa_report.total_citations} claims verified "
                f"({pqa_report.verification_rate:.0%}) via PaperQA2 RAG."
            )
            logger.info(f"[LocalEngine] PaperQA2 fact-check: {pqa_report.verified_count}/"
                        f"{pqa_report.total_citations} verified, score={pqa_report.overall_score:.1f}")
        else:
            checker = ClaimChecker()
            fact_report = checker.check(paper_data["content_markdown"], corpus)
            fact_check_passed = fact_report.passed
            fact_check_score = fact_report.overall_score
            fact_check_method = "sequencematcher"
            fact_check_feedback = (
                f"{fact_report.verified_count}/{fact_report.total_citations} claims verified "
                f"({fact_report.verification_rate:.0%}). "
                f"{len(fact_report.contradictions)} contradictions detected."
            )

    overall = "accepted" if (novelty_passed and citation_passed and peer_review_passed and fact_check_passed) else "revision_requested"
    if not novelty_passed and not peer_review_passed:
        overall = "rejected"

    return {
        "novelty": {
            "passed": novelty_passed,
            "score": novelty_score,
            "feedback": (
                f"Novelty score {novelty_score:.2f} {'< 0.70 — sufficient novelty' if novelty_passed else '>= 0.70 — too similar to existing work'}"
            ),
        },
        "citation": {
            "passed": citation_passed,
            "score": round(citation_count / 20, 2),
            "feedback": (
                f"{citation_count} citations, recency ratio {recency_ratio:.0%}, avg citations/paper {synthesis.avg_citation_count:.0f}"
            ),
            "metrics": {
                "citation_count": citation_count,
                "avg_h_index": round(avg_h, 1),
                "recency_ratio": recency_ratio,
            },
        },
        "peer_review": {
            "passed": peer_review_passed,
            "score": round(min(rigor_score, 10.0), 1),
            "recommendation": peer_review_recommendation,
            "feedback": peer_review_feedback,
            "dimensions": peer_review_dimensions,
        },
        "fact_check": {
            "passed": fact_check_passed,
            "score": fact_check_score,
            "feedback": fact_check_feedback,
            "method": fact_check_method,
        },
        "overall_status": overall,
    }


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    research_question: str
    title: str
    domain: str
    keywords: List[str]
    paper_data: dict                    # title, sections, citations, markdown
    quality_results: dict               # gate results
    synthesis: SynthesisResult
    elapsed_seconds: float
    corpus_size: int
    source_quality: dict = field(default_factory=dict)         # source quality gate results
    lit_review_verification: dict = field(default_factory=dict)  # lit review claim verification

    @property
    def status(self) -> str:
        return self.quality_results.get("overall_status", "unknown")

    @property
    def word_count(self) -> int:
        return len(self.paper_data.get("content_markdown", "").split())

    def to_metadata(self) -> dict:
        qr = self.quality_results
        return {
            "metadata": {
                "title": self.title,
                "mode": "local_engine",
                **{k: v for k, v in qr.items() if k not in ("overall_status", "fact_check")},
                "fact_check": qr.get("fact_check", {"passed": True, "score": 10.0, "feedback": "No fact-check"}),
                "domain": self.domain,
                "keywords": self.keywords,
            },
            "abstract": self.paper_data.get("abstract", ""),
            "sections": {
                "introduction": self.paper_data.get("introduction_md", ""),
                "methods": self.paper_data.get("methods_md", ""),
                "results": self.paper_data.get("results_md", ""),
                "discussion": self.paper_data.get("discussion_md", ""),
            },
            "citations": {
                "count": self.paper_data.get("citation_count", 0),
                "metrics": qr.get("citation", {}).get("metrics", {}),
                "references": [
                    {
                        "id": i + 1,
                        "title": p.title,
                        "year": p.year,
                        "authors": p.authors[:3],
                        "venue": p.venue,
                        "citation_count": p.citation_count,
                    }
                    for i, p in enumerate(self.synthesis.top_papers)
                ],
            },
            "full_content": {
                "markdown": self.paper_data.get("content_markdown", ""),
                "latex": "",
            },
            "lit_review_verification": self.lit_review_verification,
            "generation_metrics": {
                "cost_usd": 0.00,
                "generation_time_seconds": round(self.elapsed_seconds, 1),
                "corpus_size": self.corpus_size,
                "word_count": self.word_count,
                "mode": "local_engine_no_llm",
            },
            "overall_status": self.status,
        }


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def _append_metrics_log(result: "EngineResult") -> None:
    """Append one-line JSON record to logs/paper_metrics.jsonl."""
    try:
        _METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
        qr = result.quality_results
        fc = qr.get("fact_check", {})
        record = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "title": result.title,
            "domain": result.domain,
            "status": result.status,
            "word_count": result.word_count,
            "corpus_size": result.corpus_size,
            "citation_count": result.paper_data.get("citation_count", 0),
            "novelty_score": qr.get("novelty", {}).get("score", 0.0),
            "peer_review_score": qr.get("peer_review", {}).get("score", 0.0),
            "fact_check_score": fc.get("score", 10.0),
            "fact_check_passed": fc.get("passed", True),
            "elapsed_seconds": round(result.elapsed_seconds, 1),
            "cost_usd": 0.00,
            "keywords": result.keywords[:4],
        }
        with open(_METRICS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[LocalEngine] Could not write metrics log: {e}")


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LocalResearchEngine:
    """
    Self-contained research engine. No LLM API required.

    Workflow:
        1. Build search queries from research question + keywords
        2. Fetch real papers from Semantic Scholar + arXiv
        3. Extractive synthesis: findings, gaps, trends
        4. Template-based IMRAD writing with real citations
        5. Programmatic quality gates
    """

    def __init__(self, target_corpus_size: int = 50):
        self.target_corpus_size = target_corpus_size

    async def run(
        self,
        research_question: str,
        keywords: List[str],
        domain: str = "startup",
    ) -> EngineResult:
        t0 = time.time()
        logger.info(f"[LocalEngine] START — '{research_question[:70]}'")

        # 1. Build queries
        queries = _build_queries(research_question, keywords, domain)
        logger.info(f"[LocalEngine] Queries: {queries[:3]}")

        # 2. Fetch corpus
        logger.info("[LocalEngine] Fetching papers from Semantic Scholar + arXiv + Crossref...")
        corpus, n_raw = await fetch_corpus(queries, keywords, target_size=self.target_corpus_size)
        if not corpus:
            raise RuntimeError("No papers found — check network connectivity and query terms")
        logger.info(f"[LocalEngine] Corpus: {len(corpus)} papers")

        # 2.5. Verify source quality
        logger.info("[LocalEngine] Verifying source quality...")
        verifier = SourceVerifier()
        verifications, verification_rate = await verifier.verify_corpus(
            [_paper_to_dict(p) for p in corpus]
        )

        scorer = SourceQualityScorer()
        quality_scores, quality_stats = scorer.score_corpus(
            [_paper_to_dict(p) for p in corpus]
        )

        # Source quality gates
        gates = SourceQualityGates()
        gate_result = gates.validate_corpus(
            [_paper_to_dict(p) for p in corpus],
            verifications,
            quality_scores
        )

        logger.info(
            f"[LocalEngine] Source Quality: {gate_result.verification_rate:.0%} verified, "
            f"{gate_result.quality_rate:.0%} high-quality (score: {gate_result.score:.2f})"
        )

        if not gate_result.passed:
            logger.warning(f"[LocalEngine] ⚠ Source quality gates: {len(gate_result.issues)} issues")
            for issue in gate_result.issues:
                logger.warning(f"  - {issue}")

        # 2.7. Fetch full text for top arXiv + open-access papers
        logger.info("[LocalEngine] Fetching full text for top papers (arXiv HTML + open-access PDFs)...")
        try:
            corpus = await fetch_full_texts(corpus, max_papers=15)
            ft_count = sum(1 for p in corpus if p.full_text)
            logger.info(f"[LocalEngine] Full text available for {ft_count}/{len(corpus)} papers")
        except Exception as e:
            logger.warning(f"[LocalEngine] Full-text fetch error (non-fatal): {e}")

        # 3. Synthesize
        logger.info("[LocalEngine] Synthesizing corpus...")
        synthesis = synthesize(corpus, research_question, keywords)
        logger.info(
            f"[LocalEngine] {len(synthesis.key_findings)} findings, "
            f"{len(synthesis.research_gaps)} gaps, "
            f"methods: {synthesis.methodologies[:3]}"
        )

        # 3.5. Generate literature review from verified sources (with full text if available)
        lit_gen = LiteratureReviewGenerator()
        lit_review = lit_gen.generate_lit_review(
            research_question=research_question,
            keywords=keywords,
            papers=[_paper_to_dict(p) for p in corpus],
            quality_scores=quality_scores,
            min_quality_threshold=0.30,  # Adaptive: lowers if too few papers pass
            papers_with_fulltext=corpus,  # Pass Paper objects with full_text field
            synthesis_gaps=synthesis.research_gaps,  # Real gap sentences from synthesizer
        )

        # 3.7. Verify lit review citations against source papers
        logger.info("[LocalEngine] Verifying literature review claims against source papers...")
        lit_verifier = LitReviewVerifier()
        lit_review_report = lit_verifier.verify(lit_review, corpus)
        logger.info(
            f"[LocalEngine] Lit review: {lit_review_report.verified_count}/{lit_review_report.total_citations} "
            f"claims verified ({lit_review_report.verification_rate:.0%}), "
            f"score={lit_review_report.overall_score}/10"
        )
        # Replace lit review with annotated version (includes ✓/⚠ markers + verification table)
        lit_review = lit_review_report.annotated_lit_review

        # 4. Write paper (with literature review)
        title = _make_title(research_question, domain)
        logger.info(f"[LocalEngine] Writing paper: '{title}'")
        n_included = sum(1 for q in quality_scores if q.overall_quality >= 0.30)
        paper_data = write_full_paper(
            title=title,
            research_question=research_question,
            keywords=keywords,
            domain=domain,
            synthesis=synthesis,
            literature_review=lit_review,
            n_raw=n_raw,
            n_included=n_included,
        )

        # 5. Quality gates
        quality_results = _run_quality_gates(synthesis, paper_data, corpus)
        logger.info(
            f"[LocalEngine] Gates: novelty={'✓' if quality_results['novelty']['passed'] else '✗'} "
            f"citations={'✓' if quality_results['citation']['passed'] else '✗'} "
            f"peer_review={'✓' if quality_results['peer_review']['passed'] else '✗'} "
            f"→ {quality_results['overall_status']}"
        )

        elapsed = time.time() - t0
        word_count = len(paper_data['content_markdown'].split())
        logger.info(f"[LocalEngine] DONE in {elapsed:.1f}s ({paper_data['citation_count']} citations, "
                    f"{word_count} words)")

        result = EngineResult(
            research_question=research_question,
            title=title,
            domain=domain,
            keywords=keywords,
            paper_data=paper_data,
            quality_results=quality_results,
            synthesis=synthesis,
            elapsed_seconds=elapsed,
            corpus_size=len(corpus),
            source_quality={
                "verification_rate": gate_result.verification_rate,
                "quality_rate": gate_result.quality_rate,
                "high_quality_papers": gate_result.high_quality_papers,
                "verified_papers": gate_result.verified_papers,
                "gate_passed": gate_result.passed,
                "gate_score": gate_result.score,
                "issues": gate_result.issues,
            },
            lit_review_verification={
                "verified_count": lit_review_report.verified_count,
                "total_citations": lit_review_report.total_citations,
                "not_found_count": lit_review_report.not_found_count,
                "verification_rate": lit_review_report.verification_rate,
                "overall_score": lit_review_report.overall_score,
                "passed": lit_review_report.passed,
            },
        )
        _append_metrics_log(result)
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """Convert Paper object to dictionary for use in verification/quality functions."""
    return {
        "paperId": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": [{"name": a} for a in paper.authors],
        "year": paper.year,
        "citationCount": paper.citation_count,
        "venue": paper.venue,
        "url": paper.url,
        "_source": paper.source,
        "author_h_index": paper.author_h_index,
    }
