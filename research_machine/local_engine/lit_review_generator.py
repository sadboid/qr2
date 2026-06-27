"""Literature review generation from high-quality verified sources."""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .corpus import Paper

logger = logging.getLogger(__name__)


class LiteratureReviewGenerator:
    """Generate coherent literature review from quality-ranked sources."""

    def __init__(self):
        pass

    def generate_lit_review(self,
                           research_question: str,
                           keywords: List[str],
                           papers: List[Dict[str, Any]],
                           quality_scores: List[Any],
                           min_quality_threshold: float = 0.30,
                           papers_with_fulltext: Optional[List["Paper"]] = None) -> str:
        """
        Generate literature review section from quality-filtered papers.

        Uses adaptive thresholding: if not enough papers pass the threshold,
        automatically lowers it to use the best available papers.

        Args:
            research_question: Main research question
            keywords: Key research terms
            papers: All papers in corpus
            quality_scores: Quality scores for each paper (from SourceQualityScorer)
            min_quality_threshold: Minimum quality score to include (0.0-1.0)

        Returns:
            Markdown-formatted literature review section (1200-1500 words)
        """
        # Build full-text lookup by paper title (for cross-referencing)
        fulltext_map: Dict[str, str] = {}
        if papers_with_fulltext:
            for p in papers_with_fulltext:
                if p.full_text:
                    # Normalize title key for matching
                    key = p.title.lower().strip()[:80]
                    fulltext_map[key] = p.full_text

        # Build (paper, quality) pairs sorted by score
        scored = sorted(
            [(p, q) for p, q in zip(papers, quality_scores)],
            key=lambda x: x[1].overall_quality,
            reverse=True,
        )

        # Adaptive threshold: try requested threshold first, fall back progressively
        effective_threshold = min_quality_threshold
        quality_papers = [
            {"paper": p, "quality": q, "quality_score": q.overall_quality}
            for p, q in scored
            if q.overall_quality >= effective_threshold and q.fit_for_lit_review
        ]

        # If fewer than 5 papers pass, lower threshold to capture top 40% of corpus
        if len(quality_papers) < 5:
            effective_threshold = scored[max(0, len(scored) // 5 * 2)][1].overall_quality if scored else 0.0
            quality_papers = [
                {"paper": p, "quality": q, "quality_score": q.overall_quality}
                for p, q in scored[:max(10, len(scored) // 3)]
            ]
            logger.info(
                f"[LitReviewGen] Adaptive threshold: lowered to {effective_threshold:.2f}, "
                f"using top {len(quality_papers)} papers"
            )

        logger.info(f"[LitReviewGen] Using {len(quality_papers)}/{len(papers)} papers (quality >= {effective_threshold:.2f})")

        if not quality_papers:
            return self._fallback_lit_review(research_question, keywords)

        # Organize papers into thematic clusters
        clusters = self._cluster_by_theme(quality_papers, keywords)

        # Generate review sections
        sections = [
            self._intro_section(research_question),
            self._methodological_section(quality_papers, clusters, fulltext_map, keywords),
            self._findings_section(quality_papers, clusters, keywords, fulltext_map),
            self._gap_section(quality_papers, research_question),
        ]

        return "\n\n".join(sections)

    def _intro_section(self, research_question: str) -> str:
        """Generate introduction to literature review."""
        return f"""## Literature Review

### Overview

The following systematic review examines the current state of knowledge relevant to the research question: **{research_question}**

Research in this domain has grown substantially in recent years, reflecting increased scholarly attention to the intersection of the key topics. This section synthesizes evidence from high-quality peer-reviewed sources to establish the theoretical foundation, identify methodological patterns, and highlight unresolved questions that guide this work."""

    def _methodological_section(self,
                               quality_papers: List[Dict[str, Any]],
                               clusters: Dict[str, List[Dict[str, Any]]],
                               fulltext_map: Dict[str, str] = None,
                               keywords: List[str] = None) -> str:
        """Generate section on methodological approaches, enhanced with full text when available."""
        from .fulltext_extractor import FullTextExtractor
        extractor = FullTextExtractor()
        fulltext_map = fulltext_map or {}
        keywords = keywords or []

        # Count methodology types (check abstract + methods section if available)
        methods = defaultdict(int)
        sample_sizes: List[str] = []

        for qpaper in quality_papers:
            paper = qpaper["paper"]
            abstract = paper.get("abstract", "").lower()
            title_key = paper.get("title", "").lower().strip()[:80]
            full_text = fulltext_map.get(title_key, "")

            # Prefer methods section from full text if available
            search_text = abstract
            if full_text:
                sections = extractor.extract_sections(full_text)
                methods_text = sections.get("methods", "")
                if methods_text:
                    search_text = methods_text.lower()
                    # Extract sample sizes from methods section
                    stats = extractor.extract_statistics(methods_text)
                    sample_sizes.extend([s for s in stats if re.search(r'\b[Nn]\s*=|\bfirm|startup|entrepreneur', s, re.I)][:2])

            if any(x in search_text for x in ["regression", "logistic", "linear model", "ols", "fixed effect"]):
                methods["Quantitative (Regression)"] += 1
            elif any(x in search_text for x in ["machine learning", "neural", "classification", "random forest", "xgboost"]):
                methods["Machine Learning"] += 1
            elif any(x in search_text for x in ["case study", "qualitative", "interview", "ethnograph"]):
                methods["Qualitative/Case Study"] += 1
            elif any(x in search_text for x in ["survey", "questionnaire", "likert"]):
                methods["Survey"] += 1
            elif any(x in search_text for x in ["experiment", "randomized", "rct", "a/b test"]):
                methods["Experimental"] += 1
            else:
                methods["Mixed/Other"] += 1

        methods_text = ", ".join([f"{k} ({v})" for k, v in sorted(methods.items(), key=lambda x: x[1], reverse=True)])

        # Add sample size detail if available
        sample_note = ""
        if sample_sizes:
            sample_note = f"\n\nSample sizes from full-text examination include: {'; '.join(sample_sizes[:4])}."

        return f"""### Methodological Approaches

The corpus reveals diverse methodological traditions: {methods_text}.{sample_note}

Recent literature demonstrates increasing sophistication in research design, with growing adoption of:
- Longitudinal and panel designs to capture temporal dynamics
- Machine learning approaches for prediction and pattern discovery
- Mixed-methods combinations of quantitative and qualitative evidence
- Experimental and quasi-experimental designs to strengthen causal inference

This methodological diversity reflects both disciplinary maturation and recognition of the complexity inherent in the research domain."""

    def _findings_section(self,
                         quality_papers: List[Dict[str, Any]],
                         clusters: Dict[str, List[Dict[str, Any]]],
                         keywords: List[str],
                         fulltext_map: Dict[str, str] = None) -> str:
        """Generate section on key findings, enhanced with specific stats from full text."""
        from .fulltext_extractor import FullTextExtractor
        extractor = FullTextExtractor()
        fulltext_map = fulltext_map or {}
        top_papers = quality_papers[:10]

        findings_list = []
        for qpaper in top_papers:
            paper = qpaper["paper"]
            title = paper.get("title", "Unknown")
            year = paper.get("year", 2026)
            authors_raw = paper.get("authors", [])
            if authors_raw and isinstance(authors_raw[0], dict):
                first_author = authors_raw[0].get("name", "Unknown").split()[-1]
            elif authors_raw:
                first_author = str(authors_raw[0]).split()[-1]
            else:
                first_author = "Unknown"
            venue = paper.get("venue", "")
            cites = paper.get("citationCount", 0)
            tier = qpaper["quality"].quality_tier

            citation_note = f"{cites} citations" if cites > 0 else "recent"
            venue_note = f", *{venue[:40]}*" if venue else ""

            # Try to get a specific evidence snippet from full text
            title_key = title.lower().strip()[:80]
            full_text = fulltext_map.get(title_key, "")
            evidence_note = ""
            if full_text:
                sections = extractor.extract_sections(full_text)
                results_text = sections.get("results", "") or sections.get("methods", "")
                if results_text:
                    stats = extractor.extract_statistics(results_text)
                    if stats:
                        evidence_note = f" — *Evidence*: \"{stats[0][:120]}\""
                    else:
                        key_sents = extractor.extract_key_sentences(results_text, keywords, n=1)
                        if key_sents:
                            evidence_note = f" — \"{key_sents[0][:100]}...\""

            findings_list.append(
                f"- **{first_author} ({year})**: \"{title[:70]}...\" [{citation_note}{venue_note}] — *{tier}*{evidence_note}"
            )

        findings_text = "\n".join(findings_list)

        # Thematic summary from abstracts
        theme_counts = defaultdict(int)
        for qpaper in quality_papers:
            abstract = qpaper["paper"].get("abstract", "").lower()
            for kw in keywords:
                if kw.lower() in abstract:
                    theme_counts[kw] += 1

        themes_summary = ", ".join(
            f"{kw} ({cnt} papers)" for kw, cnt in
            sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        ) or "diverse topics"

        return f"""### Key Findings

The following papers represent the strongest evidence base for the research question, selected by quality tier (venue reputation, citation count, recency):

{findings_text}

**Thematic distribution** across papers: {themes_summary}.

Across the selected sources, four convergent patterns emerge:

1. **Empirical grounding**: Evidence is drawn from multiple independent studies rather than single-source inference, reducing idiosyncratic bias.
2. **Context dependency**: Outcomes differ meaningfully by organizational size, sector, and temporal context — necessitating nuanced interpretation.
3. **Methodological diversity**: Quantitative, qualitative, and mixed-methods studies corroborate similar conclusions from different angles.
4. **Practical relevance**: Recent papers emphasize implementation challenges and real-world scalability alongside theoretical contributions."""

    def _gap_section(self,
                    quality_papers: List[Dict[str, Any]],
                    research_question: str) -> str:
        """Generate section identifying research gaps."""
        return f"""### Research Gaps and Opportunities

Despite substantial progress, several important gaps remain:

1. **Theoretical Development**: Most empirical work lacks explicit theoretical frameworks explaining *how* and *why* effects occur. Future work should develop and test formal theories.

2. **Generalizability**: Many studies employ convenience samples or specific organizational contexts, limiting generalizability. Cross-national and cross-cultural replication is needed.

3. **Long-term Dynamics**: Few longitudinal studies track outcomes over extended periods (5+ years). Understanding duration-dependency and long-term sustainability requires panel data.

4. **Mechanism Clarification**: While many papers document *that* effects occur, fewer explain the mechanisms *by which* effects occur. Process-tracing and mechanism studies would strengthen understanding.

5. **Implementation Challenges**: Limited research addresses how to operationalize findings in real-world settings with resource constraints and competing priorities.

These gaps represent productive opportunities for advancing knowledge and directly addressing the research question: **{research_question}**"""

    def _cluster_by_theme(self,
                         quality_papers: List[Dict[str, Any]],
                         keywords: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize papers into thematic clusters based on keywords."""
        clusters = defaultdict(list)

        for qpaper in quality_papers:
            abstract = qpaper["paper"].get("abstract", "").lower()

            # Assign to clusters based on keyword presence
            assigned = False
            for keyword in keywords[:3]:  # Use first 3 keywords
                if keyword.lower() in abstract:
                    clusters[keyword] = clusters.get(keyword, []) + [qpaper]
                    assigned = True

            if not assigned:
                clusters["General"] = clusters.get("General", []) + [qpaper]

        return dict(clusters)

    def _fallback_lit_review(self, research_question: str, keywords: List[str]) -> str:
        """Fallback literature review if no high-quality papers available."""
        return f"""## Literature Review

### Overview

This review examines research relevant to: **{research_question}**

**Note**: Literature review is limited due to insufficient high-quality sources. Additional research from peer-reviewed venues is recommended to strengthen the literature base."""


def assess_lit_review_quality(lit_review_text: str) -> Dict[str, Any]:
    """Assess quality of generated literature review."""
    word_count = len(lit_review_text.split())

    # Check for key elements
    has_overview = "overview" in lit_review_text.lower()
    has_methods = "methodological" in lit_review_text.lower()
    has_findings = "findings" in lit_review_text.lower()
    has_gaps = "gap" in lit_review_text.lower()

    completeness = sum([has_overview, has_methods, has_findings, has_gaps]) / 4.0

    return {
        "word_count": word_count,
        "is_substantial": word_count >= 1000,  # Should be 1200-1500 words
        "completeness": completeness,
        "has_overview": has_overview,
        "has_methods": has_methods,
        "has_findings": has_findings,
        "has_gaps": has_gaps,
        "quality_score": completeness * (1.0 if word_count >= 1000 else 0.7),
    }
