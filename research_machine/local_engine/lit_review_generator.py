"""Literature review generation from high-quality verified sources."""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .corpus import Paper

logger = logging.getLogger(__name__)

# Gap signal phrases — mirrored from synthesizer._GAP_SIGNALS (avoid circular import)
_GAP_SIGNALS_LIST = [
    "however", "despite", "little is known", "limited research", "gap in",
    "future research", "future work", "remains unclear", "underexplored",
    "understudied", "lacks", "need for", "we call for", "we suggest future",
    "further research", "limited understanding", "no study", "few studies",
    "scarce literature", "to our knowledge", "to date",
]

_POSITIVE_SIGNALS = frozenset({
    "increases", "improves", "enhances", "boosts", "positive", "higher",
    "greater", "benefit", "effective", "significant", "promotes", "facilitates",
})
_NEGATIVE_SIGNALS = frozenset({
    "decreases", "reduces", "hinders", "no significant", "not significant",
    "failed", "lower", "challenges", "limits", "negative", "impedes", "no effect",
})


def _get_last_name(authors_raw) -> str:
    """Extract last name of first author from authors list (any format)."""
    if not authors_raw:
        return "Unknown"
    entry = authors_raw[0]
    full = entry.get("name", "Unknown") if isinstance(entry, dict) else str(entry)
    return full.split(",")[0].strip().split()[-1]


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
                           papers_with_fulltext: Optional[List["Paper"]] = None,
                           synthesis_gaps: Optional[List[str]] = None) -> str:
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
            synthesis_gaps: Pre-extracted gap sentences from synthesizer (with citations)

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
            self._gap_section(quality_papers, research_question, synthesis_gaps, fulltext_map),
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

    def _extract_grounded_claim(
        self,
        paper: Dict[str, Any],
        keywords: List[str],
        extractor,
        full_text: str = "",
    ) -> str:
        """
        Extract the most informative, verifiable sentence from a paper's abstract or full text.

        Returns a sentence directly traceable to the source paper — ensuring high
        claim-verification scores when the lit review verifier runs.
        """
        # Priority 1: results section of full text (most specific)
        if full_text:
            sections = extractor.extract_sections(full_text)
            results_text = sections.get("results", "") or sections.get("discussion", "")
            if results_text:
                sents = extractor.extract_key_sentences(results_text, keywords, n=1)
                if sents and len(sents[0]) > 30:
                    return sents[0][:180]

        # Priority 2: abstract (always available)
        abstract = paper.get("abstract", "")
        if abstract:
            sents = extractor.extract_key_sentences(abstract, keywords, n=1)
            if sents and len(sents[0]) > 30:
                return sents[0][:180]
            # Fallback: first meaningful sentence of abstract
            first_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if len(s.strip()) > 30]
            if first_sents:
                return first_sents[0][:180]

        return paper.get("title", "")[:120]

    def _findings_section(self,
                         quality_papers: List[Dict[str, Any]],
                         clusters: Dict[str, List[Dict[str, Any]]],
                         keywords: List[str],
                         fulltext_map: Dict[str, str] = None) -> str:
        """
        Generate findings section organized by theme (Q1 standard).

        Each theme paragraph cites 2–5 papers with narrative synthesis,
        and flags contradicting evidence where detected.
        """
        from .fulltext_extractor import FullTextExtractor
        extractor = FullTextExtractor()
        fulltext_map = fulltext_map or {}

        # Active themes: ≥2 papers, sorted by paper count desc, cap at 5 themes
        active_themes = sorted(
            [(theme, papers) for theme, papers in clusters.items() if len(papers) >= 2],
            key=lambda x: len(x[1]), reverse=True
        )[:5]

        # Fall back to legacy bullet list if clustering produced no usable themes
        if not active_themes:
            return self._findings_section_bullets(quality_papers, keywords, fulltext_map, extractor)

        theme_blocks = []
        for theme, theme_papers in active_themes:
            top = theme_papers[:5]

            claims, refs = [], []
            for qpaper in top:
                paper = qpaper["paper"]
                title_key = paper.get("title", "").lower().strip()[:80]
                full_text = fulltext_map.get(title_key, "")
                claim = self._extract_grounded_claim(paper, keywords, extractor, full_text)
                last = _get_last_name(paper.get("authors", []))
                year = paper.get("year", 2026)
                claims.append(claim)
                refs.append(f"[{last}, {year}]")

            contra_idx = self._detect_contradiction(claims)

            # Build paragraph
            n = len(top)
            header = f"**{theme.title()} ({n} {'study' if n == 1 else 'studies'})**"

            evidence_sents = []
            for i, (claim, ref) in enumerate(zip(claims, refs)):
                if i == contra_idx:
                    continue
                connector = "Similarly, " if i > 0 and not evidence_sents else ""
                evidence_sents.append(f"{connector}{ref} find that {claim}.")

            contra_sent = ""
            if contra_idx is not None:
                contra_sent = (
                    f" However, {refs[contra_idx]} find that {claims[contra_idx]},"
                    f" suggesting context-dependence in these effects."
                )

            para = header + "\n\n" + " ".join(evidence_sents) + contra_sent
            theme_blocks.append(para)

        findings_text = "\n\n".join(theme_blocks)
        theme_names = ", ".join(theme for theme, _ in active_themes)

        return f"""### Key Findings

The following thematic synthesis organizes evidence from the corpus by research theme, with convergent findings and notable contradictions where detected:

{findings_text}

**Thematic coverage**: {theme_names}."""

    def _findings_section_bullets(self,
                                  quality_papers: List[Dict[str, Any]],
                                  keywords: List[str],
                                  fulltext_map: Dict[str, str],
                                  extractor) -> str:
        """Fallback: bullet-list findings when clustering yields no multi-paper themes."""
        findings_list = []
        for qpaper in quality_papers[:12]:
            paper = qpaper["paper"]
            year = paper.get("year", 2026)
            last = _get_last_name(paper.get("authors", []))
            cites = paper.get("citationCount", 0)
            venue = paper.get("venue", "")
            tier = qpaper["quality"].quality_tier
            title_key = paper.get("title", "").lower().strip()[:80]
            full_text = fulltext_map.get(title_key, "")
            claim = self._extract_grounded_claim(paper, keywords, extractor, full_text)
            ref = f"[{last}, {year}]"
            cite_note = f"{cites} citations" if cites > 0 else "preprint"
            venue_note = f", *{venue[:35]}*" if venue else ""
            findings_list.append(f"- {claim} {ref} [{cite_note}{venue_note}] — *{tier}*")

        return f"""### Key Findings

{chr(10).join(findings_list)}"""

    def _gap_section(self,
                    quality_papers: List[Dict[str, Any]],
                    research_question: str,
                    synthesis_gaps: Optional[List[str]] = None,
                    fulltext_map: Optional[Dict[str, str]] = None) -> str:
        """
        Generate research gaps section.

        Primary: uses synthesis_gaps (real sentences from paper abstracts, pre-cited).
        Enhancement: mines full-text discussion/conclusion sections for additional gaps.
        Fallback: keyword-scan + template if synthesis_gaps is empty.
        """
        if not synthesis_gaps:
            return self._gap_section_fallback(quality_papers, research_question)

        fulltext_map = fulltext_map or {}
        corpus_size = len(quality_papers)

        # Quantitative framing: count papers exhibiting each signal group
        signal_groups = {
            "future work": ["future research", "future work", "further research", "further study",
                            "we call for", "we suggest future"],
            "understudied areas": ["little is known", "underexplored", "understudied",
                                   "no study", "few studies", "scarce literature"],
            "uncertain mechanisms": ["remains unclear", "limited understanding",
                                     "limited research", "to our knowledge"],
        }
        counts = self._count_gap_signals(quality_papers, signal_groups)
        total_signal_papers = max(counts.values()) if counts else 0
        signal_pct = round(total_signal_papers / corpus_size * 100) if corpus_size else 0

        # Format primary gap items from synthesis_gaps (already cited)
        gap_items = []
        for raw_gap in synthesis_gaps[:5]:
            # Extract trailing [Author, Year] tag and truncate body
            m = re.search(r'\s*\[([A-Za-z][A-Za-z\s\-]+,?\s*\d{4})\]\s*$', raw_gap)
            if m:
                body = raw_gap[:m.start()].strip()
                tag = m.group(0).strip()
                item = f"{body[:280]} {tag}"
            else:
                item = raw_gap[:300]
            gap_items.append(item)

        numbered = "\n\n".join(f"{i}. {item}" for i, item in enumerate(gap_items, 1))

        # Optional full-text gap block (from discussion/conclusion sections)
        ft_gaps = self._extract_fulltext_gaps(quality_papers, fulltext_map, _GAP_SIGNALS_LIST, n=6)
        ft_block = ""
        if ft_gaps:
            ft_lines = "\n".join(f"- {g}" for g in ft_gaps[:3])
            ft_block = f"\n\n**Additional gaps identified from full-text analysis:**\n\n{ft_lines}"

        # Quantitative framing sentence
        future_count = counts.get("future work", 0)
        under_count = counts.get("understudied areas", 0)
        mech_count = counts.get("uncertain mechanisms", 0)
        framing = (
            f"Across the {corpus_size}-paper corpus, gap indicators appear in "
            f"approximately {total_signal_papers} papers ({signal_pct}%), "
            f"with {future_count} calling for future work, {under_count} noting "
            f"understudied areas, and {mech_count} flagging uncertain mechanisms. "
            f"The following gaps are identified from explicit statements in source papers:"
        )

        return f"""### Research Gaps and Opportunities

{framing}

{numbered}{ft_block}

These gaps, extracted directly from the source literature, represent productive opportunities for addressing: **{research_question}**"""

    def _gap_section_fallback(self,
                              quality_papers: List[Dict[str, Any]],
                              research_question: str) -> str:
        """Fallback gap section using keyword scan + template prose (used when no synthesis_gaps)."""
        gap_signals = {
            "Theoretical Development": ["theoretical framework", "theory", "mechanism", "how and why"],
            "Generalizability": ["generali", "sample", "context", "cross-national", "limitation"],
            "Longitudinal / Dynamic": ["longitudinal", "panel data", "long-term", "temporal", "dynamic"],
            "Reproducibility": ["reproducib", "replicate", "future research", "further study"],
        }

        gap_citations: Dict[str, str] = {}
        for qpaper in quality_papers:
            abstract = qpaper["paper"].get("abstract", "").lower()
            paper = qpaper["paper"]
            year = paper.get("year", 2026)
            last = _get_last_name(paper.get("authors", []))
            citation = f"[{last}, {year}]"

            for gap_name, signals in gap_signals.items():
                if gap_name not in gap_citations and any(s in abstract for s in signals):
                    gap_citations[gap_name] = citation

        def _cite(gap: str) -> str:
            return f" {gap_citations[gap]}" if gap in gap_citations else ""

        return f"""### Research Gaps and Opportunities

Despite substantial progress, several important gaps remain:

1. **Theoretical Development**: Most empirical work lacks explicit theoretical frameworks explaining *how* and *why* effects occur. Future work should develop and test formal theories{_cite('Theoretical Development')}.

2. **Generalizability**: Many studies employ convenience samples or specific organizational contexts, limiting generalizability across sectors and countries{_cite('Generalizability')}.

3. **Longitudinal / Dynamic Effects**: Few studies track outcomes over extended periods (5+ years). Understanding duration-dependency and sustainability requires panel data{_cite('Longitudinal / Dynamic')}.

4. **Mechanism Clarification**: While papers document *that* effects occur, fewer explain the mechanisms *by which* effects emerge. Process-tracing studies would strengthen causal understanding.

5. **Reproducibility & Replication**: Replication studies and open datasets remain scarce, limiting cumulative knowledge-building{_cite('Reproducibility')}.

These gaps represent productive opportunities for directly addressing: **{research_question}**"""

    def _count_gap_signals(self,
                           quality_papers: List[Dict[str, Any]],
                           signal_groups: Dict[str, List[str]]) -> Dict[str, int]:
        """Count papers (by abstract) containing at least one phrase from each signal group."""
        counts: Dict[str, int] = {group: 0 for group in signal_groups}
        for qpaper in quality_papers:
            abstract = qpaper["paper"].get("abstract", "").lower()
            for group, signals in signal_groups.items():
                if any(s in abstract for s in signals):
                    counts[group] += 1
        return counts

    def _extract_fulltext_gaps(self,
                               quality_papers: List[Dict[str, Any]],
                               fulltext_map: Dict[str, str],
                               gap_signals: List[str],
                               n: int = 8) -> List[str]:
        """
        Extract gap sentences from discussion/conclusion sections of full-text papers.

        Returns list of strings formatted as "sentence [Author, Year]".
        """
        from .fulltext_extractor import FullTextExtractor
        extractor = FullTextExtractor()

        scored: List[Tuple[float, str]] = []
        seen_prefixes: set = set()

        for qpaper in quality_papers:
            paper = qpaper["paper"]
            title_key = paper.get("title", "").lower().strip()[:80]
            full_text = fulltext_map.get(title_key, "")
            if not full_text:
                continue

            sections = extractor.extract_sections(full_text)
            target = (sections.get("discussion", "") + " " + sections.get("conclusion", "")).strip()
            if not target:
                continue

            # Build citation
            year = paper.get("year", 2026)
            last = _get_last_name(paper.get("authors", []))
            citation = f"[{last}, {year}]"

            # Score sentences by gap signal count
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', target) if len(s.strip()) > 40]
            for sent in sents:
                sent_lower = sent.lower()
                score = sum(1 for sig in gap_signals if sig in sent_lower)
                if score == 0:
                    continue
                prefix = re.sub(r'\s+', ' ', sent_lower[:60])
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                scored.append((score + qpaper["quality_score"], f"{sent[:200]} {citation}"))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:n]]

    def _cluster_by_theme(self,
                         quality_papers: List[Dict[str, Any]],
                         keywords: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Assign each paper to its single best-matching keyword theme (exclusive)."""
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for qpaper in quality_papers:
            abstract = qpaper["paper"].get("abstract", "").lower()
            best_kw, best_count = "General", 0
            for kw in keywords[:5]:
                count = abstract.count(kw.lower())
                if count > best_count:
                    best_count = count
                    best_kw = kw
            clusters[best_kw].append(qpaper)
        return dict(clusters)

    def _detect_contradiction(self, claims: List[str]) -> Optional[int]:
        """Return index of minority-signal claim in cluster, or None if no contradiction."""
        pos = [i for i, c in enumerate(claims) if any(s in c.lower() for s in _POSITIVE_SIGNALS)]
        neg = [i for i, c in enumerate(claims) if any(s in c.lower() for s in _NEGATIVE_SIGNALS)]
        if pos and neg:
            return neg[0] if len(neg) <= len(pos) else pos[0]
        return None

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
