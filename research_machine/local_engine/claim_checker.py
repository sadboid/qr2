"""Fact-checking and claim verification for generated papers using abstract-level matching."""

import re
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

from .corpus import Paper

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.25  # Lowered: claims are often paraphrased from abstracts


@dataclass
class ClaimCheckResult:
    """Result of checking a single citation/claim."""
    citation_ref: str                 # e.g., "[Smith, 2023]"
    claim_text: str                   # Extracted claim sentence from generated paper
    paper_abstract: str               # Abstract of cited paper
    match_score: float                # 0.0–1.0 (SequenceMatcher ratio)
    verified: bool                    # True if match_score >= THRESHOLD
    matched_sentence: str             # Best matching sentence from abstract
    reason: str                       # Human-readable explanation


@dataclass
class FactCheckReport:
    """Overall fact-check report for a generated paper."""
    total_citations: int
    verified_count: int
    unverified_count: int
    verification_rate: float          # verified / total
    verified_claims: List[ClaimCheckResult] = field(default_factory=list)
    unverified_claims: List[ClaimCheckResult] = field(default_factory=list)
    contradictions: List[dict] = field(default_factory=list)  # List of dicts: {keyword, paper_a, paper_b, signal_a, signal_b}
    overall_score: float = 0.0        # 0-10 (10 = all verified)
    passed: bool = False              # score >= 6.0


class ClaimChecker:
    """Verify claims in generated papers against source paper abstracts."""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        self.positive_signals = ["increases", "improves", "enhances", "promotes", "facilitates", "accelerates", "strengthens"]
        self.negative_signals = ["decreases", "reduces", "hinders", "impairs", "inhibits", "weakens", "slows"]
        self.uncertain_signals = ["mixed", "no significant", "unclear", "uncertain", "inconclusive"]

    def check(self, generated_paper_md: str, corpus: List[Paper]) -> FactCheckReport:
        """
        Verify claims in generated paper against corpus abstracts.

        Args:
            generated_paper_md: Full markdown content of generated paper
            corpus: List of Paper objects used for generation

        Returns:
            FactCheckReport with verification results
        """
        # Extract all citations from paper
        citations = self._extract_citations(generated_paper_md)
        if not citations:
            logger.info("No citations found in paper")
            return FactCheckReport(
                total_citations=0,
                verified_count=0,
                unverified_count=0,
                verification_rate=0.0,
                overall_score=10.0,
                passed=True
            )

        # Create lookup dict: (last_name, year) -> Paper
        paper_lookup = {(p.authors[0].split(",")[0].lower() if p.authors else "", p.year): p for p in corpus}

        verified = []
        unverified = []

        # Check each citation
        for citation_ref, claim_text in citations:
            # Parse citation: "[Author, Year]" or similar
            match = self._parse_citation(citation_ref)
            if not match:
                logger.debug(f"Could not parse citation: {citation_ref}")
                continue

            author, year = match
            paper = paper_lookup.get((author.lower(), year))

            if not paper:
                # Paper not in corpus (shouldn't happen, but handle gracefully)
                result = ClaimCheckResult(
                    citation_ref=citation_ref,
                    claim_text=claim_text,
                    paper_abstract="[Paper not in corpus]",
                    match_score=0.0,
                    verified=False,
                    matched_sentence="",
                    reason=f"Paper not found in corpus for {citation_ref}"
                )
                unverified.append(result)
                continue

            # Fuzzy-match claim against abstract sentences
            score, matched_sent = self._fuzzy_match_to_abstract(claim_text, paper.abstract)

            verified_flag = score >= self.threshold
            reason = f"Match score: {score:.2f} (threshold: {self.threshold})"

            result = ClaimCheckResult(
                citation_ref=citation_ref,
                claim_text=claim_text,
                paper_abstract=paper.abstract,
                match_score=score,
                verified=verified_flag,
                matched_sentence=matched_sent,
                reason=reason
            )

            if verified_flag:
                verified.append(result)
            else:
                unverified.append(result)

        # Detect contradictions
        contradictions = self._detect_contradictions(corpus)

        # Calculate overall score and determine pass/fail
        total = len(verified) + len(unverified)
        verification_rate = len(verified) / total if total > 0 else 1.0
        overall_score = verification_rate * 10.0  # Score 0-10
        passed = overall_score >= 6.0

        return FactCheckReport(
            total_citations=total,
            verified_count=len(verified),
            unverified_count=len(unverified),
            verification_rate=verification_rate,
            verified_claims=verified,
            unverified_claims=unverified,
            contradictions=contradictions,
            overall_score=round(overall_score, 1),
            passed=passed
        )

    def _extract_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all [Author, Year] citations and their surrounding claim sentences.

        Returns:
            List of (citation_ref, claim_text) tuples
        """
        # Pattern: [Author, Year] or [Author Year]
        citation_pattern = r'\[([A-Za-z\s]+,?\s*\d{4})\]'
        citations = []

        for match in re.finditer(citation_pattern, text):
            citation_ref = f"[{match.group(1)}]"
            start_pos = match.start()

            # Extract sentence containing the citation
            # Find sentence boundaries (., !, ?, newline)
            sentence_start = text.rfind('. ', 0, start_pos) + 2 if text.rfind('. ', 0, start_pos) >= 0 else \
                            text.rfind('! ', 0, start_pos) + 2 if text.rfind('! ', 0, start_pos) >= 0 else \
                            text.rfind('? ', 0, start_pos) + 2 if text.rfind('? ', 0, start_pos) >= 0 else 0

            sentence_end = text.find('.', start_pos)
            if sentence_end < 0:
                sentence_end = len(text)

            claim_text = text[sentence_start:sentence_end].strip()
            if len(claim_text) > 20:  # Only include non-trivial claims
                citations.append((citation_ref, claim_text))

        return citations

    def _parse_citation(self, citation_ref: str) -> Optional[Tuple[str, int]]:
        """
        Parse [Author, Year] format.

        Returns:
            (author_last_name, year) or None if parsing fails
        """
        # Remove brackets
        content = citation_ref.strip('[]')

        # Try pattern: "Author, Year" or "Author Year"
        match = re.match(r'([A-Za-z\s]+)[,\s]+(\d{4})', content)
        if match:
            author = match.group(1).strip().split()[-1]  # Last name only
            year = int(match.group(2))
            return (author, year)

        return None

    def _fuzzy_match_to_abstract(self, claim: str, abstract: str) -> Tuple[float, str]:
        """
        Fuzzy-match claim sentence against abstract sentences using SequenceMatcher.

        Returns:
            (best_match_score, best_matching_sentence)
        """
        # Split abstract into sentences
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return 0.0, ""

        best_score = 0.0
        best_sentence = ""

        for sentence in sentences:
            # Use SequenceMatcher to compute similarity
            ratio = SequenceMatcher(None, claim.lower(), sentence.lower()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_sentence = sentence

        return best_score, best_sentence

    def _detect_contradictions(self, corpus: List[Paper]) -> List[dict]:
        """
        Detect contradictions: find papers with opposing claims about same keywords.

        Returns:
            List of dicts: {keyword, paper_a_ref, paper_b_ref, signal_a, signal_b}
        """
        contradictions = []

        # Build keyword → papers mapping
        keyword_map = {}
        keywords_to_check = ["AI", "productivity", "performance", "improve", "startup", "survival", "resilience"]

        for paper in corpus:
            abstract_lower = (paper.title + " " + paper.abstract).lower()

            for keyword in keywords_to_check:
                if keyword.lower() in abstract_lower:
                    if keyword not in keyword_map:
                        keyword_map[keyword] = []
                    keyword_map[keyword].append(paper)

        # For each keyword, find contradictions
        for keyword, papers in keyword_map.items():
            if len(papers) < 2:
                continue

            # Categorize each paper's signal (positive/negative/uncertain)
            signals = {}  # paper_id -> signal
            paper_map = {}  # paper_id -> paper
            for paper in papers:
                text = (paper.title + " " + paper.abstract).lower()
                signal = self._detect_signal(text)
                signals[paper.paper_id] = signal
                paper_map[paper.paper_id] = paper

            # Find contradictions: positive vs negative about same keyword
            positive_ids = [pid for pid, s in signals.items() if s == "positive"]
            negative_ids = [pid for pid, s in signals.items() if s == "negative"]
            positive_papers = [paper_map[pid] for pid in positive_ids]
            negative_papers = [paper_map[pid] for pid in negative_ids]

            if positive_papers and negative_papers:
                # Report a few contradictions
                for p_pos in positive_papers[:1]:  # Limit to avoid explosion
                    for p_neg in negative_papers[:1]:
                        contradictions.append({
                            "keyword": keyword,
                            "paper_a": f"{p_pos.short_ref()}",
                            "paper_b": f"{p_neg.short_ref()}",
                            "signal_a": "positive (improves/enhances)",
                            "signal_b": "negative (reduces/hinders)",
                            "title_a": p_pos.title[:60],
                            "title_b": p_neg.title[:60]
                        })

        return contradictions

    def _detect_signal(self, text: str) -> str:
        """Detect overall signal in text: positive, negative, or uncertain."""
        positive_count = sum(1 for sig in self.positive_signals if sig in text)
        negative_count = sum(1 for sig in self.negative_signals if sig in text)
        uncertain_count = sum(1 for sig in self.uncertain_signals if sig in text)

        if positive_count > negative_count and positive_count > uncertain_count:
            return "positive"
        elif negative_count > positive_count and negative_count > uncertain_count:
            return "negative"
        else:
            return "uncertain"
