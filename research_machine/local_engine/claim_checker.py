"""Fact-checking and claim verification for generated papers using abstract-level matching."""

import re
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

from .corpus import Paper

logger = logging.getLogger(__name__)

# Calibrated for the 50% SequenceMatcher + 50% token-Jaccard blend — same
# formula and threshold LitReviewVerifier validated in Week 8. The old 0.25
# was calibrated for raw SequenceMatcher; with the blend it rejected
# legitimately paraphrased claims (3/33 verified on an otherwise-sound paper).
MATCH_THRESHOLD = 0.18


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
        # Strip the References section — reference entries are not claims, and
        # the narrative-citation pattern would otherwise match entry headers.
        body = re.split(r'\n## References\b', generated_paper_md)[0]

        # Extract all citations from the paper body
        citations = self._extract_citations(body)
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

        # Create lookup dict: (last_name_only, year) -> Paper
        # Key uses last word of first author's name (matches _parse_citation output)
        paper_lookup = {}
        for p in corpus:
            if p.authors:
                full = p.authors[0].split(",")[0].strip()  # e.g. "Joel Becker" or "D. Rajasekaran"
                last = full.split()[-1].lower()             # last word = last name
            else:
                last = ""
            paper_lookup[(last, p.year)] = p

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

            # Prefer full text for matching (more accurate); fall back to abstract
            search_text = paper.full_text if (hasattr(paper, "full_text") and paper.full_text) else paper.abstract
            effective_threshold = self.threshold * 0.85 if search_text != paper.abstract else self.threshold
            score, matched_sent = self._fuzzy_match_to_abstract(claim_text, search_text)

            verified_flag = score >= effective_threshold
            source_label = "full text" if search_text != paper.abstract else "abstract"
            reason = f"Match score: {score:.2f} (threshold: {effective_threshold:.2f}, source: {source_label})"

            result = ClaimCheckResult(
                citation_ref=citation_ref,
                claim_text=claim_text,
                paper_abstract=search_text[:500],  # Store first 500 chars of search text
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

    # Bracket [Author, Year], APA parenthetical (Author, 2023) / (Author et al., 2023)
    # / (Author & Author, 2023), and narrative Author et al. (2023). The old
    # bracket-only pattern missed every APA citation the lit-review generator
    # emits, so fact-check silently examined ~2 of 20 citations per paper.
    _CITATION_PATTERNS = [
        r'\[([A-Za-z][A-Za-z\s]+,?\s*\d{4})\]',
        r'\(([A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][A-Za-z\-]+))?,\s*\d{4}[a-z]?)\)',
        # narrative: "Author (2023)", "Author et al. (2023)", "Author and Author (2023)"
        r'\b([A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][A-Za-z\-]+))?)\s+\((\d{4}[a-z]?)\)',
    ]

    def _extract_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract citations (bracket, APA parenthetical, and narrative styles)
        with their surrounding claim sentences.

        Returns:
            List of (citation_ref, claim_text) tuples, deduplicated by
            (citation, claim) pair.
        """
        matches = []  # (start_pos, normalized "[Author, Year]" ref)
        for i, pattern in enumerate(self._CITATION_PATTERNS):
            for match in re.finditer(pattern, text):
                if i == 2:  # narrative: groups are (author-part, year)
                    author, year = match.group(1), match.group(2)
                else:  # bracket / parenthetical: single group "Author…, Year"
                    content = match.group(1)
                    year_m = re.search(r'\d{4}', content)
                    if not year_m:
                        continue
                    year = year_m.group(0)
                    author = content
                # FIRST author only — corpus lookup keys on the first author's
                # last name, so "Giuggioli and Pellegrini" must yield Giuggioli.
                author = re.split(r',|&|\band\b', author)[0]
                author = author.replace(' et al.', '').replace(' et al', '').strip()
                if author:
                    matches.append((match.start(), f"[{author}, {year}]"))

        citations = []
        seen = set()
        for start_pos, citation_ref in matches:

            # Extract sentence containing the citation
            # Find sentence boundaries (., !, ?, newline)
            sentence_start = text.rfind('. ', 0, start_pos) + 2 if text.rfind('. ', 0, start_pos) >= 0 else \
                            text.rfind('! ', 0, start_pos) + 2 if text.rfind('! ', 0, start_pos) >= 0 else \
                            text.rfind('? ', 0, start_pos) + 2 if text.rfind('? ', 0, start_pos) >= 0 else 0

            sentence_end = text.find('.', start_pos)
            if sentence_end < 0:
                sentence_end = len(text)

            claim_text = text[sentence_start:sentence_end].strip()
            key = (citation_ref, claim_text[:80])
            if len(claim_text) > 20 and key not in seen:
                seen.add(key)
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

        # Handle: "Joel Becker, 2025" / "D. Rajasekaran, 2026" / "Cheng-Kui Huang, 2025"
        match = re.match(r'([A-Za-z][A-Za-z.\-\s]+)[,\s]+(\d{4})', content)
        if match:
            name_part = match.group(1).strip()
            author = name_part.split()[-1].lower()  # Last word = last name, lowercased
            year = int(match.group(2))
            return (author, year)

        return None

    @staticmethod
    def _normalize_claim(claim: str) -> str:
        """Strip citation markers and markdown so framing text doesn't dilute matching."""
        claim = re.sub(r'\[[A-Za-z][A-Za-z\s\-.&]+,?\s*\d{4}\]', '', claim)
        claim = re.sub(r'\([A-Za-z][A-Za-z\s\-.&]+,\s*\d{4}[a-z]?\)', '', claim)
        claim = re.sub(r'^[-*#>]+\s*', '', claim.strip())
        claim = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', claim)
        return re.sub(r'\s+', ' ', claim).strip()

    def _fuzzy_match_to_abstract(self, claim: str, abstract: str) -> Tuple[float, str]:
        """
        Match a claim against abstract sentences with the blend validated in
        LitReviewVerifier: 50% character-level SequenceMatcher + 50% token
        Jaccard. Pure SequenceMatcher under-scored legitimately paraphrased
        claims (same content words, different order/connectives), which made
        the thematic Results synthesis look unverified when it wasn't.

        Returns:
            (best_match_score, best_matching_sentence)
        """
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences:
            return 0.0, ""

        clean = self._normalize_claim(claim).lower()
        claim_words = {w for w in re.findall(r'[a-z]{4,}', clean)}

        best_score = 0.0
        best_sentence = ""
        for sentence in sentences:
            s_low = sentence.lower()
            sm = SequenceMatcher(None, clean, s_low).ratio()
            sent_words = {w for w in re.findall(r'[a-z]{4,}', s_low)}
            union = claim_words | sent_words
            jaccard = len(claim_words & sent_words) / len(union) if union else 0.0
            combined = 0.5 * sm + 0.5 * jaccard
            if combined > best_score:
                best_score = combined
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
