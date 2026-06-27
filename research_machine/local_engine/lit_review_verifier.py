"""
Lit review fact-checker: verify each cited claim against its source paper.

Workflow:
  1. Parse citations from lit review text  ([Author, Year] patterns)
  2. For each citation, extract the surrounding claim sentence
  3. Retrieve that paper's abstract (or full text if available)
  4. Fuzzy-match claim against source text sentences
  5. Annotate lit review with ✓/⚠ markers and return a verification report
"""

import re
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Threshold for abstract-level matching (abstracts are brief summaries)
_ABSTRACT_THRESHOLD = 0.22
# Lower threshold for full-text matching (full text is more specific / paraphrased)
_FULLTEXT_THRESHOLD = 0.18


@dataclass
class CitationEvidence:
    """Verification result for a single citation in the lit review."""
    paper_ref: str               # "[Smith, 2023]"
    paper_title: str             # Full title of cited paper
    claimed_finding: str         # Sentence from lit review about this paper
    source_sentence: str         # Best matching sentence from abstract/full text
    match_score: float           # SequenceMatcher ratio (0–1)
    verified: bool               # True if match_score >= threshold
    source_type: str             # "abstract" | "full_text" | "not_found"
    threshold_used: float        # Threshold applied


@dataclass
class LitReviewVerificationReport:
    """Aggregate verification report for a literature review section."""
    total_citations: int
    verified_count: int
    unverified_count: int
    not_found_count: int
    verification_rate: float          # verified / (verified + unverified)
    evidence: List[CitationEvidence] = field(default_factory=list)
    overall_score: float = 0.0        # 0–10
    passed: bool = False              # score >= 6.0
    annotated_lit_review: str = ""    # Lit review with ✓/⚠ markers


class LitReviewVerifier:
    """Verify lit review citations against source paper abstracts/full text."""

    def verify(
        self,
        lit_review_text: str,
        corpus: list,  # List[Paper]
    ) -> LitReviewVerificationReport:
        """
        Run fact-check on the literature review section.

        Args:
            lit_review_text: Markdown text of the lit review
            corpus: List of Paper objects (with .abstract, optionally .full_text)

        Returns:
            LitReviewVerificationReport with per-citation evidence + annotated text
        """
        # Build lookup: (last_name_lower, year) -> Paper
        paper_lookup = self._build_lookup(corpus)

        # Extract (citation_ref, claim_sentence) pairs from lit review
        citations = self._extract_citations_with_claims(lit_review_text)

        evidence_list: List[CitationEvidence] = []

        for citation_ref, claim_sentence in citations:
            parsed = self._parse_citation(citation_ref)
            if not parsed:
                continue
            author_key, year = parsed
            paper = paper_lookup.get((author_key, year))

            if paper is None:
                evidence_list.append(CitationEvidence(
                    paper_ref=citation_ref,
                    paper_title="[not in corpus]",
                    claimed_finding=claim_sentence[:200],
                    source_sentence="",
                    match_score=0.0,
                    verified=False,
                    source_type="not_found",
                    threshold_used=_ABSTRACT_THRESHOLD,
                ))
                continue

            # Try full text first, fall back to abstract
            full_text = getattr(paper, "full_text", None)
            if full_text and len(full_text) > 200:
                score, matched = self._best_match(claim_sentence, full_text)
                threshold = _FULLTEXT_THRESHOLD
                source_type = "full_text"
                # Also try abstract if full-text score is low
                if score < threshold:
                    a_score, a_matched = self._best_match(claim_sentence, paper.abstract)
                    if a_score > score:
                        score, matched = a_score, a_matched
                        threshold = _ABSTRACT_THRESHOLD
                        source_type = "abstract"
            else:
                score, matched = self._best_match(claim_sentence, paper.abstract)
                threshold = _ABSTRACT_THRESHOLD
                source_type = "abstract"

            verified = score >= threshold
            evidence_list.append(CitationEvidence(
                paper_ref=citation_ref,
                paper_title=paper.title,
                claimed_finding=claim_sentence[:200],
                source_sentence=matched[:200],
                match_score=round(score, 3),
                verified=verified,
                source_type=source_type,
                threshold_used=threshold,
            ))

        # Aggregate stats
        verified_items = [e for e in evidence_list if e.verified]
        unverified_items = [e for e in evidence_list if not e.verified and e.source_type != "not_found"]
        not_found_items = [e for e in evidence_list if e.source_type == "not_found"]

        total = len(verified_items) + len(unverified_items)
        verification_rate = len(verified_items) / total if total > 0 else 1.0
        overall_score = round(verification_rate * 10.0, 1)
        passed = overall_score >= 6.0

        # Annotate lit review with verification markers
        annotated = self._annotate_lit_review(lit_review_text, evidence_list)

        report = LitReviewVerificationReport(
            total_citations=len(evidence_list),
            verified_count=len(verified_items),
            unverified_count=len(unverified_items),
            not_found_count=len(not_found_items),
            verification_rate=verification_rate,
            evidence=evidence_list,
            overall_score=overall_score,
            passed=passed,
            annotated_lit_review=annotated,
        )

        logger.info(
            f"[LitReviewVerifier] {len(verified_items)}/{total} claims verified "
            f"({verification_rate:.0%}), score={overall_score}/10, passed={passed}"
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_lookup(self, corpus: list) -> Dict[Tuple[str, int], object]:
        """Build (last_name_lower, year) -> Paper lookup dict."""
        lookup: Dict[Tuple[str, int], object] = {}
        for paper in corpus:
            if not paper.authors:
                continue
            full_name = paper.authors[0].split(",")[0].strip()
            last = full_name.split()[-1].lower()
            key = (last, paper.year)
            if key not in lookup:
                lookup[key] = paper
        return lookup

    def _extract_citations_with_claims(
        self, text: str
    ) -> List[Tuple[str, str]]:
        """
        Find all [Author, Year] citations and extract the surrounding claim.

        Returns list of (citation_ref, claim_sentence).
        """
        pattern = re.compile(r'\[([A-Za-z][A-Za-z\s\-]+,?\s*\d{4})\]')
        results: List[Tuple[str, str]] = []

        for m in pattern.finditer(text):
            ref = f"[{m.group(1)}]"
            pos = m.start()

            # Find sentence boundaries around the citation
            before = text[:pos]
            after = text[pos:]

            # Sentence start: last '. ', '\n', or beginning
            start_candidates = [
                before.rfind('. '),
                before.rfind('.\n'),
                before.rfind('\n- '),
                before.rfind('\n'),
            ]
            sentence_start = max(c for c in start_candidates if c >= 0) + 1 if any(c >= 0 for c in start_candidates) else 0

            # Sentence end: next '.', '\n', or end
            end_candidates = [
                after.find('. '),
                after.find('.\n'),
                after.find('\n'),
            ]
            sentence_end = pos + min(c for c in end_candidates if c >= 0) + 1 if any(c >= 0 for c in end_candidates) else len(text)

            claim = text[sentence_start:sentence_end].strip()
            # Remove markdown formatting from claim
            claim = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', claim)
            claim = re.sub(r'#+\s*', '', claim)

            if len(claim) >= 20:
                results.append((ref, claim))

        return results

    def _parse_citation(self, citation_ref: str) -> Optional[Tuple[str, int]]:
        """Parse [Author, Year] → (last_name_lower, year) or None."""
        content = citation_ref.strip('[]')
        m = re.match(r'([A-Za-z][A-Za-z.\-\s]+?)[,\s]+(\d{4})', content)
        if m:
            last = m.group(1).strip().split()[-1].lower()
            return (last, int(m.group(2)))
        return None

    def _best_match(self, claim: str, source_text: str) -> Tuple[float, str]:
        """
        Return (best_score, best_sentence) comparing claim against source text.

        Uses SequenceMatcher on each sentence from source_text.
        Also tries keyword overlap as a fallback signal.
        """
        if not source_text:
            return 0.0, ""

        sentences = re.split(r'(?<=[.!?])\s+', source_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if not sentences:
            return 0.0, ""

        claim_lower = claim.lower()
        claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim_lower))

        best_score = 0.0
        best_sent = ""

        for sent in sentences:
            sent_lower = sent.lower()

            # SequenceMatcher ratio
            sm_score = SequenceMatcher(None, claim_lower, sent_lower).ratio()

            # Keyword overlap bonus (normalized)
            sent_words = set(re.findall(r'\b[a-z]{4,}\b', sent_lower))
            if claim_words:
                overlap = len(claim_words & sent_words) / len(claim_words)
                # Combined score: 70% SM + 30% keyword overlap
                combined = 0.70 * sm_score + 0.30 * overlap
            else:
                combined = sm_score

            if combined > best_score:
                best_score = combined
                best_sent = sent

        return best_score, best_sent

    def _annotate_lit_review(
        self,
        text: str,
        evidence: List[CitationEvidence],
    ) -> str:
        """
        Insert ✓ or ⚠ markers after each citation in the lit review.

        Also append a verification summary table at the end.
        """
        # Build citation → marker map (verified status takes priority over unverified)
        markers: Dict[str, str] = {}
        for ev in evidence:
            ref = ev.paper_ref
            if ev.source_type == "not_found":
                if ref not in markers:
                    markers[ref] = "⚠?"
            elif ev.verified:
                markers[ref] = "✓"        # Verified wins — overwrite any prior ⚠
            else:
                if markers.get(ref) != "✓":  # Don't downgrade a verified marker
                    markers[ref] = "⚠"

        annotated = text
        for citation_ref, marker in markers.items():
            escaped = re.escape(citation_ref)
            annotated = re.sub(
                escaped,
                f"{citation_ref}{marker}",
                annotated,
            )

        # Append verification summary table
        if evidence:
            verified = [e for e in evidence if e.verified]
            unverified = [e for e in evidence if not e.verified and e.source_type != "not_found"]
            not_found = [e for e in evidence if e.source_type == "not_found"]
            rate = len(verified) / (len(verified) + len(unverified)) if (verified or unverified) else 1.0

            table_rows = []
            for ev in evidence[:20]:  # Cap table at 20 rows
                status = "✓ Verified" if ev.verified else ("⚠? Not found" if ev.source_type == "not_found" else "⚠ Unverified")
                src = ev.source_type.replace("_", " ").title()
                table_rows.append(
                    f"| {ev.paper_ref} | {ev.paper_title[:45]}... | {status} | {ev.match_score:.2f} | {src} |"
                )

            table = (
                "\n\n---\n\n### Citation Verification Report\n\n"
                f"**Verification rate**: {len(verified)}/{len(verified)+len(unverified)} claims verified "
                f"({rate:.0%}). {len(not_found)} citations not found in corpus.\n\n"
                "| Citation | Source Paper | Status | Score | Source |\n"
                "|----------|-------------|--------|-------|--------|\n"
                + "\n".join(table_rows)
            )
            annotated += table

        return annotated
