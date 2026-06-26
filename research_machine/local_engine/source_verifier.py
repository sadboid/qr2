"""Source verification system — ensure all sources are real and legitimate."""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import httpx

logger = logging.getLogger(__name__)

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_API_KEY = __import__("os").environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
_ss_semaphore = asyncio.Semaphore(1)
_SS_DELAY = 1.1

# Cache for verified papers (avoid re-verification)
_VERIFICATION_CACHE: Dict[str, dict] = {}


@dataclass
class SourceVerification:
    """Result of source verification."""
    paper_id: str
    title: str
    year: int
    authors: List[str]
    venue: str
    citation_count: int

    exists: bool                           # Paper found in Semantic Scholar
    metadata_match: float                  # 0.0-1.0: how well metadata matches
    citation_legitimate: bool              # Citation count seems reasonable
    peer_reviewed: bool                    # Published vs preprint
    verification_score: float              # 0.0-1.0: overall verification confidence
    issues: List[str]                      # Any problems found

    def is_verified(self, threshold: float = 0.70) -> bool:
        """Paper passes verification if score >= threshold."""
        return self.verification_score >= threshold


class SourceVerifier:
    """Verify sources exist and are legitimate."""

    def __init__(self):
        self.cache = _VERIFICATION_CACHE

    async def verify_paper(self,
                          title: str,
                          year: int,
                          authors: List[str],
                          venue: str = "",
                          citation_count: int = 0) -> SourceVerification:
        """
        Verify a single paper exists in Semantic Scholar with matching metadata.

        Returns: SourceVerification with confidence score and issues list
        """
        # Check cache first
        cache_key = f"{title[:50]}_{year}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        logger.info(f"[SourceVerifier] Verifying: {title[:60]}... ({year})")

        issues = []
        exists = False
        metadata_match = 0.0
        citation_legitimate = False
        peer_reviewed = False
        verification_score = 0.0

        try:
            # Search for paper in Semantic Scholar
            ss_paper = await self._search_in_semantic_scholar(title, year)

            if ss_paper:
                exists = True

                # Check metadata match
                title_match = self._similarity(title, ss_paper.get("title", ""))
                year_match = 1.0 if ss_paper.get("year") == year else 0.5

                metadata_match = (title_match * 0.7) + (year_match * 0.3)

                # Check venue (if provided)
                ss_venue = ss_paper.get("venue", "")
                if venue and ss_venue:
                    venue_match = self._similarity(venue.lower(), ss_venue.lower())
                    if venue_match < 0.3:
                        issues.append(f"Venue mismatch: expected '{venue}', found '{ss_venue}'")

                # Check authors
                ss_authors = [a.get("name", "") for a in ss_paper.get("authors", [])]
                if authors and ss_authors:
                    author_match = self._match_authors(authors, ss_authors)
                    if author_match < 0.5:
                        issues.append(f"Author mismatch: only {author_match:.0%} match")
                else:
                    if not authors:
                        issues.append("No authors provided for verification")

                # Check citation count
                ss_cites = ss_paper.get("citationCount", 0)
                citation_legitimate = self._is_citation_count_reasonable(ss_cites, year)
                if citation_count > 0 and ss_cites > 0:
                    cite_diff = abs(citation_count - ss_cites) / max(citation_count, ss_cites)
                    if cite_diff > 0.3:  # >30% difference
                        issues.append(f"Citation count mismatch: expected {citation_count}, found {ss_cites}")

                # Check peer review status
                peer_reviewed = self._is_peer_reviewed(ss_venue)
                if not peer_reviewed and citation_count < 3:
                    issues.append("Preprint with <3 citations — low credibility")

                # Calculate verification score
                # Formula: 0.4 * metadata_match + 0.3 * citation_legit + 0.2 * peer_review + 0.1 * author_match
                author_score = self._match_authors(authors, ss_authors) if authors else 0.0
                verification_score = (
                    0.4 * metadata_match +
                    0.3 * (1.0 if citation_legitimate else 0.3) +
                    0.2 * (1.0 if peer_reviewed else 0.5) +
                    0.1 * author_score
                )
            else:
                issues.append("Paper not found in Semantic Scholar")
                verification_score = 0.0

        except Exception as e:
            logger.warning(f"Verification error: {e}")
            issues.append(f"Verification error: {str(e)}")
            verification_score = 0.0

        result = SourceVerification(
            paper_id=title[:30],
            title=title,
            year=year,
            authors=authors,
            venue=venue,
            citation_count=citation_count,
            exists=exists,
            metadata_match=metadata_match,
            citation_legitimate=citation_legitimate,
            peer_reviewed=peer_reviewed,
            verification_score=verification_score,
            issues=issues,
        )

        # Cache result
        self.cache[cache_key] = result

        status = "✓" if result.is_verified() else "✗"
        logger.info(f"[SourceVerifier] {status} {title[:50]}... (score: {verification_score:.2f}, issues: {len(issues)})")

        return result

    async def verify_corpus(self, papers: List[Dict[str, Any]]) -> tuple[List[SourceVerification], float]:
        """
        Verify multiple papers in corpus.

        Returns: (verifications, verification_rate)
        """
        verifications = []

        for paper in papers:
            verification = await self.verify_paper(
                title=paper.get("title", ""),
                year=paper.get("year", 2026),
                authors=[a.get("name", "") for a in paper.get("authors", [])],
                venue=paper.get("venue", ""),
                citation_count=paper.get("citationCount", 0),
            )
            verifications.append(verification)

        verified_count = sum(1 for v in verifications if v.is_verified())
        verification_rate = verified_count / len(verifications) if verifications else 0.0

        logger.info(f"[SourceVerifier] Corpus verification: {verified_count}/{len(verifications)} ({verification_rate:.0%})")

        return verifications, verification_rate

    async def _search_in_semantic_scholar(self, title: str, year: int) -> Optional[Dict[str, Any]]:
        """Search Semantic Scholar for paper by title and year."""
        async with _ss_semaphore:
            await asyncio.sleep(_SS_DELAY)
            try:
                headers = {"x-api-key": _SS_API_KEY} if _SS_API_KEY else {}
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(
                        f"{_SS_BASE}/paper/search",
                        params={
                            "query": title,
                            "limit": 1,
                            "fields": "paperId,title,abstract,authors,year,citationCount,venue,url",
                        },
                        headers=headers,
                        timeout=30,
                    )

                    if r.status_code != 200:
                        return None

                    data = r.json().get("data", [])
                    if not data:
                        return None

                    # Return first result if it matches year reasonably well
                    first = data[0]
                    if first.get("year") == year or abs(first.get("year", year) - year) <= 1:
                        return first

                    return None
            except Exception as e:
                logger.debug(f"SS search failed: {e}")
                return None

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (0.0-1.0) using SequenceMatcher."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _match_authors(self, provided: List[str], ss_authors: List[str]) -> float:
        """Calculate how well provided authors match SS authors."""
        if not provided or not ss_authors:
            return 0.0

        matches = 0
        for pauth in provided[:3]:  # Check first 3 provided authors
            pauth_lower = pauth.lower().split()[-1]  # Last name
            for ssauth in ss_authors[:5]:  # Check first 5 SS authors
                ssauth_lower = ssauth.lower().split()[-1]
                if pauth_lower == ssauth_lower:
                    matches += 1
                    break

        return matches / len(provided) if provided else 0.0

    def _is_citation_count_reasonable(self, citation_count: int, year: int) -> bool:
        """Check if citation count is reasonable for paper age."""
        age = 2026 - year

        # Expected citations by age
        # 0-1 years: 0-50
        # 1-3 years: 0-200
        # 3-5 years: 0-500
        # 5+ years: any amount

        if age <= 1:
            return citation_count <= 50
        elif age <= 3:
            return citation_count <= 200
        elif age <= 5:
            return citation_count <= 500
        else:
            return True  # Older papers can have any count

    def _is_peer_reviewed(self, venue: str) -> bool:
        """Check if venue indicates peer review (vs preprint)."""
        if not venue:
            return False

        venue_lower = venue.lower()

        # Known preprint venues
        if any(x in venue_lower for x in ["arxiv", "preprint", "technical report", "working paper"]):
            return False

        # Known peer-reviewed venues
        if any(x in venue_lower for x in ["conference", "journal", "proceedings", "transactions", "review"]):
            return True

        # Default: assume peer-reviewed for named venues
        return len(venue) > 5
