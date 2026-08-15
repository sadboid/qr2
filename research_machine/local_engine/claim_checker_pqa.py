"""
PaperQA2-based claim verification — replaces SequenceMatcher fuzzy matching.

Cloned pattern from Future-House/paper-qa (8.8k stars).
Works text-only (no PDF required): abstracts fed as Text objects via aadd_texts().
Scoring: 0–10 relevance score from PaperQA2's embedding retrieval + LLM re-ranking.

Usage (from engine.py):
    from .claim_checker_pqa import PaperQAClaimChecker
    checker = PaperQAClaimChecker()
    report = await checker.check(paper_md, corpus)
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .corpus import Paper

logger = logging.getLogger(__name__)

_HAS_PAPERQA = False
try:
    from paperqa import Docs, Settings, Doc, Text  # type: ignore
    _HAS_PAPERQA = True
except ImportError:
    pass

_HAS_LLM_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
)

# Score threshold: PaperQA2 scores 0–10; ≥4 = claim is relevantly supported
_RELEVANCE_THRESHOLD = 4


@dataclass
class PQAClaimResult:
    citation_ref: str
    claim_text: str
    top_score: int          # 0–10 from PaperQA2
    verified: bool          # score >= threshold
    evidence: str           # best matching passage
    paper_citation: str


@dataclass
class PQAFactReport:
    total_citations: int
    verified_count: int
    unverified_count: int
    verification_rate: float
    verified_claims: List[PQAClaimResult] = field(default_factory=list)
    unverified_claims: List[PQAClaimResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    method: str = "paperqa2"


class PaperQAClaimChecker:
    """
    Claim verifier backed by PaperQA2 RAG.

    Requires: pip install paper-qa
    Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.

    Falls back to None if dependencies unavailable — engine.py then uses
    the original SequenceMatcher-based ClaimChecker.
    """

    def __init__(self, llm: str = "claude-haiku-4-5"):
        self.llm = llm

    @staticmethod
    def is_available() -> bool:
        return _HAS_PAPERQA and _HAS_LLM_KEY

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def check(self, generated_paper_md: str, corpus: List[Paper]) -> Optional[PQAFactReport]:
        """
        Verify claims in generated paper against corpus abstracts via PaperQA2.

        Returns PQAFactReport or None if dependencies unavailable.
        """
        if not self.is_available():
            logger.warning(
                "[PaperQAChecker] paper-qa not installed or no LLM key — "
                "falling back to SequenceMatcher checker"
            )
            return None

        # Build paper lookup: (last_name, year) -> Paper
        paper_lookup = self._build_lookup(corpus)

        # Build PaperQA2 Docs object with all abstracts
        docs = await self._build_docs(corpus)

        # Extract all [Author, Year] citations from generated paper
        citations = self._extract_citations(generated_paper_md)
        if not citations:
            return PQAFactReport(
                total_citations=0, verified_count=0, unverified_count=0,
                verification_rate=1.0, overall_score=10.0, passed=True,
            )

        settings = self._make_settings()
        results: List[PQAClaimResult] = []

        for citation_ref, claim_sentence in citations:
            key = self._parse_citation_key(citation_ref)
            paper = paper_lookup.get(key)
            if paper is None:
                continue

            score, evidence = await self._score_claim(
                docs, claim_sentence, citation_ref, settings
            )
            result = PQAClaimResult(
                citation_ref=citation_ref,
                claim_text=claim_sentence,
                top_score=score,
                verified=score >= _RELEVANCE_THRESHOLD,
                evidence=evidence,
                paper_citation=paper.short_ref(),
            )
            results.append(result)

        if not results:
            return PQAFactReport(
                total_citations=0, verified_count=0, unverified_count=0,
                verification_rate=1.0, overall_score=10.0, passed=True,
            )

        verified = [r for r in results if r.verified]
        unverified = [r for r in results if not r.verified]
        rate = len(verified) / len(results)
        score = round(rate * 10, 1)

        return PQAFactReport(
            total_citations=len(results),
            verified_count=len(verified),
            unverified_count=len(unverified),
            verification_rate=rate,
            verified_claims=verified,
            unverified_claims=unverified,
            overall_score=score,
            passed=score >= 6.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _build_docs(self, corpus: List[Paper]) -> "Docs":
        """Add all paper abstracts to a PaperQA2 Docs object (text-only, no PDF)."""
        docs = Docs()
        for p in corpus:
            if not p.abstract:
                continue
            # Combine abstract + full_text if available for richer matching
            content = p.abstract
            if p.full_text and len(p.full_text) > len(p.abstract):
                # Prepend abstract, then first 2000 chars of full text
                content = p.abstract + "\n\n" + p.full_text[:2000]
            citation_str = f"{p.short_ref()} — {p.title[:80]}"
            docname = re.sub(r"[^a-zA-Z0-9_]", "_", citation_str[:40])
            doc = Doc(docname=docname, dockey=citation_str, citation=citation_str)
            texts = [Text(text=content, name=f"{docname}_main", doc=doc)]
            try:
                await docs.aadd_texts(texts=texts, doc=doc)
            except Exception as e:
                logger.debug(f"[PaperQA] Failed to add paper {docname}: {e}")
        return docs

    async def _score_claim(
        self,
        docs: "Docs",
        claim: str,
        citation_ref: str,
        settings: "Settings",
    ) -> tuple[int, str]:
        """Query PaperQA2 for evidence supporting the claim. Returns (score, evidence_text)."""
        try:
            # aget_evidence is cheaper than aquery — no final LLM answer generation
            session = await docs.aget_evidence(claim, settings=settings)
            if not session.contexts:
                return 0, ""
            top_ctx = max(session.contexts, key=lambda c: c.score)
            return top_ctx.score, top_ctx.context[:300]
        except Exception as e:
            logger.debug(f"[PaperQA] Evidence query failed for {citation_ref}: {e}")
            return 0, ""

    def _make_settings(self) -> "Settings":
        settings = Settings()
        # Use Claude Haiku (cheap) for re-ranking; fall back to GPT-4o-mini
        if os.environ.get("ANTHROPIC_API_KEY"):
            settings.llm = "claude-haiku-4-5-20251001"
            settings.summary_llm = "claude-haiku-4-5-20251001"
        else:
            settings.llm = "gpt-4o-mini"
            settings.summary_llm = "gpt-4o-mini"
        settings.answer.evidence_k = 5
        settings.answer.answer_max_sources = 3
        settings.answer.evidence_relevance_score_cutoff = 1
        return settings

    @staticmethod
    def _build_lookup(corpus: List[Paper]) -> dict:
        lookup = {}
        for p in corpus:
            if p.authors:
                last = p.authors[0].split(",")[0].strip().split()[-1].lower()
                lookup[(last, p.year)] = p
        return lookup

    @staticmethod
    def _parse_citation_key(citation_ref: str) -> tuple:
        m = re.search(r"\[([A-Za-z][A-Za-z\s\-]*),?\s*(\d{4})\]", citation_ref)
        if m:
            last = m.group(1).strip().split()[-1].lower()
            year = int(m.group(2))
            return (last, year)
        return ("", 0)

    @staticmethod
    def _extract_citations(paper_md: str) -> List[tuple]:
        """
        Extract (citation_ref, surrounding_claim_sentence) pairs from markdown.
        Returns list of (ref_string, sentence_containing_ref).
        """
        citation_pattern = re.compile(
            r"\[([A-Za-z][A-Za-z\s\-]+,?\s*\d{4}(?:;\s*[A-Za-z][A-Za-z\s\-]+,?\s*\d{4})*)\]"
        )
        # Split paper into sentences
        sentences = re.split(r"(?<=[.!?])\s+", paper_md)

        results = []
        seen = set()
        for sent in sentences:
            for m in citation_pattern.finditer(sent):
                ref = f"[{m.group(1)}]"
                key = ref + sent[:40]
                if key not in seen:
                    seen.add(key)
                    results.append((ref, sent.strip()))
        return results


# ------------------------------------------------------------------
# Sync wrapper for non-async callers
# ------------------------------------------------------------------

def check_claims_sync(paper_md: str, corpus: List[Paper]) -> Optional[PQAFactReport]:
    """Synchronous entry point — wraps async check() for use in engine.py."""
    checker = PaperQAClaimChecker()
    if not checker.is_available():
        return None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context — create task instead
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, checker.check(paper_md, corpus))
                return future.result(timeout=120)
        return loop.run_until_complete(checker.check(paper_md, corpus))
    except Exception as e:
        logger.warning(f"[PaperQAChecker] check failed: {e}")
        return None
