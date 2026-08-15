"""Peer review gate — two-stage review: Haiku quick gate + Sonnet detailed gate."""

import json
import logging
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_machine.config import settings
from research_machine.quality.novelty_check import QualityGateResult

logger = logging.getLogger(__name__)


class QuickReviewResult(BaseModel):
    """Result from Haiku quick review."""

    passed: bool
    issues: list[str]  # Critical issues found


class DetailedReviewResult(BaseModel):
    """Result from Sonnet detailed review."""

    score: float  # 0-10 scale
    feedback: str  # Detailed feedback
    recommendation: str  # "accept", "revision", "reject"


class PeerReviewGate:
    """Two-stage peer review using Haiku (quick) → Sonnet (detailed)."""

    def __init__(self):
        self.haiku = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )
        self.sonnet = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=settings.anthropic_api_key,
            max_tokens=2048,
        )

    async def quick_review(self, paper_content: str) -> QuickReviewResult:
        """Quick Haiku review (~30 sec) checking for critical methodological flaws.

        Args:
            paper_content: Full paper text

        Returns:
            QuickReviewResult with critical issues list
        """
        prompt = f"""Review this research paper for CRITICAL METHODOLOGICAL FLAWS only.

Look for:
- Missing or vague research question
- Completely unclear methodology
- No mention of limitations
- Obvious logical errors

If you find critical issues, list them. Otherwise, return empty list.

Paper:
{paper_content}

Return JSON: {{"issues": ["issue1", "issue2", ...] }}
If no critical issues, return {{"issues": []}}"""

        try:
            response = self.haiku.invoke(
                [HumanMessage(content=prompt)]
            )

            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())
            issues = data.get("issues", [])
            passed = len(issues) == 0

            logger.info(f"[QuickReview] passed={passed}, issues={len(issues)}")

            return QuickReviewResult(
                passed=passed,
                issues=issues,
            )

        except Exception as e:
            logger.error(f"[QuickReview] error: {e}")
            return QuickReviewResult(
                passed=False,
                issues=[f"Review error: {e}"],
            )

    async def detailed_review(
        self, paper_content: str, domain: str
    ) -> DetailedReviewResult:
        """Detailed Sonnet review (~2 min) for Q1 venue acceptance.

        Args:
            paper_content: Full paper text
            domain: 'startup' or 'enterprise'

        Returns:
            DetailedReviewResult with score, feedback, recommendation
        """
        domain_context = (
            "startup ecosystem and business impact"
            if domain == "startup"
            else "enterprise AI adoption and organizational impact"
        )

        prompt = f"""You are an editor of a Q1 research journal in business and AI.
Rate this paper for acceptance on a 0-10 scale.

Focus on:
- Research question clarity & novelty
- Methodology rigor & appropriateness
- Literature review quality
- Contribution to {domain_context}
- Writing clarity
- Limitations acknowledgment

Paper:
{paper_content}

Provide:
1. Score (0-10)
2. 3-4 sentence feedback
3. Recommendation (accept/revision/reject)

Return JSON: {{"score": 7.5, "feedback": "...", "recommendation": "accept"}}
"""

        try:
            response = self.sonnet.invoke(
                [HumanMessage(content=prompt)]
            )

            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())
            score = float(data.get("score", 5.0))
            feedback = data.get("feedback", "")
            recommendation = data.get("recommendation", "revision").lower()

            logger.info(f"[DetailedReview] score={score}, recommendation={recommendation}")

            return DetailedReviewResult(
                score=score,
                feedback=feedback,
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"[DetailedReview] error: {e}")
            return DetailedReviewResult(
                score=0.0,
                feedback=f"Review error: {e}",
                recommendation="reject",
            )

    async def evaluate(
        self, paper_content: str, domain: str
    ) -> QualityGateResult:
        """Run full two-stage review.

        Returns QualityGateResult (passed if Sonnet score >= 7.0)
        """
        # Stage 1: Quick review
        quick = await self.quick_review(paper_content)
        if not quick.passed:
            return QualityGateResult(
                passed=False,
                score=0.0,
                feedback=f"Critical issues found: {'; '.join(quick.issues)}",
            )

        # Stage 2: Detailed review
        detailed = await self.detailed_review(paper_content, domain)

        passed = detailed.score >= 7.0
        feedback = f"Score: {detailed.score}/10. {detailed.feedback}"

        logger.info(f"[PeerReview] final={passed}, score={detailed.score}")

        return QualityGateResult(
            passed=passed,
            score=detailed.score / 10.0,  # Normalize to 0-1
            feedback=feedback,
        )
