"""Hypothesis Agent — generates novel research questions for business+AI domains.

Uses Claude Haiku 4.5 for cost efficiency (~$0.08 per run).
"""

import json
import logging
from typing import List, Optional
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_machine.config import settings

logger = logging.getLogger(__name__)

# Snapshot of known research trends — informs novelty detection
_KNOWN_TRENDS = """
Current saturated areas (avoid):
- Basic LLM chatbot evaluation benchmarks
- Generic transformer architecture improvements
- Standard NLP task fine-tuning comparisons

Emerging gaps (good targets):
- AI-driven startup operational efficiency measurement frameworks
- Longitudinal studies of AI adoption ROI in SMEs
- Founder cognitive load reduction via agentic AI tools
- Predicting Series A success using ML on public signals (GitHub, LinkedIn, Crunchbase)
- Enterprise AI governance maturity models with empirical validation
- LLM hallucination impact on business decision quality
- Agentic AI workflows in knowledge-intensive professions (legal, finance, consulting)
- AI-enabled dynamic pricing in platform economies
- Organisational network analysis of AI adoption diffusion
- Cross-cultural differences in enterprise AI resistance
"""

_SYSTEM_PROMPT = f"""You are a senior business & AI researcher helping identify novel, publishable research questions.

Known research landscape:
{_KNOWN_TRENDS}

Rules:
1. The question must be empirically testable (case study, survey, natural experiment, secondary data)
2. The contribution must be clear and non-obvious
3. Target Q1/Scopus venues in management, information systems, or applied AI
4. Output ONLY valid JSON — no prose before or after
"""


class HypothesisOutput(BaseModel):
    primary_question: str       # The main research question
    angles: List[str]           # 3-5 novel research angles to investigate
    search_queries: List[str]   # 4-6 literature search queries for the Literature Agent
    rationale: str              # Why this is novel and publishable
    suggested_methodology: str  # Brief methodology suggestion
    target_venues: List[str]    # 2-3 Q1 venue suggestions
    domain: str                 # 'startup' or 'enterprise'


class HypothesisAgent:
    """Generates novel research hypotheses using Claude Haiku."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.llm = ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )

    async def generate(
        self,
        domain: str,
        keywords: Optional[List[str]] = None,
    ) -> HypothesisOutput:
        """
        Generate a novel research question for the given domain and keywords.

        Args:
            domain: 'startup' or 'enterprise'
            keywords: Optional focus keywords (e.g. ['AI', 'founder', 'productivity'])

        Returns:
            HypothesisOutput with question, angles, and search queries
        """
        if domain not in ("startup", "enterprise"):
            raise ValueError("domain must be 'startup' or 'enterprise'")

        kw_str = ", ".join(keywords) if keywords else "none specified"
        domain_context = (
            "startup ecosystems, founder behaviour, early-stage company dynamics, venture capital"
            if domain == "startup"
            else "enterprise digital transformation, large organisation AI adoption, MLOps, AI ROI"
        )

        user_msg = f"""Generate a novel research question for the {domain} domain.

Domain context: {domain_context}
Focus keywords: {kw_str}

Produce a JSON object with these exact keys:
{{
  "primary_question": "The specific, empirically testable research question",
  "angles": ["angle 1", "angle 2", "angle 3"],
  "search_queries": ["query 1", "query 2", "query 3", "query 4"],
  "rationale": "Why this is novel and what gap it fills",
  "suggested_methodology": "Brief method (e.g. survey of 200 SMEs, secondary data from Crunchbase)",
  "target_venues": ["Journal of Management Information Systems", "..."],
  "domain": "{domain}"
}}"""

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        logger.info(f"Generating hypothesis for domain='{domain}', keywords={keywords}")
        response = await self.llm.ainvoke(messages)

        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            output = HypothesisOutput(**data)
            logger.info(f"Hypothesis generated: {output.primary_question[:80]}...")
            return output
        except Exception as e:
            logger.error(f"Failed to parse hypothesis output: {e}\nRaw: {raw[:300]}")
            raise ValueError(f"Hypothesis agent returned unparseable JSON: {e}") from e
