"""Analysis Agent — synthesizes literature and identifies research gaps.

Uses Claude Sonnet 4.6 for reasoning depth (~$2.50 per run).
"""

import json
import logging
from typing import List
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_machine.config import settings
from research_machine.agents.hypothesis_agent import HypothesisOutput
from research_machine.agents.literature_agent import PaperSummary

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior business research analyst synthesizing a literature corpus.

Your task:
1. Identify key findings consistently across papers
2. Find contradictions and unresolved questions
3. Spot methodological patterns (which approaches dominate?)
4. Assess how the research question fills a specific gap
5. Recommend a research methodology grounded in what you see

Output ONLY valid JSON — no prose before or after.
"""


class AnalysisOutput(BaseModel):
    key_findings: List[str]            # 5-10 major findings from papers
    research_gaps: List[str]           # 3-5 identified gaps in literature
    methodologies_used: List[str]      # Common research methods seen
    contradictions: List[str]          # Conflicting findings across papers
    trend_analysis: str                # 2-3 paragraph trend summary
    contribution_angle: str            # How hypothesis fills gap
    recommended_methodology: str       # Refined methodology suggestion
    key_citations: List[str]           # ~10-15 most relevant paper titles


class AnalysisAgent:
    """Analyzes literature corpus to identify gaps and inform paper structure."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.llm = ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )

    async def analyze(
        self,
        hypothesis: HypothesisOutput,
        papers: List[PaperSummary],
        domain: str,
    ) -> AnalysisOutput:
        """Analyze literature corpus and synthesize findings.

        Args:
            hypothesis: Research question + context
            papers: List of 15-50 papers with relevance summaries
            domain: 'startup' or 'enterprise'

        Returns:
            AnalysisOutput with key findings, gaps, methodology recommendation
        """
        logger.info(f"[AnalysisAgent] analyzing {len(papers)} papers for domain={domain}")

        # Build context from papers
        papers_context = self._format_papers_for_analysis(papers)

        # Build prompt
        user_prompt = f"""Research Question: {hypothesis.primary_question}

Domain: {domain}
Suggested Methodology: {hypothesis.suggested_methodology}
Target Venues: {', '.join(hypothesis.target_venues)}

Literature Corpus ({len(papers)} papers):
{papers_context}

Analysis Task:
1. What are the 5-10 KEY FINDINGS across these papers?
2. What are the 3-5 RESEARCH GAPS (things NOT studied or contradictions)?
3. What METHODOLOGIES are most common? (survey, case study, experiment, etc.)
4. What CONTRADICTIONS exist between papers?
5. Summarize the TRENDS in 2-3 paragraphs
6. How does the research question FILL A SPECIFIC GAP?
7. RECOMMEND a methodology (survey, case study, experiment, analysis) grounded in what you see

Return as JSON matching AnalysisOutput schema.
"""

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

            # Parse response (strip markdown code fences if present)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())
            result = AnalysisOutput(**data)
            logger.info(f"[AnalysisAgent] analysis complete: {len(result.key_findings)} findings, {len(result.research_gaps)} gaps")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[AnalysisAgent] JSON parse error: {e}")
            raise ValueError(f"Failed to parse analysis response: {e}")
        except Exception as e:
            logger.error(f"[AnalysisAgent] error: {e}")
            raise

    def _format_papers_for_analysis(self, papers: List[PaperSummary]) -> str:
        """Format papers into analysis context."""
        lines = []
        for i, paper in enumerate(papers[:50], 1):  # Limit to top 50 for token efficiency
            line = f"{i}. [{paper.title}] ({paper.year}, {paper.citation_count} citations) - {paper.relevance_summary}"
            if paper.abstract and len(paper.abstract) > 200:
                line += f" Abstract snippet: {paper.abstract[:150]}..."
            lines.append(line)
        return "\n".join(lines)
