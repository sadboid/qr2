"""Writing Agent — generates IMRAD-structured research paper drafts.

Uses Claude Sonnet 4.6 for writing quality (~$1.80 per run).
"""

import json
import logging
from typing import List
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_machine.config import settings
from research_machine.agents.hypothesis_agent import HypothesisOutput
from research_machine.agents.analysis_agent import AnalysisOutput
from research_machine.agents.literature_agent import PaperSummary

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert academic writer crafting research papers for Q1 venues.

Guidelines:
1. Follow IMRAD structure: Introduction, Methods, Results/Findings, Discussion
2. Write for a business or AI audience (depending on domain)
3. Cite papers as [Paper Title, Year] — they will be mapped to IDs later
4. Each section should flow logically and be substantial
5. Include concrete examples from the literature
6. Output ONLY valid JSON — no prose before or after

Writing standards:
- Introduction: 800-1000 words (hook → problem → gap → hypothesis)
- Methods: 400-600 words (proposed research approach with clear methodology)
- Results: 600-800 words (key findings from literature + synthesis)
- Discussion: 800-1000 words (implications, limitations, future work)
- Abstract: 150-250 words (structured: background, gap, question, approach, contribution)
"""


# DraftPaper now lives in research_machine.schemas so that consumers which
# only need the shape (formatter, local engine CLI) do not import langchain.
from research_machine.schemas import DraftPaper  # noqa: E402,F401


class WritingAgent:
    """Generates IMRAD-structured paper drafts from analysis + literature."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.llm = ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=8192,
        )

    async def write(
        self,
        hypothesis: HypothesisOutput,
        analysis: AnalysisOutput,
        papers: List[PaperSummary],
        domain: str,
    ) -> DraftPaper:
        """Generate IMRAD-structured paper draft.

        Args:
            hypothesis: Research question + context
            analysis: Key findings, gaps, methodology recommendation
            papers: Top 15-20 papers for reference (by citation count)
            domain: 'startup' or 'enterprise'

        Returns:
            DraftPaper with IMRAD sections + full markdown
        """
        logger.info(f"[WritingAgent] generating {domain} paper for: {hypothesis.primary_question[:60]}...")

        # Select top papers by citation count for writing context
        sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)
        top_papers = sorted_papers[:20]

        # Build papers reference
        papers_ref = self._format_papers_for_writing(top_papers)

        # Build prompt
        user_prompt = f"""Research Question: {hypothesis.primary_question}

Domain: {domain}
Target Venues: {', '.join(hypothesis.target_venues)}

Analysis Summary:
- Key Findings: {'; '.join(analysis.key_findings[:3])}
- Research Gaps: {'; '.join(analysis.research_gaps[:2])}
- Recommended Methodology: {analysis.recommended_methodology}
- Contribution Angle: {analysis.contribution_angle}

Reference Papers (top {len(top_papers)} by citations):
{papers_ref}

Write a research paper with:
1. TITLE: Concise, research-question-focused title
2. ABSTRACT: 150-250 words (Background, Gap, Research Question, Approach, Expected Contribution)
3. INTRODUCTION: 800-1000 words
   - Hook the reader with business/AI context
   - State the problem & why it matters
   - Review existing approaches (cite papers)
   - Identify the research gap
   - State your research question
4. METHODS: 400-600 words
   - Describe your proposed research approach
   - Explain why this methodology is appropriate
   - Mention data sources or sampling strategy
5. RESULTS: 600-800 words
   - Present key findings from literature that answer your question
   - Synthesize analysis insights
   - Use specific examples from papers
6. DISCUSSION: 800-1000 words
   - Interpret findings & discuss implications
   - Explain how you fill the research gap
   - Acknowledge limitations
   - Suggest future research directions

Citations: Use [Paper Title, Year] format (e.g., [Smith on AI Adoption, 2023])

Return as JSON with fields: title, abstract, introduction_md, methods_md, results_md, discussion_md, citation_count, content_markdown
"""

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

            # Parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())

            # Build full markdown
            full_markdown = f"""# {data['title']}

## Abstract

{data['abstract']}

## Introduction

{data['introduction_md']}

## Methods

{data['methods_md']}

## Results

{data['results_md']}

## Discussion

{data['discussion_md']}
"""

            result = DraftPaper(
                title=data["title"],
                abstract=data["abstract"],
                introduction_md=data["introduction_md"],
                methods_md=data["methods_md"],
                results_md=data["results_md"],
                discussion_md=data["discussion_md"],
                citation_count=data.get("citation_count", 0),
                content_markdown=full_markdown,
            )

            logger.info(f"[WritingAgent] draft complete: {result.title}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[WritingAgent] JSON parse error: {e}")
            raise ValueError(f"Failed to parse writing response: {e}")
        except Exception as e:
            logger.error(f"[WritingAgent] error: {e}")
            raise

    def _format_papers_for_writing(self, papers: List[PaperSummary]) -> str:
        """Format papers as reference list."""
        lines = []
        for i, paper in enumerate(papers, 1):
            line = f"{i}. {paper.title} ({paper.year}) - {paper.authors[0] if paper.authors else 'Unknown'} et al. [{paper.citation_count} citations]"
            lines.append(line)
        return "\n".join(lines)
