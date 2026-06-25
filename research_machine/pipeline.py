"""LangGraph pipeline orchestrating the research workflow.

Week 2 scope: hypothesis → literature search → paper indexing.
State flows through typed nodes; errors in one node do not kill the whole run.

Graph structure:
    START → hypothesis → literature → indexing → END
"""

import asyncio
import logging
from typing import List, Optional, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from research_machine.agents.hypothesis_agent import HypothesisAgent, HypothesisOutput
from research_machine.agents.literature_agent import LiteratureAgent, LiteratureOutput
from research_machine.agents.indexing_agent import IndexingAgent, NoveltyResult
from research_machine.agents.cost_tracker import CostTracker
from research_machine.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    domain: str
    keywords: List[str]
    hypothesis: Optional[HypothesisOutput]
    literature: Optional[LiteratureOutput]
    papers_indexed: int
    novelty: Optional[NoveltyResult]
    cost: Optional[dict]          # CostTracker.summary() at end
    errors: List[str]             # Non-fatal errors collected during run
    messages: Annotated[list, add_messages]   # LangGraph message history


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def hypothesis_node(state: ResearchState) -> dict:
    """Generate a research question from domain + keywords."""
    agent = HypothesisAgent()
    try:
        result = await agent.generate(
            domain=state["domain"],
            keywords=state.get("keywords") or [],
        )
        logger.info(f"[hypothesis_node] question: {result.primary_question[:80]}")
        return {"hypothesis": result}
    except Exception as e:
        logger.error(f"[hypothesis_node] failed: {e}")
        return {"errors": state.get("errors", []) + [f"hypothesis: {e}"]}


async def literature_node(state: ResearchState) -> dict:
    """Search for relevant papers using the hypothesis search queries."""
    hyp: Optional[HypothesisOutput] = state.get("hypothesis")
    if hyp is None:
        msg = "Skipping literature search — no hypothesis available."
        logger.warning(f"[literature_node] {msg}")
        return {"errors": state.get("errors", []) + [msg]}

    agent = LiteratureAgent()
    try:
        result = await agent.search(
            queries=hyp.search_queries,
            domain=state["domain"],
        )
        logger.info(
            f"[literature_node] found {len(result.papers)} papers "
            f"(total raw: {result.total_found})"
        )
        return {"literature": result}
    except Exception as e:
        logger.error(f"[literature_node] failed: {e}")
        return {"errors": state.get("errors", []) + [f"literature: {e}"]}


def indexing_node(state: ResearchState) -> dict:
    """Index found papers into Qdrant and run novelty check on the hypothesis."""
    lit: Optional[LiteratureOutput] = state.get("literature")

    agent = IndexingAgent()
    indexed = 0

    if lit and lit.papers:
        indexed = agent.index_papers(lit.papers)

    # Run novelty check on the hypothesis question itself
    hyp: Optional[HypothesisOutput] = state.get("hypothesis")
    novelty: Optional[NoveltyResult] = None
    if hyp:
        novelty = agent.check_novelty(
            abstract=hyp.primary_question,
            title=hyp.primary_question,
        )
        logger.info(
            f"[indexing_node] novelty score={novelty.score:.2f}, "
            f"is_novel={novelty.is_novel}"
        )

    return {
        "papers_indexed": indexed,
        "novelty": novelty,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    """Assemble and compile the research pipeline graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("literature", literature_node)
    graph.add_node("indexing", indexing_node)

    graph.set_entry_point("hypothesis")
    graph.add_edge("hypothesis", "literature")
    graph.add_edge("literature", "indexing")
    graph.add_edge("indexing", END)

    return graph.compile()


# Singleton compiled pipeline — created once per process
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    domain: str,
    keywords: Optional[List[str]] = None,
) -> ResearchState:
    """
    Run the full research pipeline.

    Args:
        domain: 'startup' or 'enterprise'
        keywords: Optional focus keywords

    Returns:
        Final ResearchState with hypothesis, literature, and indexing results.
    """
    initial_state: ResearchState = {
        "domain": domain,
        "keywords": keywords or [],
        "hypothesis": None,
        "literature": None,
        "papers_indexed": 0,
        "novelty": None,
        "cost": None,
        "errors": [],
        "messages": [],
    }

    pipeline = get_pipeline()
    logger.info(f"Starting research pipeline for domain='{domain}', keywords={keywords}")

    final_state = await pipeline.ainvoke(initial_state)

    # Log cost summary (placeholder — wire to CostTracker in Week 3)
    errors = final_state.get("errors", [])
    if errors:
        logger.warning(f"Pipeline completed with {len(errors)} non-fatal error(s): {errors}")

    logger.info(
        f"Pipeline done — papers indexed: {final_state.get('papers_indexed', 0)}, "
        f"novelty: {final_state.get('novelty')}"
    )

    return final_state
