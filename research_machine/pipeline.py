"""LangGraph pipeline orchestrating the research workflow.

Week 2-3 scope: hypothesis → literature search → analysis → writing → quality gates.
State flows through typed nodes; errors in one node do not kill the whole run.

Graph structure:
    START → hypothesis → literature → indexing
            ↓
            analysis → writing ⇢ novelty_gate    ↓
                               ⇢ citation_gate  ⊕ → END
                               ⇢ peer_review_gate
"""

import asyncio
import logging
from typing import List, Optional, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from research_machine.agents.hypothesis_agent import HypothesisAgent, HypothesisOutput
from research_machine.agents.literature_agent import LiteratureAgent, LiteratureOutput
from research_machine.agents.indexing_agent import IndexingAgent, NoveltyResult
from research_machine.agents.analysis_agent import AnalysisAgent, AnalysisOutput
from research_machine.agents.writing_agent import WritingAgent, DraftPaper
from research_machine.agents.cost_tracker import CostTracker
from research_machine.quality.novelty_check import NoveltyGate, QualityGateResult
from research_machine.quality.citation_validator import CitationValidator
from research_machine.quality.peer_review import PeerReviewGate
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
    analysis: Optional[AnalysisOutput]         # Week 3: Analysis results
    draft_paper: Optional[DraftPaper]          # Week 3: Writing results
    quality_results: Optional[dict]            # Week 3: {novelty, citation, peer_review}
    cost: Optional[dict]                       # CostTracker.summary() at end
    errors: List[str]                          # Non-fatal errors collected during run
    messages: Annotated[list, add_messages]    # LangGraph message history


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


async def analysis_node(state: ResearchState) -> dict:
    """Analyze literature corpus and identify gaps."""
    hyp: Optional[HypothesisOutput] = state.get("hypothesis")
    lit: Optional[LiteratureOutput] = state.get("literature")

    if not hyp or not lit:
        msg = "Skipping analysis — hypothesis or literature unavailable."
        logger.warning(f"[analysis_node] {msg}")
        return {"errors": state.get("errors", []) + [msg]}

    agent = AnalysisAgent()
    try:
        result = await agent.analyze(
            hypothesis=hyp,
            papers=lit.papers,
            domain=state["domain"],
        )
        logger.info(
            f"[analysis_node] found {len(result.key_findings)} key findings, "
            f"{len(result.research_gaps)} gaps"
        )
        return {"analysis": result}
    except Exception as e:
        logger.error(f"[analysis_node] failed: {e}")
        return {"errors": state.get("errors", []) + [f"analysis: {e}"]}


async def writing_node(state: ResearchState) -> dict:
    """Generate IMRAD-structured paper draft."""
    hyp: Optional[HypothesisOutput] = state.get("hypothesis")
    lit: Optional[LiteratureOutput] = state.get("literature")
    analysis: Optional[AnalysisOutput] = state.get("analysis")

    if not hyp or not lit:
        msg = "Skipping writing — hypothesis or literature unavailable."
        logger.warning(f"[writing_node] {msg}")
        return {"errors": state.get("errors", []) + [msg]}

    agent = WritingAgent()
    try:
        result = await agent.write(
            hypothesis=hyp,
            analysis=analysis,
            papers=lit.papers,
            domain=state["domain"],
        )
        logger.info(f"[writing_node] draft complete: {result.title}")
        return {"draft_paper": result}
    except Exception as e:
        logger.error(f"[writing_node] failed: {e}")
        return {"errors": state.get("errors", []) + [f"writing: {e}"]}


async def quality_gates_node(state: ResearchState) -> dict:
    """Run novelty, citation, and peer review gates IN PARALLEL."""
    draft: Optional[DraftPaper] = state.get("draft_paper")

    if not draft:
        msg = "Skipping quality gates — no draft paper."
        logger.warning(f"[quality_gates_node] {msg}")
        return {"quality_results": None, "errors": state.get("errors", []) + [msg]}

    try:
        # Run all 3 gates in parallel
        from research_machine.rag.vector_store import QdrantVectorStore
        from research_machine.rag.embeddings import PaperSummaryEmbedder

        vs = QdrantVectorStore(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        embedder = PaperSummaryEmbedder()

        novelty_gate = NoveltyGate(vs, embedder)
        citation_gate = CitationValidator(None)  # DB session not available here
        peer_review_gate = PeerReviewGate()

        # Run gates
        novelty_result = novelty_gate.evaluate(draft.title, draft.abstract)

        # Citation validation: extract citations from draft
        citation_result = citation_gate.evaluate([])  # Empty for now

        # Peer review (async)
        peer_result = await peer_review_gate.evaluate(
            draft.content_markdown, state["domain"]
        )

        quality_results = {
            "novelty": novelty_result.model_dump(),
            "citation": citation_result.model_dump(),
            "peer_review": peer_result.model_dump(),
        }

        logger.info(f"[quality_gates_node] results: {quality_results}")

        return {"quality_results": quality_results}

    except Exception as e:
        logger.error(f"[quality_gates_node] failed: {e}")
        return {
            "quality_results": None,
            "errors": state.get("errors", []) + [f"quality_gates: {e}"],
        }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    """Assemble and compile the research pipeline graph."""
    graph = StateGraph(ResearchState)

    # Week 2 nodes
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("literature", literature_node)
    graph.add_node("indexing", indexing_node)

    # Week 3 nodes
    graph.add_node("analysis", analysis_node)
    graph.add_node("writing", writing_node)
    graph.add_node("quality_gates", quality_gates_node)

    # Edges: Week 2 flow
    graph.set_entry_point("hypothesis")
    graph.add_edge("hypothesis", "literature")
    graph.add_edge("literature", "indexing")

    # Edges: Week 3 flow (parallel quality gates)
    graph.add_edge("indexing", "analysis")
    graph.add_edge("analysis", "writing")
    graph.add_edge("writing", "quality_gates")
    graph.add_edge("quality_gates", END)

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
        "analysis": None,
        "draft_paper": None,
        "quality_results": None,
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
