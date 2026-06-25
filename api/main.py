"""FastAPI application for Research Machine"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from research_machine.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database tables and Qdrant collection on startup."""
    logger.info("Research Machine starting up...")

    # DB: create all tables
    try:
        from research_machine.db.database import init_db
        init_db()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.error(f"DB init failed (is Postgres running?): {e}")

    # Vector store: ensure the papers collection exists
    try:
        from research_machine.rag.vector_store import QdrantVectorStore
        vs = QdrantVectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        created = vs.create_collection("papers", vector_size=384)
        if created:
            logger.info("Qdrant 'papers' collection created.")
        else:
            logger.info("Qdrant 'papers' collection already exists.")
    except Exception as e:
        logger.error(f"Qdrant init failed (is Qdrant running?): {e}")

    yield
    logger.info("Research Machine shutting down.")


app = FastAPI(
    title="Research Machine API",
    description="Autonomous Q1-quality research paper generation system",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    domain: str               # 'startup' or 'enterprise'
    keywords: Optional[List[str]] = None


class PipelineRequest(BaseModel):
    domain: str
    keywords: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.2.0", "environment": settings.environment}


@app.get("/")
async def root():
    return {"name": "Research Machine API", "version": "0.2.0", "status": "running"}


@app.post("/research/search")
async def search_papers(query: str, limit: int = 50, source: str = "semantic_scholar"):
    """Search for research papers from Semantic Scholar or arXiv."""
    from research_machine.search.semantic_scholar import SemanticScholarClient
    from research_machine.search.arxiv_search import ArxivClient

    try:
        if source == "semantic_scholar":
            papers = await SemanticScholarClient().search_papers(query, limit=limit)
        elif source == "arxiv":
            papers = await ArxivClient().search_papers(query, max_results=limit)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

        return {"query": query, "source": source, "count": len(papers), "papers": papers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/question")
async def generate_research_question(body: QuestionRequest):
    """Generate a novel research question using the Hypothesis Agent (Claude Haiku)."""
    from research_machine.agents.hypothesis_agent import HypothesisAgent

    if body.domain not in ("startup", "enterprise"):
        raise HTTPException(status_code=400, detail="domain must be 'startup' or 'enterprise'")

    try:
        agent = HypothesisAgent()
        result = await agent.generate(domain=body.domain, keywords=body.keywords)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Hypothesis agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/pipeline/run")
async def run_research_pipeline(body: PipelineRequest):
    """
    Run the full research pipeline: hypothesis → literature search → paper indexing.

    Returns the pipeline state including generated question, papers found,
    papers indexed into Qdrant, and novelty assessment.
    """
    from research_machine.pipeline import run_pipeline

    if body.domain not in ("startup", "enterprise"):
        raise HTTPException(status_code=400, detail="domain must be 'startup' or 'enterprise'")

    try:
        state = await run_pipeline(domain=body.domain, keywords=body.keywords)

        hyp = state.get("hypothesis")
        lit = state.get("literature")
        nov = state.get("novelty")

        return {
            "domain": body.domain,
            "keywords": body.keywords,
            "question": hyp.primary_question if hyp else None,
            "search_queries_used": lit.search_queries_used if lit else [],
            "papers_found": lit.total_found if lit else 0,
            "papers_indexed": state.get("papers_indexed", 0),
            "novelty": {
                "score": nov.score if nov else None,
                "is_novel": nov.is_novel if nov else None,
                "feedback": nov.feedback if nov else None,
            } if nov else None,
            "errors": state.get("errors", []),
            "hypothesis": hyp.model_dump() if hyp else None,
            "top_papers": [p.model_dump() for p in (lit.papers[:10] if lit else [])],
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
    )
