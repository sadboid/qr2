"""FastAPI application for Research Machine"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from research_machine.config import settings

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Research Machine starting up...")
    yield
    logger.info("Research Machine shutting down...")


app = FastAPI(
    title="Research Machine API",
    description="Autonomous research paper generation system",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.environment
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Research Machine API",
        "version": "0.1.0",
        "status": "running"
    }


@app.post("/research/search")
async def search_papers(query: str, limit: int = 50, source: str = "semantic_scholar"):
    """
    Search for research papers.

    Args:
        query: Search query
        limit: Maximum results (default 50)
        source: 'semantic_scholar' or 'arxiv'

    Returns:
        List of papers matching the query
    """
    from research_machine.search.semantic_scholar import SemanticScholarClient
    from research_machine.search.arxiv_search import ArxivClient

    try:
        if source == "semantic_scholar":
            client = SemanticScholarClient()
            papers = await client.search_papers(query, limit=limit)
        elif source == "arxiv":
            client = ArxivClient()
            papers = await client.search_papers(query, max_results=limit)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

        return {
            "query": query,
            "source": source,
            "count": len(papers),
            "papers": papers
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/question")
async def generate_research_question(domain: str, keywords: list = None):
    """
    Generate a research question in a given domain.

    Args:
        domain: 'startup' or 'enterprise'
        keywords: Optional list of keywords to focus on

    Returns:
        Generated research question
    """
    try:
        if domain not in ["startup", "enterprise"]:
            raise HTTPException(
                status_code=400,
                detail="Domain must be 'startup' or 'enterprise'"
            )

        # Placeholder: will be implemented with Hypothesis Agent
        return {
            "domain": domain,
            "keywords": keywords,
            "question": f"Research question placeholder for {domain} domain",
            "status": "placeholder"
        }

    except Exception as e:
        logger.error(f"Question generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
