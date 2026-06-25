# Research Machine: Autonomous Research Paper Generation

An AI-powered system that automatically generates Q1/Scopus-quality research papers in Business + AI domains.

## Project Overview

The Research Machine is a fully autonomous system that:
- 🔍 **Searches** scholarly literature (Semantic Scholar, arXiv)
- 💡 **Generates** novel research hypotheses
- 📚 **Conducts** comprehensive literature reviews
- 🔬 **Analyzes** trends and identifies research gaps
- 📝 **Writes** complete IMRAD-structured research papers
- ✅ **Validates** papers against quality standards

**Target Domains**: Business + AI (startups, enterprise AI, ML applications)

**Quality Target**: Q1/Scopus-level research papers

## Architecture

### Core Components

```
research_machine/
├── agents/           # LLM-orchestrated agents (Hypothesis, Literature, Analysis, Writing)
├── search/          # Literature search integrations (Semantic Scholar, arXiv)
├── rag/             # Retrieval-Augmented Generation (embeddings, vector store)
├── quality/         # Quality gates and verification
└── db/              # Database models
```

### Tech Stack

- **Backend**: Python 3.11+ with FastAPI
- **LLM Orchestration**: LangGraph
- **Models**: Claude Haiku (cheap tasks) + Sonnet (critical tasks)
- **Vector DB**: Qdrant (self-hosted, free)
- **Relational DB**: PostgreSQL
- **Search APIs**: Semantic Scholar, arXiv

## Quick Start

### 1. Setup Environment

```bash
# Clone repo and navigate
cd qr2

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys:
# - ANTHROPIC_API_KEY: Get from https://console.anthropic.com
# - SEMANTIC_SCHOLAR_API_KEY: (optional, free tier available)
# - Others are pre-configured for local development
```

### 3. Start Local Services

```bash
# Start PostgreSQL and Qdrant via Docker
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 4. Start API Server

```bash
cd api
python -m uvicorn main:app --reload
```

API will be available at: `http://localhost:8000`

### 5. Test the System

```bash
# Health check
curl http://localhost:8000/health

# Search papers
curl -X POST "http://localhost:8000/research/search?query=AI+startup+funding&source=semantic_scholar"
```

## Development

### Week 1: Infrastructure & Search (CURRENT)

- [x] FastAPI backend setup
- [x] PostgreSQL + Qdrant services
- [x] Semantic Scholar API integration
- [x] arXiv API integration
- [x] Embeddings pipeline
- [x] Vector store (Qdrant)
- [x] Database models

**Next**: Generate test papers using seed research questions

### Week 2: Agent Layer

- [ ] Hypothesis Agent (LangGraph)
- [ ] Literature Agent
- [ ] Citation tracking
- [ ] Paper indexing

### Week 3: Analysis & Writing

- [ ] Analysis Agent
- [ ] Writing Agent (IMRAD structure)
- [ ] Quality gates

### Week 4: Testing & Refinement

- [ ] Generate 3-5 test papers
- [ ] Multi-format output (Markdown, LaTeX, PDF, DOCX)
- [ ] Cost & quality metrics

## File Structure

```
qr2/
├── research_machine/        # Core research engine
│   ├── agents/              # Agent implementations (Week 2+)
│   ├── search/              # Literature search
│   │   ├── semantic_scholar.py
│   │   └── arxiv_search.py
│   ├── rag/                 # Vector search
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   ├── quality/             # Quality gates (Week 3+)
│   ├── db/                  # Database
│   │   └── models.py
│   ├── config.py            # Configuration
│   └── __init__.py
│
├── api/                     # FastAPI application
│   ├── main.py
│   └── __init__.py
│
├── tests/                   # Unit tests
│
├── docker-compose.yml       # Local dev services
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template
└── README.md                # This file
```

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /` - API info

### Literature Search
- `POST /research/search` - Search papers
  - Query params: `query`, `limit`, `source` (semantic_scholar|arxiv)
  
### Research Generation (Week 2+)
- `POST /research/question` - Generate research question
- `POST /research/papers` - Generate papers (coming soon)
- `GET /papers/{id}` - Get paper details (coming soon)

## Cost Model

**Per Paper Cost Estimate**: ~$6 (Phase 1)

- Semantic Scholar API: $0.20
- Model API calls: $5.23
- Infrastructure: $0.50

**Cost Breakdown by Task**:
- Hypothesis generation: $0.08 (Haiku)
- Literature search: $0.15 (Haiku)
- Analysis: $2.50 (Sonnet)
- Writing: $1.80 (Sonnet)
- Peer review: $0.60 (Haiku + Sonnet)
- Novelty check: $0.30 (local + Haiku)

## Quality Standards (Q1/Scopus)

Papers must meet:
- ✅ Novelty: < 0.70 similarity to existing papers
- ✅ Citations: 15+ quality sources (h-index > 5)
- ✅ Recency: 30% from last 3 years
- ✅ Rigor: Clear methodology, limitations acknowledged
- ✅ Business relevance: Connected to market metrics

## Database Setup

The application uses PostgreSQL for metadata and Qdrant for vector search.

```bash
# View database (after docker-compose up)
PGPASSWORD=research_pass psql -h localhost -U research_user -d research_db -c "\dt"

# Qdrant dashboard
# Open http://localhost:6333/dashboard
```

## Logging

Logs are output to console with configurable level (set via `LOG_LEVEL` env var).

```bash
# Debug mode
LOG_LEVEL=DEBUG python -m uvicorn main:app --reload
```

## Contributing

Development workflow:
1. Create feature branch from `claude/automatic-research-machine-cvimt3`
2. Implement changes following the weekly sprint plan
3. Run tests: `pytest tests/`
4. Commit with descriptive messages
5. Push to origin

## References

### Research Machine Benchmarks
- AI Scientist v2: Generates papers with novelty validation
- STORM (Stanford): Multi-stage literature curation
- RD-Agent (Microsoft): Real-world feedback loops
- GPT Researcher: Multi-source aggregation

### Key Papers & Resources
- [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- [arXiv API](https://arxiv.org/help/api)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph/)

## Support

For issues or questions:
1. Check the plan file: `/root/.claude/plans/root-claude-uploads-15f9ad19-4faa-573e-resilient-pond.md`
2. Review API documentation at `http://localhost:8000/docs`

---

**Project Status**: Phase 1 - Week 1 complete ✅

Next milestone: Phase 1 - Week 2 (Agent Layer)