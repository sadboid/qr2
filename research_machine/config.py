from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

    # Database
    database_url: str = "postgresql://research_user:research_pass@localhost:5432/research_db"

    # API Keys — optional so tests can import without credentials
    anthropic_api_key: Optional[str] = None

    # Qdrant Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = "research_api_key"

    # External APIs
    semantic_scholar_api_url: str = "https://api.semanticscholar.org/graph/v1"
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    serpapi_key: Optional[str] = None

    # Application
    environment: str = "development"
    log_level: str = "INFO"

    # Quality Gates
    novelty_threshold: float = 0.70
    min_citations: int = 15
    min_h_index: int = 5
    min_recency_ratio: float = 0.30
    rigor_score_threshold: float = 6.5


settings = Settings()
