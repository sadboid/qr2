"""Local research engine — no LLM API required.

Generates real research papers by:
1. Searching Semantic Scholar + arXiv (free APIs)
2. Extractive synthesis from real paper abstracts
3. Template-based IMRAD writing with real citations
"""
from .engine import LocalResearchEngine, EngineResult

__all__ = ["LocalResearchEngine", "EngineResult"]
