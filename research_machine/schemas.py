"""Shared data shapes with no agent dependencies.

DraftPaper lived in agents/writing_agent.py, which imports langchain_anthropic
at module load. Everything that merely wanted the shape — the output
formatter, the local engine's CLI — was therefore dragged into the LLM-API
stack, and the local engine's whole point is that it runs without one. The
definition lives here; writing_agent re-exports it, so existing imports keep
working.
"""

from pydantic import BaseModel


class DraftPaper(BaseModel):
    title: str                     # Research paper title
    abstract: str                  # 150-250 word structured abstract
    introduction_md: str           # Introduction section (800-1000 words)
    methods_md: str                # Methods section (400-600 words)
    results_md: str                # Results/Findings section (600-800 words)
    discussion_md: str             # Discussion section (800-1000 words)
    citation_count: int            # Total citation count
    content_markdown: str          # Full paper in Markdown
