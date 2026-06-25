"""Cost tracking for Claude API usage.

Tracks token consumption per agent and computes USD cost at end of pipeline run.
Haiku 4.5 pricing: $0.80/M input tokens, $4.00/M output tokens.
Sonnet 4.6 pricing: $3.00/M input tokens, $15.00/M output tokens.
"""

from dataclasses import dataclass, field
from typing import Dict

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

DEFAULT_PRICING = {"input": 3.00, "output": 15.00}  # Fallback


@dataclass
class CostTracker:
    """Accumulates token usage across multiple LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    model: str = "claude-haiku-4-5-20251001"
    per_agent: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, response, agent_name: str = "unknown") -> None:
        """Record token usage from a langchain_anthropic response."""
        usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("usage", {})

        if hasattr(usage, "input_tokens"):
            inp = usage.input_tokens or 0
            out = usage.output_tokens or 0
        elif isinstance(usage, dict):
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
        else:
            inp, out = 0, 0

        self.input_tokens += inp
        self.output_tokens += out
        self.calls += 1

        if agent_name not in self.per_agent:
            self.per_agent[agent_name] = {"input": 0, "output": 0, "calls": 0}
        self.per_agent[agent_name]["input"] += inp
        self.per_agent[agent_name]["output"] += out
        self.per_agent[agent_name]["calls"] += 1

    def cost_usd(self, model: str = None) -> float:
        """Compute total USD cost for accumulated usage."""
        pricing = MODEL_PRICING.get(model or self.model, DEFAULT_PRICING)
        return (
            self.input_tokens * pricing["input"] / 1_000_000
            + self.output_tokens * pricing["output"] / 1_000_000
        )

    def summary(self) -> Dict:
        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_calls": self.calls,
            "estimated_cost_usd": round(self.cost_usd(), 4),
            "per_agent": self.per_agent,
        }
