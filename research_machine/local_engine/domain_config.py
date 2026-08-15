"""Domain-specific vocabulary and metrics injected into paper sections."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DomainVocabulary:
    name: str
    metrics: List[str]           # Business metrics to reference
    terminology: List[str]       # Domain-specific terms
    context_phrases: List[str]   # Contextualizing phrases for results
    practical_phrases: List[str] # Practical implication phrases
    venue_types: List[str]       # Publication venues to target
    growth_indicators: List[str] # Domain-specific growth signals


STARTUP_VOCAB = DomainVocabulary(
    name="startup",
    metrics=[
        "total addressable market (TAM)",
        "serviceable addressable market (SAM)",
        "customer acquisition cost (CAC)",
        "lifetime value (LTV)",
        "monthly recurring revenue (MRR)",
        "annual recurring revenue (ARR)",
        "runway (months of operating capital)",
        "burn rate",
        "net revenue retention (NRR)",
        "conversion rate",
        "churn rate",
        "gross margin",
        "product-market fit score",
        "viral coefficient (K-factor)",
        "time-to-value (TTV)",
    ],
    terminology=[
        "early-stage startups",
        "Series A/B/C funding rounds",
        "venture capital (VC) backed ventures",
        "bootstrapped startups",
        "founder-led organizations",
        "pivot strategy",
        "minimum viable product (MVP)",
        "go-to-market (GTM) strategy",
        "growth hacking",
        "product-led growth (PLG)",
        "lean startup methodology",
        "agile development cycles",
        "runway extension",
        "exit strategy (M&A, IPO)",
        "accelerator/incubator programs",
    ],
    context_phrases=[
        "In the startup ecosystem, where resources are constrained and timelines are compressed,",
        "For early-stage founders navigating product-market fit,",
        "Within venture-backed startup environments,",
        "Across the startup lifecycle from pre-seed to Series B,",
        "In high-growth, high-uncertainty startup contexts,",
    ],
    practical_phrases=[
        "Founders can leverage these findings to optimize CAC/LTV ratios.",
        "Early-stage teams should prioritize metrics that signal product-market fit.",
        "Venture-backed startups with limited runway benefit most from rapid iteration cycles.",
        "Accelerator programs can integrate these evidence-based practices into their curricula.",
        "Startup boards and investors should track leading indicators identified in this review.",
    ],
    venue_types=[
        "Journal of Business Venturing",
        "Strategic Management Journal",
        "Entrepreneurship Theory and Practice",
        "Small Business Economics",
        "Journal of Small Business Management",
    ],
    growth_indicators=[
        "user growth rate",
        "revenue growth rate",
        "team headcount scaling",
        "market penetration rate",
        "customer cohort retention",
    ],
)

ENTERPRISE_VOCAB = DomainVocabulary(
    name="enterprise",
    metrics=[
        "return on investment (ROI)",
        "total cost of ownership (TCO)",
        "net present value (NPV)",
        "earnings before interest and taxes (EBIT)",
        "total shareholder return (TSR)",
        "AI adoption rate",
        "model accuracy (F1/AUC)",
        "operational efficiency gain",
        "throughput improvement",
        "defect rate reduction",
        "employee productivity index",
        "technology maturity level (TML)",
        "data quality score",
        "model drift rate",
        "inference latency (p95/p99)",
    ],
    terminology=[
        "Fortune 500 enterprises",
        "large-scale organizational deployment",
        "MLOps (machine learning operations)",
        "AI governance framework",
        "responsible AI principles",
        "enterprise data architecture",
        "center of excellence (CoE)",
        "change management at scale",
        "digital transformation roadmap",
        "cloud-native AI infrastructure",
        "model risk management (MRM)",
        "algorithmic accountability",
        "federated learning deployment",
        "AI ethics board",
        "stakeholder alignment",
    ],
    context_phrases=[
        "In large-scale enterprise deployments where governance and risk management are paramount,",
        "For Fortune 500 organizations navigating responsible AI adoption,",
        "Within complex multi-stakeholder enterprise environments,",
        "Across diverse enterprise AI deployment contexts,",
        "In mature organizations with established data governance practices,",
    ],
    practical_phrases=[
        "Enterprise CIOs should establish AI centers of excellence with clear governance charters.",
        "MLOps maturity frameworks help organizations benchmark and improve deployment practices.",
        "Model risk management teams should integrate findings into validation protocols.",
        "Change management programs should address skills gaps identified in this review.",
        "AI governance boards can use these evidence-based metrics as KPIs for responsible deployment.",
    ],
    venue_types=[
        "MIS Quarterly",
        "Information Systems Research",
        "Journal of Management Information Systems",
        "Harvard Business Review",
        "MIT Sloan Management Review",
    ],
    growth_indicators=[
        "AI project portfolio size",
        "models in production",
        "AI-enabled revenue streams",
        "automation coverage rate",
        "data pipeline reliability",
    ],
)

DEFAULT_VOCAB = DomainVocabulary(
    name="general",
    metrics=["performance improvement", "efficiency gain", "adoption rate", "satisfaction score"],
    terminology=["organizations", "practitioners", "stakeholders", "knowledge workers"],
    context_phrases=["In modern organizational contexts,", "For practitioners navigating this space,"],
    practical_phrases=["Organizations should adopt evidence-based practices identified in this review."],
    venue_types=["Nature", "Science", "PNAS"],
    growth_indicators=["adoption rate", "impact score"],
)

DOMAIN_MAP: Dict[str, DomainVocabulary] = {
    "startup": STARTUP_VOCAB,
    "enterprise": ENTERPRISE_VOCAB,
}


def get_vocab(domain: str) -> DomainVocabulary:
    return DOMAIN_MAP.get(domain, DEFAULT_VOCAB)


def domain_metrics_sentence(domain: str, n: int = 3) -> str:
    """Return a sentence referencing key domain metrics."""
    vocab = get_vocab(domain)
    metrics = vocab.metrics[:n]
    if len(metrics) == 1:
        return f"Key performance indicators such as {metrics[0]} serve as primary outcome measures."
    elif len(metrics) == 2:
        return f"Performance metrics including {metrics[0]} and {metrics[1]} serve as primary outcome measures."
    else:
        return (
            f"Performance metrics including {', '.join(metrics[:-1])}, and {metrics[-1]} "
            f"serve as primary outcome measures in {domain} research."
        )


def domain_context_phrase(domain: str) -> str:
    vocab = get_vocab(domain)
    return vocab.context_phrases[0] if vocab.context_phrases else ""


def domain_practical_sentence(domain: str) -> str:
    vocab = get_vocab(domain)
    return vocab.practical_phrases[0] if vocab.practical_phrases else ""


def domain_terminology_list(domain: str, n: int = 5) -> List[str]:
    vocab = get_vocab(domain)
    return vocab.terminology[:n]
