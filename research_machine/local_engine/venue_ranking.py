"""Venue reputation ranking for paper selection — implements Tier 1/2/3 system."""

from typing import Dict
import math

# Tier 1: Highest reputation venues (boost +0.15)
TIER_1_VENUES = {
    # Machine Learning (Top conferences)
    "ICML": "International Conference on Machine Learning",
    "NeurIPS": "Neural Information Processing Systems",
    "ICLR": "International Conference on Learning Representations",

    # ML Journals
    "JMLR": "Journal of Machine Learning Research",
    "IEEE TPAMI": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "Machine Learning": "Springer Machine Learning Journal",

    # Business/Entrepreneurship
    "Journal of Business Venturing": "JBV",
    "Strategic Management Journal": "SMJ",
    "Management Science": "MS",
    "Entrepreneurship Theory and Practice": "ETP",
    "Small Business Economics": "SBE",

    # AI/NLP
    "ACL": "Association for Computational Linguistics",
    "EMNLP": "Empirical Methods in Natural Language Processing",
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "IJCAI": "International Joint Conference on AI",
    "ACM Transactions on Information Systems": "TOIS",

    # Systems/Enterprise
    "IEEE Transactions on Software Engineering": "TSE",
    "OSDI": "Operating Systems Design and Implementation",
    "SIGMOD": "Special Interest Group on Management of Data",
}

# Tier 2: Medium reputation venues (boost +0.08)
TIER_2_VENUES = {
    # ML/AI conferences
    "ICLR Workshop": "ICLR Workshop",
    "CVPR": "Computer Vision and Pattern Recognition",
    "ICCV": "International Conference on Computer Vision",
    "ECCV": "European Conference on Computer Vision",
    "KDD": "Knowledge Discovery and Data Mining",
    "CSCW": "Computer-Supported Cooperative Work",
    "CHI": "ACM Conference on Human Factors in Computing",
    "FAT": "Fairness, Accountability, and Transparency",

    # Business/Management journals
    "Journal of Management": "JOM",
    "Organization Science": "OS",
    "Harvard Business Review": "HBR",
    "MIT Sloan Management Review": "MSMR",
    "Journal of Management Information Systems": "JMIS",
    "MIS Quarterly": "MISQ",
    "Information Systems Research": "ISR",

    # Economics
    "Journal of Economic Literature": "JEL",
    "Review of Economic Studies": "RES",
    "Journal of Finance": "JF",
    "Journal of Financial Economics": "JFE",

    # NLP journals
    "Computational Linguistics": "CL",
    "Transactions of the Association for Computational Linguistics": "TACL",
}

# Tier 3: Lower reputation (boost +0.03)
TIER_3_VENUES = {
    "arXiv": "arXiv Preprint",
    "IEEE": "IEEE (generic)",
    "ACM": "ACM (generic)",
    "Workshop": "Workshop",
    "Preprint": "Preprint",
    "Technical Report": "Technical Report",
}

def get_venue_tier(venue: str) -> tuple[int, float]:
    """
    Determine venue tier and reputation boost.

    Returns:
        (tier: int 1-3, boost: float) where boost is added to relevance score
    """
    if not venue:
        return (3, 0.03)

    venue_lower = venue.lower()

    # Check Tier 1
    for tier1_venue in TIER_1_VENUES:
        if tier1_venue.lower() in venue_lower or venue_lower in tier1_venue.lower():
            return (1, 0.15)

    # Check Tier 2
    for tier2_venue in TIER_2_VENUES:
        if tier2_venue.lower() in venue_lower or venue_lower in tier2_venue.lower():
            return (2, 0.08)

    # Check for known preprint/workshop patterns
    if any(pattern in venue_lower for pattern in ["arxiv", "preprint", "workshop", "technical report"]):
        return (3, 0.03)

    # Default: unknown venue = lowest tier
    return (3, 0.03)


def author_h_index_boost(h_index: int) -> float:
    """
    Convert author h-index to reputation boost (0.0-0.10).

    H-index scale:
    - 0-5: New researcher → 0.00
    - 5-10: Early career → 0.03
    - 10-20: Established → 0.06
    - 20-50: Senior → 0.09
    - 50+: Highly cited → 0.10
    """
    if h_index is None or h_index < 0:
        return 0.00
    if h_index < 5:
        return 0.00
    elif h_index < 10:
        return 0.03
    elif h_index < 20:
        return 0.06
    elif h_index < 50:
        return 0.09
    else:
        return 0.10


def recency_bonus(year: int, current_year: int = 2026) -> float:
    """
    Time-based relevance boost with gradual decay.

    - 0-1 years old: +0.15 (very recent)
    - 1-3 years old: +0.12
    - 3-5 years old: +0.08
    - 5-7 years old: +0.03
    - 7+ years old: 0.00 (foundational, no bonus)
    """
    age = current_year - year
    if age < 0:
        return 0.00
    elif age <= 1:
        return 0.15
    elif age <= 3:
        return 0.12
    elif age <= 5:
        return 0.08
    elif age <= 7:
        return 0.03
    else:
        return 0.00


def calculate_reputation_score(
    keywords_match: float,
    citation_count: int,
    venue: str,
    year: int,
    author_h_index: int = None,
    current_year: int = 2026,
) -> float:
    """
    Advanced reputation scoring formula.

    Args:
        keywords_match: 0.0-1.0 (fraction of keywords in abstract)
        citation_count: number of citations
        venue: publication venue name
        year: publication year
        author_h_index: lead author's h-index
        current_year: current year for recency calculation

    Returns:
        Reputation score 0.0-1.0 (higher is better)
    """
    # Normalize citation count to 0-1 scale with logarithmic scaling
    # Papers with 1 cite: 0.1, 10 cites: 0.3, 100 cites: 0.5, 1000+ cites: 0.7
    citation_boost = min(0.20, (math.log(max(1, citation_count) + 1) / 10.0) * 0.20)

    # Get venue boost
    venue_tier, venue_boost = get_venue_tier(venue)

    # Get recency boost
    recency = recency_bonus(year, current_year)

    # Get author h-index boost
    h_boost = author_h_index_boost(author_h_index) if author_h_index else 0.0

    # Weighted combination
    score = (
        0.25 * keywords_match +           # Keyword relevance
        citation_boost +                   # Citation count (0-0.20)
        0.20 * venue_boost +               # Venue reputation (0-0.20)
        0.15 * recency +                   # Recency (0-0.15)
        0.10 * h_boost                     # Author h-index (0-0.10)
    )

    return min(1.0, score)  # Cap at 1.0


def should_include_paper(
    tier: int,
    citation_count: int,
    score: float,
    venue: str = None,
) -> bool:
    """
    Quality gate: determine if paper should be included in corpus.

    Rules:
    - Tier 1 venue → always include (trusted source)
    - Tier 2 venue + 3+ citations → include
    - Tier 3 + 10+ citations → include (well-established preprint)
    - Score < 0.30 → reject (low relevance)
    """
    if score < 0.30:
        return False

    if tier == 1:
        return True
    elif tier == 2:
        return citation_count >= 3
    else:  # tier 3
        return citation_count >= 10
