"""Extract structured text and statistics from academic paper full text."""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Section header patterns (ordered by priority)
_SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r'(?:^|\n)#{1,3}\s*(?:1[.\s]+)?introduction\b', "introduction"),
    (r'(?:^|\n)#{1,3}\s*(?:2[.\s]+)?(?:related work|background|literature review)\b', "background"),
    (r'(?:^|\n)#{1,3}\s*(?:3[.\s]+)?(?:method(?:s|ology)?|approach|design|framework)\b', "methods"),
    (r'(?:^|\n)#{1,3}\s*(?:4[.\s]+)?(?:result(?:s)?|finding(?:s)?|analysis|experiment(?:s)?)\b', "results"),
    (r'(?:^|\n)#{1,3}\s*(?:5[.\s]+)?(?:discussion|implication(?:s)?|interpretation)\b', "discussion"),
    (r'(?:^|\n)#{1,3}\s*(?:6[.\s]+)?conclusion(?:s)?\b', "conclusion"),
]

# Plain-text section headers (for PDFs and plain text)
_PLAIN_SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r'(?:^|\n)\s*(?:1[.\s]+)?INTRODUCTION\s*\n', "introduction"),
    (r'(?:^|\n)\s*(?:2[.\s]+)?(?:RELATED WORK|BACKGROUND|LITERATURE REVIEW)\s*\n', "background"),
    (r'(?:^|\n)\s*(?:3[.\s]+)?(?:METHOD(?:S|OLOGY)?|APPROACH|DESIGN|FRAMEWORK)\s*\n', "methods"),
    (r'(?:^|\n)\s*(?:4[.\s]+)?(?:RESULT(?:S)?|FINDING(?:S)?|ANALYSIS|EXPERIMENT(?:S)?)\s*\n', "results"),
    (r'(?:^|\n)\s*(?:5[.\s]+)?(?:DISCUSSION|IMPLICATION(?:S)?)\s*\n', "discussion"),
    (r'(?:^|\n)\s*(?:6[.\s]+)?CONCLUSION(?:S)?\s*\n', "conclusion"),
]

# Quantitative stat patterns
_STAT_PATTERNS = [
    r'\b\d+\.?\d*\s*%',                       # percentages: 23%, 3.5%
    r'\b[Nn]\s*[=≈]\s*\d[\d,]+',             # sample sizes: N=523, n=1,200
    r'\bp\s*[<>=≤≥]\s*0\.\d+',               # p-values: p<0.001, p=0.05
    r'\br\s*=\s*[-−]?0\.\d+',                # correlations: r=0.64
    r'\bR²?\s*=\s*0\.\d+',                   # R-squared: R²=0.34
    r'\bβ\s*=\s*[-−]?\d+\.?\d*',             # beta coefficients: β=0.23
    r'\bOR\s*=\s*\d+\.?\d*',                 # odds ratios: OR=2.3
    r'\bHR\s*=\s*\d+\.?\d*',                 # hazard ratios: HR=1.5
    r'\b(?:Cohen\'?s?\s+)?[df]\s*=\s*\d+\.?\d*',  # Cohen's d/f: d=0.8
    r'\b\d+\s+(?:firm|company|startup|entrepreneur|manager|founder)s?\b',  # entity counts
]


class FullTextExtractor:
    """Extract structured sections and statistics from academic paper text."""

    def extract_sections(self, full_text: str) -> Dict[str, str]:
        """
        Split full text into named sections.

        Returns dict with keys from _SECTION_PATTERNS (introduction, methods, results, etc.).
        Falls back to splitting text in thirds if no headers found.
        """
        if not full_text or len(full_text) < 200:
            return {}

        sections: Dict[str, str] = {}
        combined_patterns = _SECTION_PATTERNS + _PLAIN_SECTION_PATTERNS

        # Find all section boundaries
        boundaries: List[Tuple[int, str]] = []
        for pattern, section_name in combined_patterns:
            for m in re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE):
                boundaries.append((m.end(), section_name))

        if not boundaries:
            # Fallback: split into thirds (intro / body / conclusion)
            third = len(full_text) // 3
            sections["introduction"] = full_text[:third]
            sections["methods"] = full_text[third:2*third]
            sections["results"] = full_text[2*third:]
            return sections

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        # Extract text between consecutive section headers
        for i, (start_pos, name) in enumerate(boundaries):
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            section_text = full_text[start_pos:end_pos].strip()
            if section_text and name not in sections:  # First match wins
                sections[name] = section_text

        return sections

    def extract_key_sentences(
        self,
        text: str,
        keywords: List[str],
        n: int = 5,
    ) -> List[str]:
        """
        Return the top-n sentences from text most relevant to given keywords.

        Scoring: count of keyword matches per sentence (TF-style, no IDF needed for short text).
        """
        if not text or not keywords:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        if not sentences:
            return []

        kw_lower = [kw.lower() for kw in keywords]

        scored: List[Tuple[float, str]] = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in kw_lower if kw in sent_lower)
            # Bonus for sentences containing statistics
            if re.search(r'\b\d+\.?\d*\s*%|\b[Nn]\s*=|\bp\s*[<>]', sent):
                score += 0.5
            if score > 0:
                scored.append((score, sent))

        # Sort by score descending, then by position (earlier is better for ties)
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:n]]

    def extract_statistics(self, text: str) -> List[str]:
        """
        Extract quantitative statistics with surrounding context.

        Returns list of short strings like: "N=523 startups over 3 years"
        """
        if not text:
            return []

        stats: List[str] = []
        seen: set = set()

        for pattern in _STAT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 60)
                snippet = text[start:end].strip()
                # Trim to sentence boundaries if possible
                snippet = re.sub(r'^[^A-Z0-9]*', '', snippet)
                snippet = re.split(r'(?<=[.!?])\s+', snippet)[0]
                snippet = snippet.strip()
                key = re.sub(r'\s+', ' ', snippet).lower()[:80]
                if snippet and key not in seen and len(snippet) > 10:
                    seen.add(key)
                    stats.append(snippet)

        return stats[:15]  # Cap to avoid overwhelming output

    def clean_html(self, html: str) -> str:
        """Strip HTML tags and clean whitespace for plain-text extraction."""
        # Remove script/style blocks
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
                   .replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()
