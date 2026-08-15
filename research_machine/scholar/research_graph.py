"""Research graph — what the field has studied, and where the holes are.

Each paper is reduced to the four coordinates that define an empirical study:
THEORY it builds on, METHOD it uses, POPULATION it studies, and OUTCOME it
explains. Crossing those coordinates turns a reading list into a matrix, and
the empty or thin cells of that matrix are what a supervisor means by "find a
gap".

Detection is lexicon- plus pattern-based, which is crude — so every extracted
coordinate keeps the sentence and DOI it came from. A gap claim that cannot
be traced back to the cells it rests on is an opinion, and this module refuses
to emit one.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pdf_parser import ParsedPaper, split_sentences

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Lexicons. Deliberately conservative: a term must be specific enough that a
# match means the paper really uses it, not merely mentions it in passing.
# --------------------------------------------------------------------------

THEORIES = {
    "Resource-Based View": r"resource[- ]based (?:view|theory)|\bRBV\b",
    "Dynamic Capabilities": r"dynamic capabilit",
    "Institutional Theory": r"institutional theory|institutional logics|isomorphism",
    "Effectuation": r"\beffectuation\b|effectual (?:logic|reasoning)",
    "Bricolage": r"entrepreneurial bricolage|\bbricolage\b",
    "Technology Acceptance Model": r"technology acceptance model|\bTAM\b|perceived usefulness",
    "UTAUT": r"\bUTAUT\b|unified theory of acceptance",
    "Knowledge-Based View": r"knowledge[- ]based (?:view|theory)|\bKBV\b",
    "Absorptive Capacity": r"absorptive capacity",
    "Social Capital Theory": r"social capital (?:theory|perspective)",
    "Human Capital Theory": r"human capital (?:theory|perspective)",
    "Upper Echelons": r"upper echelons",
    "Agency Theory": r"agency theory|principal[- ]agent",
    "Transaction Cost Economics": r"transaction cost",
    "Signalling Theory": r"signal(?:l)?ing theory",
    "External Enabler Framework": r"external enabler",
    "Opportunity Recognition": r"opportunity (?:recognition|identification|discovery)",
    "Attention-Based View": r"attention[- ]based view",
    "Organizational Learning": r"organi[sz]ational learning|exploration and exploitation|ambidexterity",
    "Diffusion of Innovation": r"diffusion of innovation|rogers.{0,15}diffusion",
    "Planned Behaviour": r"theory of planned behavio",
    "Self-Efficacy": r"self[- ]efficacy",
    "Legitimacy Theory": r"\blegitimacy\b",
    "Real Options": r"real options",
}

METHODS = {
    "regression": r"\b(?:OLS|logit|probit|tobit|negative binomial|regression analys|regression model)",
    "panel data": r"panel (?:data|regression)|fixed[- ]effects|random[- ]effects",
    "causal-identification": r"difference[- ]in[- ]differences|\bDiD\b|instrumental variable|regression discontinuity"
                             r"|propensity score|natural experiment",
    "experiment": r"randomi[sz]ed (?:controlled )?(?:experiment|trial)|conjoint (?:experiment|analysis)"
                  r"|vignette experiment|lab experiment|field experiment",
    "survey": r"survey (?:of|data|instrument|respondents)|questionnaire|Likert",
    "SEM/PLS": r"structural equation model|\bSEM\b|\bPLS-SEM\b|confirmatory factor analys",
    "case study": r"case study|multiple[- ]case|single[- ]case|comparative case",
    "interviews": r"semi[- ]structured interview|in[- ]depth interview|\binterviewees?\b",
    "qualitative coding": r"grounded theory|open coding|axial coding|thematic analys|Gioia method",
    "content/text analysis": r"content analysis|text mining|topic model|\bLDA\b|natural language processing"
                             r"|sentiment analysis|dictionary[- ]based",
    "machine learning": r"machine learning|random forest|neural network|supervised learning|\bXGBoost\b",
    "meta-analysis": r"meta[- ]analy(?:sis|tic)",
    "systematic review": r"systematic (?:literature )?review|\bPRISMA\b",
    "bibliometric": r"bibliometric|co[- ]citation analysis|co[- ]word|VOSviewer|Bibliometrix",
    "simulation": r"agent[- ]based model|monte carlo simulation|computational model",
    "configurational/fsQCA": r"\bfsQCA\b|qualitative comparative analysis|configurational",
}

POPULATIONS = {
    "startups/new ventures": r"\b(?:start[- ]?ups?|new ventures?|nascent (?:venture|firm|entrepreneur))",
    "founders/entrepreneurs": r"\b(?:founders?|entrepreneurs?|founding team)",
    "SMEs": r"\bSMEs?\b|small (?:and medium|business(?:es)?)|micro[- ]enterprise",
    "large firms": r"large (?:firms?|corporations?)|incumbent firms?|multinational",
    "students": r"\b(?:students?|undergraduate|university sample|classroom)",
    "employees/managers": r"\b(?:employees?|managers?|workers?|middle management)",
    "investors/VCs": r"venture capital|\bVCs?\b|business angel|investors?",
    "platforms/gig": r"platform (?:workers?|economy)|gig (?:economy|workers?)|crowdfunding",
    "emerging economies": r"emerging (?:economies|markets)|developing (?:countries|economies)"
                          r"|Global South|low[- ]income countries",
}

OUTCOMES = {
    "performance/growth": r"\b(?:firm performance|venture performance|sales growth|employment growth|profitability)",
    "survival/failure": r"\b(?:survival|failure rate|exit|bankruptcy|discontinuance)",
    "innovation output": r"\b(?:innovation performance|patent|new product|R&D output|radical innovation)",
    "funding/finance": r"\b(?:funding|fundrais|external finance|access to (?:capital|credit)|valuation)",
    "decision-making": r"\b(?:decision[- ]making|judgment|choice|decision quality|decision speed)",
    "productivity/efficiency": r"\b(?:productivity|efficiency|time savings|task completion)",
    "opportunity/idea generation": r"\b(?:opportunity (?:recognition|exploitation)|idea generation|creativity)",
    "intention/adoption": r"\b(?:entrepreneurial intention|adoption intention|usage intention|acceptance)",
    "internationalisation": r"\b(?:internationali[sz]ation|export|foreign market entry)",
    "sustainability/impact": r"\b(?:sustainab|social impact|ESG|circular economy)",
}

_DIMENSIONS = {
    "theory": THEORIES, "method": METHODS,
    "population": POPULATIONS, "outcome": OUTCOMES,
}
_COMPILED = {
    dim: {label: re.compile(pat, re.IGNORECASE) for label, pat in lex.items()}
    for dim, lex in _DIMENSIONS.items()
}

# A term must clear this many mentions to count as "used" rather than "named".
_MIN_MENTIONS = {"theory": 3, "method": 2, "population": 3, "outcome": 3}


@dataclass
class PaperCoding:
    doi: str
    year: Optional[int] = None
    genre: str = ""
    theories: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    populations: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)   # "theory:RBV" -> sentence


def code_paper(paper: ParsedPaper, year: Optional[int] = None) -> PaperCoding:
    """Reduce one paper to its research coordinates, keeping the evidence."""
    coding = PaperCoding(doi=paper.doi, year=year, genre=paper.genre)
    # Method claims live in Methods/Abstract; theory in the front half; the
    # rest is read whole. Searching everything for everything invites the
    # literature review's mentions of other people's work to be coded as ours.
    scopes = {
        "theory": " ".join(paper.sections.get(s, "") for s in
                           ("abstract", "introduction", "literature", "discussion")),
        "method": " ".join(paper.sections.get(s, "") for s in
                           ("abstract", "methods", "results")),
        "population": " ".join(paper.sections.get(s, "") for s in
                               ("abstract", "introduction", "methods", "results")),
        "outcome": " ".join(paper.sections.get(s, "") for s in
                            ("abstract", "introduction", "results", "discussion")),
    }
    for dim, patterns in _COMPILED.items():
        text = scopes[dim]
        if not text:
            continue
        found = []
        for label, pat in patterns.items():
            hits = pat.findall(text)
            if len(hits) >= _MIN_MENTIONS[dim]:
                found.append(label)
                for s in split_sentences(text):
                    if pat.search(s) and 40 < len(s) < 320:
                        coding.evidence[f"{dim}:{label}"] = re.sub(r"\s+", " ", s).strip()
                        break
        setattr(coding, {"theory": "theories", "method": "methods",
                         "population": "populations", "outcome": "outcomes"}[dim], found)
    return coding


class ResearchGraph:
    """The field as a matrix of theory x method x population x outcome."""

    def __init__(self, codings: List[PaperCoding]):
        self.codings = [c for c in codings if c.populations or c.outcomes]
        self.n = len(self.codings)

    # ------------------------------------------------------------ inventory
    def counts(self, dim: str) -> Counter:
        attr = {"theory": "theories", "method": "methods",
                "population": "populations", "outcome": "outcomes"}[dim]
        c: Counter = Counter()
        for coding in self.codings:
            c.update(getattr(coding, attr))
        return c

    def cell(self, dim_a: str, val_a: str, dim_b: str, val_b: str) -> List[str]:
        """DOIs sitting in one cell of a two-dimensional crossing."""
        attr = {"theory": "theories", "method": "methods",
                "population": "populations", "outcome": "outcomes"}
        return [c.doi for c in self.codings
                if val_a in getattr(c, attr[dim_a]) and val_b in getattr(c, attr[dim_b])]

    def matrix(self, dim_a: str, dim_b: str, min_support: int = 2) -> dict:
        """Crossing of two dimensions, restricted to terms the corpus actually
        supports — a matrix full of one-paper rows produces fake gaps."""
        rows = [t for t, n in self.counts(dim_a).items() if n >= min_support]
        cols = [t for t, n in self.counts(dim_b).items() if n >= min_support]
        return {
            "rows": rows, "cols": cols,
            "cells": {f"{a}||{b}": self.cell(dim_a, a, dim_b, b)
                      for a, b in product(rows, cols)},
        }

    # ----------------------------------------------------------------- gaps
    def structural_gaps(self, dim_a: str, dim_b: str,
                        min_support: int = 3, max_gaps: int = 15) -> List[dict]:
        """Empty cells whose ROW and COLUMN are both well populated.

        An empty cell between two rare terms means nothing — the corpus is
        simply thin there. An empty cell where both the theory and the
        population are otherwise well studied is the real signal: the field
        had every reason to make this connection and did not.
        """
        m = self.matrix(dim_a, dim_b, min_support=min_support)
        row_tot = {r: sum(len(m["cells"][f"{r}||{c}"]) for c in m["cols"]) for r in m["rows"]}
        col_tot = {c: sum(len(m["cells"][f"{r}||{c}"]) for r in m["rows"]) for c in m["cols"]}
        gaps = []
        for r, c in product(m["rows"], m["cols"]):
            papers = m["cells"][f"{r}||{c}"]
            if len(papers) > 0:
                continue
            if row_tot[r] < min_support or col_tot[c] < min_support:
                continue
            gaps.append({
                "type": f"{dim_a} x {dim_b}",
                "row": r, "col": c,
                "row_papers": row_tot[r], "col_papers": col_tot[c],
                "strength": min(row_tot[r], col_tot[c]),
                "claim": (f"No paper in this corpus applies {r} to {c}, although "
                          f"{r} appears in {row_tot[r]} papers and {c} in {col_tot[c]}."),
            })
        gaps.sort(key=lambda g: -g["strength"])
        return gaps[:max_gaps]

    def mono_method_outcomes(self, min_papers: int = 3) -> List[dict]:
        """Outcomes the field only ever measures one way — a method gap."""
        by_outcome: Dict[str, Counter] = defaultdict(Counter)
        papers_per_outcome: Counter = Counter()
        for c in self.codings:
            for o in c.outcomes:
                papers_per_outcome[o] += 1
                by_outcome[o].update(c.methods)
        out = []
        for outcome, methods in by_outcome.items():
            if papers_per_outcome[outcome] < min_papers or not methods:
                continue
            total = sum(methods.values())
            top, top_n = methods.most_common(1)[0]
            share = top_n / total
            if share >= 0.7 and len(methods) <= 3:
                out.append({
                    "outcome": outcome, "dominant_method": top,
                    "share": round(share, 2), "papers": papers_per_outcome[outcome],
                    "claim": (f"{outcome} is studied almost only through {top} "
                              f"({int(share*100)}% of method mentions across "
                              f"{papers_per_outcome[outcome]} papers) — triangulation is missing."),
                })
        out.sort(key=lambda d: -d["papers"])
        return out

    def stale_topics(self, cutoff_year: int = 2021, min_papers: int = 3) -> List[dict]:
        """Cells whose entire evidence base predates a cutoff."""
        by_outcome: Dict[str, List[int]] = defaultdict(list)
        for c in self.codings:
            if c.year:
                for o in c.outcomes:
                    by_outcome[o].append(c.year)
        out = []
        for outcome, years in by_outcome.items():
            if len(years) >= min_papers and max(years) < cutoff_year:
                out.append({
                    "outcome": outcome, "papers": len(years), "latest_year": max(years),
                    "claim": (f"Evidence on {outcome} in this corpus stops at "
                              f"{max(years)} ({len(years)} papers) — nothing since."),
                })
        return sorted(out, key=lambda d: -d["papers"])

    def report(self) -> dict:
        return {
            "corpus": {
                "n_coded_papers": self.n,
                "theories": self.counts("theory").most_common(),
                "methods": self.counts("method").most_common(),
                "populations": self.counts("population").most_common(),
                "outcomes": self.counts("outcome").most_common(),
            },
            "gaps": {
                "theory_x_population": self.structural_gaps("theory", "population"),
                "theory_x_outcome": self.structural_gaps("theory", "outcome"),
                "method_x_population": self.structural_gaps("method", "population"),
                "mono_method_outcomes": self.mono_method_outcomes(),
                "stale_topics": self.stale_topics(),
            },
            "coding": [
                {"doi": c.doi, "year": c.year, "genre": c.genre,
                 "theories": c.theories, "methods": c.methods,
                 "populations": c.populations, "outcomes": c.outcomes,
                 "evidence": c.evidence}
                for c in self.codings
            ],
        }


def build_from_library(library_root: str, out_path: str,
                       max_papers: Optional[int] = None) -> dict:
    """Parse a harvested library, code every paper, and write the graph report."""
    from .pdf_parser import parse_pdf
    root = Path(library_root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [e for e in manifest["entries"] if e.get("pdf_path")]
    if max_papers:
        entries = entries[:max_papers]

    codings = []
    for e in entries:
        parsed = parse_pdf(str(root / e["pdf_path"]), e["doi"])
        if parsed.ok:
            codings.append(code_paper(parsed, year=e.get("year")))
    logger.info(f"[Graph] coded {len(codings)}/{len(entries)} papers")

    report = ResearchGraph(codings).report()
    Path(out_path).write_text(json.dumps(report, indent=1))
    logger.info(f"[Graph] research graph -> {out_path}")
    return report
