"""Turn structural holes in the research graph into candidate research questions.

The graph says where the field has not looked. This turns each hole into a
question a supervisor could actually respond to, and — the part that matters —
attaches the evidence behind it: how many papers occupy the row, how many the
column, and which papers those are. A gap claim without that is a guess with
a citation style.

Nothing here invents a finding. A question is only generated where both sides
of the empty cell are independently well studied, and every question carries
the count that makes it checkable.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# How to phrase each kind of hole. The template is deliberately plain: the
# value is in the grounding, not in the wording.
_TEMPLATES = {
    "theory x population": (
        "How does {row} explain outcomes among {col}?",
        "{row} is an established lens ({row_n} papers here) and {col} is a well-studied "
        "population ({col_n} papers), but no paper in this corpus brings them together.",
    ),
    "theory x outcome": (
        "What does {row} predict about {col}?",
        "{row} appears in {row_n} papers and {col} is studied in {col_n}, yet the corpus "
        "contains no study applying the former to the latter.",
    ),
    "method x population": (
        "What would {row} reveal about {col} that current designs cannot?",
        "{row} is used in {row_n} papers and {col} is studied in {col_n}, but never "
        "together — the population is studied with other designs only.",
    ),
}


@dataclass
class GapQuestion:
    question: str
    rationale: str
    kind: str
    strength: int                    # min(row support, col support)
    row: str = ""
    col: str = ""
    evidence: Dict[str, int] = field(default_factory=dict)
    caveat: str = ""


def _corpus_caveat(n_papers: int) -> str:
    """State the limit of the claim. The graph knows one corpus, not a field."""
    return (
        f"Absence here means absence from the {n_papers} coded papers in this library, "
        f"not from the literature. Verify with a targeted search before claiming novelty."
    )


def questions_from_graph(graph: dict, limit: int = 12,
                         min_strength: int = 4) -> List[GapQuestion]:
    """Build candidate research questions from a research_graph report."""
    n_coded = graph.get("corpus", {}).get("n_coded_papers", 0)
    caveat = _corpus_caveat(n_coded)
    out: List[GapQuestion] = []

    for kind, items in (graph.get("gaps") or {}).items():
        for g in items:
            if kind == "mono_method_outcomes":
                out.append(GapQuestion(
                    question=(f"Would {g['outcome']} hold up under a design other than "
                              f"{g['dominant_method']}?"),
                    rationale=g["claim"],
                    kind=kind, strength=g.get("papers", 0),
                    row=g["dominant_method"], col=g["outcome"],
                    evidence={"papers": g.get("papers", 0)},
                    caveat=caveat,
                ))
                continue
            if kind == "stale_topics":
                out.append(GapQuestion(
                    question=(f"Has the evidence on {g['outcome']} changed since "
                              f"{g['latest_year']}?"),
                    rationale=g["claim"],
                    kind=kind, strength=g.get("papers", 0),
                    col=g["outcome"],
                    evidence={"papers": g.get("papers", 0),
                              "latest_year": g.get("latest_year", 0)},
                    caveat=caveat,
                ))
                continue

            tmpl = _TEMPLATES.get(g.get("type", ""))
            if not tmpl or g.get("strength", 0) < min_strength:
                continue
            q_t, r_t = tmpl
            fields = {"row": g["row"], "col": g["col"],
                      "row_n": g["row_papers"], "col_n": g["col_papers"]}
            out.append(GapQuestion(
                question=q_t.format(**fields),
                rationale=r_t.format(**fields),
                kind=g["type"], strength=g["strength"],
                row=g["row"], col=g["col"],
                evidence={"row_papers": g["row_papers"], "col_papers": g["col_papers"]},
                caveat=caveat,
            ))

    out.sort(key=lambda q: -q.strength)
    return out[:limit]


def load_and_generate(graph_path: str = "library/research_graph.json",
                      limit: int = 12) -> List[GapQuestion]:
    p = Path(graph_path)
    if not p.exists():
        logger.info(f"[GapFinder] no research graph at {graph_path}")
        return []
    return questions_from_graph(json.loads(p.read_text()), limit=limit)


def render(questions: List[GapQuestion]) -> str:
    """Human-readable brief, evidence first."""
    if not questions:
        return ("No gap survives the evidence test on this corpus. That is a result, "
                "not a failure — harvest more papers and re-run.")
    lines = ["CANDIDATE RESEARCH QUESTIONS (from structural holes in the coded corpus)\n"]
    for i, q in enumerate(questions, 1):
        ev = ", ".join(f"{k}={v}" for k, v in q.evidence.items())
        lines.append(f"{i}. {q.question}")
        lines.append(f"   why: {q.rationale}")
        lines.append(f"   evidence: {ev} | kind: {q.kind}")
        lines.append("")
    lines.append(questions[0].caveat)
    return "\n".join(lines)
