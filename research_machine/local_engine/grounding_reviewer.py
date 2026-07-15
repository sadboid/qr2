"""Grounding reviewer: a dedicated review pass over the generated manuscript.

Modeled on the "dedicated reviewer agent" pattern (cf. Claude Science): after
the paper is written, a separate inspector checks things the writing pass
cannot be trusted to self-report:

1. POPULATION FIDELITY — an interpretive claim framed around the paper's focal
   population (founders/startups) must not be grounded in a source whose
   abstract is actually about students, patients, judges, or tourists.
   This is the confabulation class found in the v6 audit (education-quality
   papers dressed up as founder-decision evidence).
2. PHANTOM CITATIONS — every in-text citation must resolve to a corpus paper
   or a References entry. (v6 cited "Zhou and Cen (2024)" which existed
   nowhere.)
3. THEORY CONSISTENCY — the theory the Discussion claims to extend must be the
   one the Theoretical Framework section adopted. (v6 adopted TAM, then
   "extended" Dynamic Capabilities.)
4. NUMBER CONSISTENCY — stated evidence-split counts must match the actual
   consensus analysis. (v6 stated 15+9+10 "of 45" — sums to 34.)

High-severity issues feed a self-correction rewrite of the offending section
(see writer.revise_section_with_claude), after which the reviewer runs again.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .corpus import Paper

logger = logging.getLogger(__name__)

# Focal-population vocabulary per engine domain
_FOCAL_TERMS = {
    "startup": frozenset({
        "founder", "entrepreneur", "entrepreneurial", "startup", "start-up",
        "venture", "sme", "small business", "self-employ", "business owner",
        "new firm", "business model",
    }),
    "enterprise": frozenset({
        "enterprise", "organization", "organisation", "firm", "manager",
        "corporate", "company", "business",
    }),
}

# Off-population lexicons: if a cited source's abstract matches one of these
# and shows NO focal terms, it cannot ground a focal-population claim.
_OFF_POPULATIONS: Dict[str, frozenset] = {
    "education/students": frozenset({
        "student", "higher education", "classroom", "teaching", "curriculum",
        "learner", "pedagog", "academic integrity", "plagiarism", "course",
        "university teaching", "educators",
    }),
    "clinical/patients": frozenset({
        "patient", "clinical", "hospital", "surgery", "surgical", "physician",
        "nurse", "diagnosis", "medical decision", "healthcare professional",
        "triage", "oncology", "glioblastoma",
    }),
    "legal/judicial": frozenset({
        "judge", "courtroom", "forensic", "criminal justice", "law enforcement",
    }),
    "consumers/tourists": frozenset({
        "tourist", "tourism", "consumer decision", "influencer", "shopper",
    }),
    "employees/HR": frozenset({
        "employee", "human resource", "recruitment", "hiring", "workforce",
    }),
}


@dataclass
class GroundingIssue:
    section: str          # "Discussion" | "Results" | "Theoretical Framework" | "paper"
    issue_type: str       # population_mismatch | phantom_citation | theory_inconsistency | number_inconsistency
    citation_ref: str     # "[Author, Year]" or ""
    claim_excerpt: str    # first ~160 chars of the offending sentence
    detail: str           # human-readable explanation with the fix direction
    severity: str         # "high" | "medium"


@dataclass
class GroundingReport:
    issues: List[GroundingIssue] = field(default_factory=list)
    checked_claims: int = 0
    revised: bool = False  # set by the engine after a self-correction pass

    @property
    def high_issues(self) -> List[GroundingIssue]:
        return [i for i in self.issues if i.severity == "high"]

    @property
    def passed(self) -> bool:
        return not self.high_issues

    @property
    def score(self) -> float:
        """0-10; each high issue costs 2 points, each medium 0.5."""
        penalty = 2.0 * len(self.high_issues) + 0.5 * (len(self.issues) - len(self.high_issues))
        return round(max(0.0, 10.0 - penalty), 1)

    def feedback(self) -> str:
        if not self.issues:
            return f"{self.checked_claims} interpretive claims checked — no grounding issues."
        kinds = {}
        for i in self.issues:
            kinds[i.issue_type] = kinds.get(i.issue_type, 0) + 1
        kind_str = ", ".join(f"{v} {k}" for k, v in kinds.items())
        return (
            f"{len(self.issues)} issues ({len(self.high_issues)} high) over "
            f"{self.checked_claims} checked claims: {kind_str}."
            + (" Self-correction applied." if self.revised else "")
        )


# --- citation extraction (same three styles the claim checker recognises) ---

_CITE_PATTERNS = [
    (re.compile(r'\[([A-Za-z][A-Za-z\s]+?),?\s*(\d{4})\]'), "bracket"),
    (re.compile(r'\(([A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][A-Za-z\-]+))?),\s*(\d{4})[a-z]?\)'), "paren"),
    (re.compile(r'\b([A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][A-Za-z\-]+))?)\s+\((\d{4})[a-z]?\)'), "narrative"),
]


def _first_author_key(raw_author: str) -> str:
    first = re.split(r',|&|\band\b', raw_author)[0]
    first = first.replace(' et al.', '').replace(' et al', '').strip()
    return first.split()[-1].lower() if first.split() else ""


def _extract_cites_with_sentences(text: str) -> List[Tuple[str, int, str]]:
    """Return [(author_key, year, containing_sentence), ...] deduplicated."""
    out = []
    seen = set()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z—*])', text)
    for sent in sentences:
        for pattern, _style in _CITE_PATTERNS:
            for m in pattern.finditer(sent):
                key = (_first_author_key(m.group(1)), int(m.group(2)))
                if key[0] and (key, sent[:60]) not in seen:
                    seen.add((key, sent[:60]))
                    out.append((key[0], key[1], sent.strip()))
    return out


def _split_sections(paper_md: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for block in re.split(r'\n(?=## )', paper_md):
        header = block.split("\n", 1)[0].replace("#", "").strip()
        sections[header] = block
    return sections


class GroundingReviewer:
    def __init__(self, domain: str = "startup"):
        self.domain = domain
        self.focal_terms = _FOCAL_TERMS.get(domain, _FOCAL_TERMS["startup"])

    # ---------------------------------------------------------------- review
    def review(
        self,
        paper_md: str,
        corpus: List[Paper],
        primary_theory: str,
        consensus=None,
    ) -> GroundingReport:
        report = GroundingReport()
        sections = _split_sections(paper_md)

        corpus_lookup: Dict[Tuple[str, int], Paper] = {}
        for p in corpus:
            if p.authors:
                last = p.authors[0].split(",")[0].strip().split()[-1].lower()
                corpus_lookup[(last, p.year)] = p

        references_text = sections.get("References", "")

        # 1 + 2: population fidelity & phantom citations on interpretive sections
        for sec_name in ("Discussion", "Results"):
            body = sections.get(sec_name, "")
            if not body:
                continue
            cites = _extract_cites_with_sentences(body)
            report.checked_claims += len(cites)
            for author_key, year, sentence in cites:
                paper = corpus_lookup.get((author_key, year))
                if paper is None:
                    # not in corpus — is it at least in the references list?
                    in_refs = re.search(
                        rf'\b{re.escape(author_key)}\b.*?\({year}\)',
                        references_text, re.IGNORECASE,
                    )
                    if not in_refs:
                        report.issues.append(GroundingIssue(
                            section=sec_name,
                            issue_type="phantom_citation",
                            citation_ref=f"[{author_key.title()}, {year}]",
                            claim_excerpt=sentence[:160],
                            detail=(
                                f"'{author_key.title()} ({year})' is cited but exists neither in "
                                f"the corpus nor in the References — remove it or replace with a "
                                f"verifiable corpus citation."
                            ),
                            severity="high",
                        ))
                    continue
                # population fidelity
                issue = self._population_check(sec_name, sentence, paper)
                if issue:
                    report.issues.append(issue)

        # 3: theory consistency
        report.issues.extend(self._theory_check(sections, primary_theory))

        # 4: number consistency of the evidence split
        if consensus is not None:
            report.issues.extend(self._numbers_check(sections.get("Discussion", ""), consensus))

        return report

    # Pedagogy collocations that make an education-population paper LOOK focal
    # ("entrepreneurship education" contains "entrepreneur" but studies students).
    _EDU_COLLOCATIONS = [
        "entrepreneurship education", "entrepreneurial education",
        "entrepreneurship course", "entrepreneurship curriculum",
        "entrepreneurial intention", "entrepreneurship students",
        "entrepreneurial learning", "entrepreneurship teaching",
    ]

    # ------------------------------------------------------------- checks
    def _population_check(self, sec_name: str, sentence: str, paper: Paper) -> Optional[GroundingIssue]:
        abstract = ((paper.title or "") + " " + (paper.abstract or "")).lower()
        cleaned = abstract
        for colloc in self._EDU_COLLOCATIONS:
            cleaned = cleaned.replace(colloc, " ")
        source_has_focal = any(t in cleaned for t in self.focal_terms)
        if source_has_focal:
            return None
        matched_pop = None
        for label, terms in _OFF_POPULATIONS.items():
            if any(t in abstract for t in terms):
                matched_pop = label
                break
        if not matched_pop:
            return None
        sent_low = sentence.lower()
        claim_asserts_focal = any(t in sent_low for t in self.focal_terms)
        return GroundingIssue(
            section=sec_name,
            issue_type="population_mismatch",
            citation_ref=f"[{paper.short_ref()}]",
            claim_excerpt=sentence[:160],
            detail=(
                f"Source '{paper.title[:60]}…' studies {matched_pop}, not "
                f"{self.domain} populations. "
                + ("The claim frames it as focal-population evidence — reframe honestly as "
                   "indirect evidence from an adjacent domain (naming that domain) or drop it."
                   if claim_asserts_focal else
                   "Presented in a focal-population argument without flagging the adjacent "
                   "domain — add an explicit caveat or drop it.")
            ),
            severity="high" if claim_asserts_focal else "medium",
        )

    def _theory_check(self, sections: Dict[str, str], primary_theory: str) -> List[GroundingIssue]:
        issues: List[GroundingIssue] = []
        discussion = sections.get("Discussion", "")
        if not discussion or not primary_theory:
            return issues
        # Which theory does the Discussion claim to extend?
        m = re.search(
            r'(?:extend|extends|extending|refine[sd]?|advancing)\s+(?:the\s+)?'
            r'([A-Z][A-Za-z\- ]{3,40}?(?:Theory|View|Framework|Model))',
            discussion,
        )
        if m:
            claimed = m.group(1).strip()
            # loose match: primary theory tokens should appear in the claimed name
            primary_tokens = {w.lower() for w in primary_theory.split() if len(w) > 3}
            claimed_tokens = {w.lower() for w in claimed.split() if len(w) > 3}
            if primary_tokens and not (primary_tokens & claimed_tokens):
                issues.append(GroundingIssue(
                    section="Discussion",
                    issue_type="theory_inconsistency",
                    citation_ref="",
                    claim_excerpt=m.group(0)[:160],
                    detail=(
                        f"Discussion extends '{claimed}' but the Theoretical Framework adopted "
                        f"'{primary_theory}'. Use '{primary_theory}' consistently."
                    ),
                    severity="high",
                ))
        return issues

    def _numbers_check(self, discussion: str, consensus) -> List[GroundingIssue]:
        issues: List[GroundingIssue] = []
        if not discussion:
            return issues
        actual = (consensus.support_count, consensus.oppose_count, consensus.mixed_count)
        # Find the stated split, e.g. "15 report a clear positive ... 9 report null ... 10 report mixed"
        m = re.search(
            r'(\d+)\s+(?:papers?|studies)?\s*(?:report|show|find)[^.]{0,90}?positive[^.]*?'
            r'(\d+)\s+(?:papers?|studies)?\s*(?:report|show|find)[^.]{0,90}?(?:null|negative|cautionary)[^.]*?'
            r'(\d+)\s+(?:papers?|studies)?\s*(?:report|show|find)[^.]{0,90}?mixed',
            discussion, re.IGNORECASE | re.DOTALL,
        )
        if m:
            stated = tuple(int(g) for g in m.groups())
            if stated != actual:
                issues.append(GroundingIssue(
                    section="Discussion",
                    issue_type="number_inconsistency",
                    citation_ref="",
                    claim_excerpt=m.group(0)[:160],
                    detail=(
                        f"Stated evidence split {stated} does not match the actual consensus "
                        f"analysis {actual} (support, oppose, mixed) of "
                        f"{consensus.total_papers} stanced papers. Use the actual numbers."
                    ),
                    severity="high",
                ))
        return issues
