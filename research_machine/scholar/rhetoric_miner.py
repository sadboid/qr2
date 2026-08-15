"""Mine how Q1 papers actually write — the engine's writing guidance, learned.

Everything here is derived from parsed full texts, never asserted. The output
is a "style profile": per genre and per section, the real word budgets,
citation densities, paragraph shapes, opening moves, phrase conventions,
reporting verbs, hedging calibration, and the sentence forms authors use to
stake a gap or a contribution.

The discipline that makes it usable: a pattern is only reported if it appears
in at least `min_papers` DIFFERENT papers, and every entry carries its support
count. One author's tic is not a convention, and a profile that cannot show
its support is just priors wearing a lab coat.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

from .pdf_parser import ParsedPaper, citations_in, split_sentences

logger = logging.getLogger(__name__)

_STOP_EDGE = {
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "on", "with", "as",
    "is", "are", "was", "were", "be", "been", "that", "this", "these", "those",
    "it", "its", "by", "at", "from", "we", "our", "their", "which", "than",
}

# Sentences that stake a gap — the move a review lives or dies on.
_GAP_MARKERS = re.compile(
    r"\b(little (?:is )?(?:known|research|attention)|remains? (?:unclear|unknown|unexplored|underexplored)"
    r"|has (?:received )?(?:scant|limited|little) (?:attention|scrutiny)|few studies|no study has"
    r"|under-?(?:explored|researched|theorized)|is (?:still )?lacking|gap in (?:the|our) (?:literature|knowledge|understanding)"
    r"|yet to be|has not (?:yet )?been (?:examined|explored|studied|addressed)|calls? for (?:more|further) research"
    r"|fragmented|we know (?:little|surprisingly little))\b", re.IGNORECASE)

_CONTRIB_MARKERS = re.compile(
    r"\b(this (?:paper|study|article|review) (?:contributes|makes|advances|offers|provides|extends)"
    r"|our (?:contribution|study contributes)|we contribute|contributions? (?:is|are) (?:two|three|four)fold"
    r"|the purpose of this (?:paper|study|review)|this (?:paper|study|review) (?:aims|seeks|sets out))\b",
    re.IGNORECASE)

# Verb immediately after an integral citation: "Smith et al. (2020) argue that..."
_REPORTING = re.compile(
    r"\((?:19|20)\d{2}[a-z]?\)\s+(\w+(?:s|ed)?)\b")

_HEDGES = ["may", "might", "could", "appear", "appears", "seem", "seems", "suggest",
           "suggests", "indicate", "indicates", "likely", "possibly", "potentially",
           "arguably", "tend to", "relatively", "somewhat", "partially"]
_BOOSTERS = ["clearly", "strongly", "demonstrate", "demonstrates", "establish",
             "establishes", "prove", "proves", "conclusively", "undoubtedly",
             "significantly", "substantially", "markedly", "robustly"]


@dataclass
class SectionProfile:
    section: str
    genre: str
    n_papers: int = 0
    words_median: int = 0
    words_range: Tuple[int, int] = (0, 0)
    citations_per_1k: float = 0.0
    integral_share: float = 0.0          # Author (2020) ... vs (Author, 2020)
    words_per_sentence: float = 0.0
    sentences_per_paragraph: float = 0.0
    paragraph_openers: List[dict] = field(default_factory=list)
    connectors: List[dict] = field(default_factory=list)
    phrases: List[dict] = field(default_factory=list)
    reporting_verbs: List[dict] = field(default_factory=list)
    hedges_per_1k: float = 0.0
    boosters_per_1k: float = 0.0


# Fragments that survive extraction from running heads, DOIs and reference
# stubs. They are frequent and cross-paper, so support counting alone will not
# filter them — they have to be named.
_JUNK_GRAM = re.compile(
    r"\b(doi|http|https|www|org|com|issn|isbn|vol|pp|ed|eds|springer|elsevier|wiley"
    r"|econ|bus|res|rev|int|manag|entrep|technol|univ|press|journal of|forthcoming"
    r"|downloaded|licen[cs]e|copyright|creative commons|supplementary)\b")


def _ngrams(sentence: str, n: int) -> Iterable[str]:
    words = re.findall(r"[a-z][a-z'-]+", sentence.lower())
    for i in range(len(words) - n + 1):
        gram = words[i:i + n]
        if gram[0] in _STOP_EDGE or gram[-1] in _STOP_EDGE:
            continue
        text = " ".join(gram)
        if _JUNK_GRAM.search(text):
            continue
        yield text


def _support_ranked(per_paper: Dict[str, Counter], min_papers: int,
                    top: int) -> List[dict]:
    """Rank items by how many DIFFERENT papers use them, then by total count."""
    papers_using: Counter = Counter()
    total: Counter = Counter()
    for doi, counts in per_paper.items():
        for item, c in counts.items():
            papers_using[item] += 1
            total[item] += c
    out = []
    for item, n_papers in papers_using.most_common():
        if n_papers < min_papers:
            continue
        out.append({"text": item, "papers": n_papers, "occurrences": total[item]})
        if len(out) >= top:
            break
    return out


class RhetoricMiner:
    """Builds a style profile from a set of parsed papers."""

    def __init__(self, min_papers: int = 3):
        self.min_papers = min_papers

    def profile_section(self, papers: List[ParsedPaper], section: str,
                        genre: Optional[str] = None) -> Optional[SectionProfile]:
        subset = [p for p in papers
                  if section in p.sections and (genre is None or p.genre == genre)]
        if len(subset) < self.min_papers:
            return None

        prof = SectionProfile(section=section, genre=genre or "any", n_papers=len(subset))
        word_counts, densities, integral_shares = [], [], []
        wps, spp, hedge_rates, boost_rates = [], [], [], []
        openers: Dict[str, Counter] = {}
        connectors: Dict[str, Counter] = {}
        phrases: Dict[str, Counter] = {}
        verbs: Dict[str, Counter] = {}

        for p in subset:
            body = p.sections[section]
            words = body.split()
            n_words = len(words)
            if n_words < 100:
                continue
            word_counts.append(n_words)

            cites = citations_in(body)
            n_cites = len(cites["integral"]) + len(cites["parenthetical"])
            densities.append(1000 * n_cites / n_words)
            if n_cites:
                integral_shares.append(len(cites["integral"]) / n_cites)

            sents = split_sentences(body)
            if sents:
                wps.append(sum(len(s.split()) for s in sents) / len(sents))
            paras = p.paragraphs(section)
            if paras:
                spp.append(sum(len(split_sentences(x)) for x in paras) / len(paras))

            low = body.lower()
            hedge_rates.append(1000 * sum(low.count(h) for h in _HEDGES) / n_words)
            boost_rates.append(1000 * sum(low.count(b) for b in _BOOSTERS) / n_words)

            # Paragraph openers: the first three words carry the move (five-word
            # heads almost never repeat across authors, so they mine to nothing).
            oc = Counter()
            for para in paras:
                first = split_sentences(para)[:1]
                if not first:
                    continue
                clean = re.sub(r"\([^)]*\)", "", first[0]).strip()
                head = " ".join(re.findall(r"[A-Za-z][\w'-]*", clean)[:3]).lower()
                if len(head.split()) == 3:
                    oc[head] += 1
            openers[p.doi] = oc

            # Sentence-initial discourse markers: how these authors signal
            # contrast, addition, consequence — the connective tissue of an argument.
            cc = Counter()
            for s in sents:
                m = re.match(r"([A-Z][a-z]+(?:\s+[a-z]+){0,2}),\s", s)
                if m:
                    cc[m.group(1).lower()] += 1
            connectors[p.doi] = cc

            pc = Counter()
            for s in sents:
                s_clean = re.sub(r"\([^)]*\)", " ", s)
                for n in (3, 4):
                    for g in _ngrams(s_clean, n):
                        pc[g] += 1
            phrases[p.doi] = pc

            verbs[p.doi] = Counter(
                m.group(1).lower() for m in _REPORTING.finditer(body)
                if len(m.group(1)) > 3)

        if not word_counts:
            return None

        prof.words_median = int(median(word_counts))
        prof.words_range = (min(word_counts), max(word_counts))
        prof.citations_per_1k = round(median(densities), 1) if densities else 0.0
        prof.integral_share = round(median(integral_shares), 2) if integral_shares else 0.0
        prof.words_per_sentence = round(median(wps), 1) if wps else 0.0
        prof.sentences_per_paragraph = round(median(spp), 1) if spp else 0.0
        prof.hedges_per_1k = round(median(hedge_rates), 1) if hedge_rates else 0.0
        prof.boosters_per_1k = round(median(boost_rates), 1) if boost_rates else 0.0
        prof.paragraph_openers = _support_ranked(openers, self.min_papers, 25)
        prof.connectors = _support_ranked(connectors, self.min_papers, 20)
        prof.phrases = _support_ranked(phrases, self.min_papers, 40)
        prof.reporting_verbs = _support_ranked(verbs, self.min_papers, 25)
        return prof

    def mine_move_sentences(self, papers: List[ParsedPaper],
                            pattern: re.Pattern, section: str = "introduction",
                            max_per_paper: int = 3, limit: int = 60) -> List[dict]:
        """Collect real sentences performing a rhetorical move (gap, contribution).

        These are exemplars, not templates: each is stored with its DOI so a
        writer can be shown what the move looks like in the wild and a reader
        can check it against the source.
        """
        out = []
        for p in papers:
            body = p.sections.get(section, "")
            if not body:
                continue
            hits = 0
            for s in split_sentences(body):
                if 40 < len(s) < 400 and pattern.search(s):
                    out.append({"doi": p.doi, "genre": p.genre,
                                "sentence": re.sub(r"\s+", " ", s).strip()})
                    hits += 1
                    if hits >= max_per_paper:
                        break
        return out[:limit]

    def build_profile(self, papers: List[ParsedPaper]) -> dict:
        """Full style profile: per-genre section profiles + move exemplars."""
        genres = Counter(p.genre for p in papers)
        sections = ["abstract", "introduction", "literature", "methods",
                    "results", "discussion", "conclusion"]
        profile: dict = {
            "corpus": {
                "n_papers": len(papers),
                "genres": dict(genres),
                "min_papers_per_pattern": self.min_papers,
            },
            "sections": {},
            "moves": {},
        }
        for genre in [g for g, n in genres.items() if n >= self.min_papers] + ["any"]:
            g = None if genre == "any" else genre
            for sec in sections:
                prof = self.profile_section(papers, sec, g)
                if prof:
                    profile["sections"].setdefault(genre, {})[sec] = {
                        k: v for k, v in vars(prof).items()
                        if k not in ("section", "genre")
                    }
        profile["moves"]["gap_statements"] = self.mine_move_sentences(
            papers, _GAP_MARKERS, "introduction")
        profile["moves"]["contribution_statements"] = self.mine_move_sentences(
            papers, _CONTRIB_MARKERS, "introduction")
        profile["moves"]["discussion_openers"] = [
            {"doi": p.doi, "sentence": re.sub(r"\s+", " ", s).strip()}
            for p in papers if p.sections.get("discussion")
            for s in split_sentences(p.sections["discussion"])[:1]
            if 40 < len(s) < 400
        ][:40]
        return profile


def mine_library(library_root: str, out_path: str,
                 min_papers: int = 3, max_papers: Optional[int] = None) -> dict:
    """Parse every PDF in a harvested library and write its style profile."""
    from .pdf_parser import parse_pdf
    root = Path(library_root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [e for e in manifest["entries"] if e.get("pdf_path")]
    if max_papers:
        entries = entries[:max_papers]

    parsed: List[ParsedPaper] = []
    failed = 0
    for e in entries:
        p = parse_pdf(str(root / e["pdf_path"]), e["doi"])
        if p.ok:
            parsed.append(p)
        else:
            failed += 1
    logger.info(f"[Miner] parsed {len(parsed)}/{len(entries)} papers ({failed} unmineable)")

    profile = RhetoricMiner(min_papers=min_papers).build_profile(parsed)
    profile["corpus"]["unmineable"] = failed
    Path(out_path).write_text(json.dumps(profile, indent=1))
    logger.info(f"[Miner] style profile -> {out_path}")
    return profile
