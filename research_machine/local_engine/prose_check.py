#!/usr/bin/env python3
"""Academic prose linter — enforces the writing rules in playbooks/ that were
previously only advice. Standard library only; runs on any Python >= 3.9.

USAGE (from repo root):
    python3 tools/prose_check.py paper2/05-manuscript/manuscript.md
    python3 tools/prose_check.py <ms.md> --strict     # exit 1 if any HIGH finding
    python3 tools/prose_check.py <a.md> <b.md>         # lint several files

Default is ADVISORY (exit 0): prose quality needs judgment, and some flags are
legitimate in context, so this reports rather than blocks. Use --strict in CI to
fail on HIGH-severity findings (the ones almost always wrong).

It SKIPS: the References section, tables (| … |), headings (#), images (![…]),
figure/table caption lines (*Figure*/*Table*), block quotes (>), and fenced code.

Categories:
  HIGH   significant/-ly with no number in the sentence (unquantified stat claim);
         throat-clearing openers; asserted novelty ("novel", "first to")
  MEDIUM adjective/adverb inflation; hedge-stacking; sentences > 45 words
  LOW    weasel words; passive-voice density (informational)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

INFLATION = [
    "very", "vast", "hugely", "huge", "massive", "dramatically", "dramatic",
    "substantially", "substantial", "remarkably", "remarkable", "strikingly",
    "striking", "clearly", "obviously", "undoubtedly", "tremendous", "tremendously",
    "crucial", "essential", "vitally", "extremely", "highly notable", "profound",
    "profoundly", "immense", "compelling", "powerful",
]
THROAT = [
    "it is important to note", "it should be noted", "it is worth noting",
    "it is worth mentioning", "it is interesting to note", "needless to say",
    "it goes without saying", "as is well known", "in order to note",
    "it must be emphasized", "it is well established that", "importantly,",
]
NOVELTY = [
    "novel", "novelty", "for the first time", "to the best of our knowledge",
    "first study to", "first to examine", "first to show", "unprecedented",
]
HEDGE_STACK = [
    "may potentially", "could possibly", "might possibly", "may possibly",
    "could potentially", "potentially could", "may be able to potentially",
]
WEASEL = ["somewhat", "rather", "quite", "fairly", "relatively", "arguably",
          "to some extent", "more or less"]
PASSIVE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+(\w+ed|written|shown|found|given|taken|"
    r"made|seen|done|known|held|built|drawn|set|put|kept|drawn|chosen|driven)\b", re.I)
NUM = re.compile(r"\d|%|\bp\s*[<=>]|\bp-value|standard deviation|percent")
ABBR = ["et al.", "e.g.", "i.e.", "cf.", "vs.", "U.S.", "U.K.", "Dr.", "Fig.",
        "No.", "pp.", "ca.", "Inc.", "Ltd.", "St.", "approx."]


def prose_lines(text):
    """Return (lineno, line) for prose; a break is emitted as (lineno, "") at
    blanks / headings / tables / captions so paragraphs split correctly. Skips
    the References section and fenced code entirely."""
    out, in_code, in_refs = [], False, False
    for i, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if s.startswith("```"):
            in_code = not in_code
            out.append((i, ""))
            continue
        if in_code:
            continue
        if re.match(r"^#{1,6}\s", s):
            in_refs = bool(re.search(r"\breferences\b", s, re.I))
            out.append((i, ""))
            continue
        if in_refs:
            continue
        if (not s or s.startswith(("|", ">", "!", "![", "---"))
                or re.match(r"^\*(figure|table)\b", s, re.I)
                or (re.match(r"^\*.*\*$", s) and len(s) < 200)):
            out.append((i, ""))  # structural line = paragraph break
            continue
        out.append((i, raw))
    return out


def split_sentences(paragraph):
    t = paragraph
    for a in ABBR:
        t = t.replace(a, a.replace(".", "\x00"))
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def find_terms(line, terms):
    hits = []
    for term in terms:
        for m in re.finditer(r"(?<![\w])" + re.escape(term) + r"(?![\w])", line, re.I):
            hits.append((term, m.start()))
    return hits


def lint(path=None, text=None):
    if text is None:
        text = Path(path).read_text()
    lines = prose_lines(text)
    findings = []  # (severity, category, lineno, snippet)

    def snip(line, pos=0):
        s = line.strip()
        return (s[:100] + "…") if len(s) > 100 else s

    # per-line lexical checks
    for ln, line in lines:
        if not line:
            continue
        low = line.lower()
        for term, _ in find_terms(low, THROAT):
            findings.append(("HIGH", "throat-clearing", ln, snip(line)))
            break
        for term, _ in find_terms(low, NOVELTY):
            findings.append(("HIGH", f"asserted novelty ('{term}')", ln, snip(line)))
        for term, _ in find_terms(low, HEDGE_STACK):
            findings.append(("MEDIUM", f"hedge-stacking ('{term}')", ln, snip(line)))
        for term, _ in find_terms(low, INFLATION):
            findings.append(("MEDIUM", f"inflation ('{term}')", ln, snip(line)))
        for term, pos in find_terms(low, WEASEL):
            # "X rather than Y" / "relative to" are legitimate contrasts, not weasel
            if term == "rather" and low[pos:pos + 12].startswith("rather than"):
                continue
            findings.append(("LOW", f"weasel ('{term}')", ln, snip(line)))

    # sentence-level: "significant" without a number, and long sentences.
    # rebuild paragraphs from prose lines (keep first line no. of each paragraph)
    para, para_start, passive_ct, word_ct, emdash_ct = [], None, 0, 0, 0
    paras = []
    for ln, line in lines:
        if line.strip():
            if para_start is None:
                para_start = ln
            para.append(line)
        elif para:  # break marker ends the current paragraph
            paras.append((para_start, " ".join(para)))
            para, para_start = [], None
    if para:
        paras.append((para_start, " ".join(para)))

    for pstart, ptext in paras:
        word_ct += len(ptext.split())
        passive_ct += len(PASSIVE.findall(ptext))
        emdash_ct += ptext.count("—")
        for sent in split_sentences(ptext):
            n = len(sent.split())
            if re.search(r"\bsignificant(ly)?\b", sent, re.I) and not NUM.search(sent):
                findings.append(("HIGH", "'significant' with no number in sentence",
                                 pstart, sent[:100] + ("…" if len(sent) > 100 else "")))
            if n > 45:
                findings.append(("MEDIUM", f"long sentence ({n} words)",
                                 pstart, sent[:100] + "…"))
            if sent.count("—") >= 2:
                findings.append(("MEDIUM", f"em-dash pile-up ({sent.count('—')} in one sentence)",
                                 pstart, sent[:100] + ("…" if len(sent) > 100 else "")))

    passive_density = round(passive_ct / max(word_ct, 1) * 1000, 1)
    emdash_density = round(emdash_ct / max(word_ct, 1) * 1000, 1)
    if emdash_density > 12:
        findings.append(("LOW", f"em-dash density {emdash_density}/1000 words "
                         f"({emdash_ct} total — a crutch; prefer commas/periods)", 0, ""))
    return findings, passive_density, word_ct, emdash_density


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    if not files:
        print(__doc__)
        return 1
    total_high = 0
    for f in files:
        findings, pd, wc, ed = lint(f)
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda x: (order[x[0]], x[2]))
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for sev, *_ in findings:
            counts[sev] += 1
        total_high += counts["HIGH"]
        print(f"\nPROSE CHECK — {f}  ({wc} prose words)")
        print("=" * 64)
        shown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        cap = {"HIGH": 99, "MEDIUM": 12, "LOW": 8}
        for sev, cat, ln, sn in findings:
            if shown[sev] >= cap[sev]:
                continue
            shown[sev] += 1
            mark = {"HIGH": "‼", "MEDIUM": "•", "LOW": "·"}[sev]
            print(f"  {mark} {sev:6} L{ln:<4} {cat}")
            print(f"        “{sn}”")
        for sev in ("MEDIUM", "LOW"):
            if counts[sev] > cap[sev]:
                print(f"  … +{counts[sev] - cap[sev]} more {sev}")
        print("-" * 64)
        print(f"  HIGH {counts['HIGH']} · MEDIUM {counts['MEDIUM']} · "
              f"LOW {counts['LOW']} · passive {pd}/1k"
              + ("(high)" if pd > 25 else "")
              + f" · em-dash {ed}/1k" + ("(high)" if ed > 12 else ""))
    if strict and total_high:
        print(f"\nRESULT: FAIL ({total_high} HIGH findings; --strict)")
        return 1
    print(f"\nRESULT: {'PASS' if not total_high else 'ADVISORY'} "
          f"({total_high} HIGH total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
