#!/usr/bin/env python3
"""AI-tell detector ("de-slop") — flags the patterns that make prose read as
LLM-generated. Standard library only; runs on any Python >= 3.9.

Rules adapted for ACADEMIC writing from two open rule sets:
  - github.com/blader/humanizer  (24-pattern framework; also ported zh by op7418)
  - github.com/conorbronsdon/avoid-ai-writing  (tiered blocklists)
DELIBERATELY NOT USED: translation-chain "humanizers" (e.g. lynote/humanize-text)
— round-tripping through other languages corrupts numbers, DOIs, and quotations,
which is disqualifying for a paper. We detect and rewrite by hand instead.

Academic tuning: terms that are legitimate constructs in business/management
research are NOT flagged — ecosystem, paradigm, disruptive (innovation),
robust(ness), framework, significant (handled by prose_check/qa_check).

USAGE (from repo root):
    python3 tools/deslop.py paper2/05-manuscript/manuscript.md
    python3 tools/deslop.py <ms.md> --strict     # exit 1 on any HIGH
    python3 tools/deslop.py <a.md> <b.md>

Categories:
  HIGH   chatbot artifacts, emoji, unfilled placeholders (never OK in a paper)
  MEDIUM AI vocabulary; negative parallelism; high transition-word density
  LOW    copula-avoidance verbs; filler; uniform sentence rhythm
Default advisory (exit 0). --strict fails on HIGH.
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

# AI vocabulary rarely justified in rigorous academic prose (curated to avoid
# false hits on real constructs: ecosystem/paradigm/disruptive/robust are NOT here)
AI_VOCAB = [
    "delve", "delving", "leverage", "leveraging", "utilize", "utilizing",
    "utilization", "seamless", "seamlessly", "showcase", "showcases",
    "showcasing", "boast", "boasts", "tapestry", "testament to", "underscore",
    "underscores", "underscoring", "realm", "myriad", "plethora", "groundbreaking",
    "cutting-edge", "game-changer", "game-changing", "vibrant", "nestled",
    "watershed", "pivotal moment", "in today's", "ever-evolving", "ever-changing",
    "navigating the", "at the forefront", "a beacon", "unlock the", "unlocking",
    "harness", "harnessing", "usher in", "reimagine", "reimagining",
]
NEG_PARALLEL = [
    r"not only\b[^.]{0,80}?\bbut also", r"it'?s not (just|only)\b[^.]{0,60}?\bit'?s",
    r"is not (just|merely)\b[^.]{0,60}?\bbut rather",
]
COPULA = ["serves as", "serve as", "acts as", "act as", "functions as a"]
FILLER = {"in order to": "to", "due to the fact that": "because",
          "in the event that": "if", "a majority of": "most",
          "with regard to": "on", "in spite of the fact that": "although"}
TRANSITIONS = ["moreover", "furthermore", "additionally", "notably", "importantly",
               "crucially", "indeed", "ultimately", "arguably"]
CHATBOT = ["i hope this helps", "let me know if", "certainly!", "great question",
           "as an ai", "i cannot", "here's what you need to know", "let's dive",
           "in conclusion, ", "as we can see", "it's worth noting that"]
PLACEHOLDER = [r"\[your name\]", r"\[insert[^\]]*\]", r"\[tbd\]", r"xxxx", r"citeturn",
               r"utm_source=chatgpt"]
EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def prose_lines(text):
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
                or (re.match(r"^\*.*\*$", s) and len(s) < 200)):
            out.append((i, ""))
            continue
        out.append((i, raw))
    return out


def split_sentences(t):
    for a in ("et al.", "e.g.", "i.e.", "cf.", "vs.", "U.S.", "U.K.", "Fig.", "No.",
              "pp.", "ca.", "Inc.", "St."):
        t = t.replace(a, a.replace(".", "\x00"))
    return [p.replace("\x00", ".").strip()
            for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]


def lint(path=None, text=None):
    if text is None:
        text = Path(path).read_text()
    lines = prose_lines(text)
    findings, sent_lens = [], []
    word_ct, trans_ct = 0, 0

    def add(sev, cat, ln, snip):
        findings.append((sev, cat, ln, snip.strip()[:100]))

    for ln, line in lines:
        if not line:
            continue
        low = line.lower()
        for m in EMOJI.finditer(line):
            add("HIGH", "emoji", ln, line)
            break
        for pat in PLACEHOLDER:
            if re.search(pat, low):
                add("HIGH", f"unfilled placeholder / artifact ({pat})", ln, line)
        for p in CHATBOT:
            if p in low:
                add("HIGH", f"chatbot artifact ('{p.strip()}')", ln, line)
        for term in AI_VOCAB:
            if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", low):
                add("MEDIUM", f"AI vocabulary ('{term}')", ln, line)
        for pat in NEG_PARALLEL:
            if re.search(pat, low):
                add("MEDIUM", "negative parallelism ('not only… but also' / 'not X but Y')", ln, line)
        for term in COPULA:
            if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", low):
                add("LOW", f"copula-avoidance ('{term}' → 'is/has')", ln, line)
        for term, repl in FILLER.items():
            if term in low:
                add("LOW", f"filler ('{term}' → '{repl}')", ln, line)
        if '"' in line and ("“" in line or "”" in line):
            pass

    # paragraph-level: transition density + sentence rhythm
    para, buf = [], []
    for ln, line in lines:
        if line.strip():
            buf.append(line)
        elif buf:
            para.append(" ".join(buf)); buf = []
    if buf:
        para.append(" ".join(buf))
    for ptext in para:
        word_ct += len(ptext.split())
        low = ptext.lower()
        for t in TRANSITIONS:
            trans_ct += len(re.findall(r"(?<![\w])" + t + r"(?![\w])", low))
        for s in split_sentences(ptext):
            sent_lens.append(len(s.split()))

    trans_density = round(trans_ct / max(word_ct, 1) * 1000, 1)
    if trans_density > 6:
        findings.append(("MEDIUM", f"transition-word density {trans_density}/1000 "
                         f"({trans_ct} moreover/furthermore/notably…)", 0, ""))
    cv = 0.0
    if len(sent_lens) > 5 and statistics.mean(sent_lens):
        cv = statistics.pstdev(sent_lens) / statistics.mean(sent_lens)
        if cv < 0.45:
            findings.append(("LOW", f"uniform sentence rhythm (CV={cv:.2f}; vary "
                             "short/long — AI writes at one length)", 0, ""))
    return findings, word_ct, trans_density, cv


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    if not files:
        print(__doc__)
        return 1
    total_high = 0
    for f in files:
        findings, wc, td, cv = lint(f)
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda x: (order[x[0]], x[2]))
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for sev, *_ in findings:
            counts[sev] += 1
        total_high += counts["HIGH"]
        print(f"\nDE-SLOP — {f}  ({wc} prose words)")
        print("=" * 64)
        cap = {"HIGH": 99, "MEDIUM": 15, "LOW": 8}
        shown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for sev, cat, ln, sn in findings:
            if shown[sev] >= cap[sev]:
                continue
            shown[sev] += 1
            mark = {"HIGH": "‼", "MEDIUM": "•", "LOW": "·"}[sev]
            loc = f"L{ln}" if ln else "—"
            print(f"  {mark} {sev:6} {loc:<6} {cat}")
            if sn:
                print(f"        “{sn}”")
        for sev in ("MEDIUM", "LOW"):
            if counts[sev] > cap[sev]:
                print(f"  … +{counts[sev]-cap[sev]} more {sev}")
        print("-" * 64)
        print(f"  HIGH {counts['HIGH']} · MEDIUM {counts['MEDIUM']} · "
              f"LOW {counts['LOW']} · transitions {td}/1k · rhythm CV {cv:.2f}")
    if strict and total_high:
        print(f"\nRESULT: FAIL ({total_high} HIGH; --strict)")
        return 1
    print(f"\nRESULT: {'PASS' if not total_high else 'ADVISORY'} ({total_high} HIGH total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
