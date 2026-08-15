# The scholar layer — how the engine learns from real papers

The generator used to run on my priors: hand-written notes about how a Q1
paper "should" read, and a style gate tuned to what I assumed good academic
prose looks like. This layer replaces both with measurements taken off real
published papers, and keeps the evidence for every claim it makes.

```
harvest_library.py          mine_library.py                the engine
     │                            │                             │
  Crossref                  style_profile.json  ─────► writer prompts
  Semantic Scholar    ──►   style_norms.json    ─────► style gate
  Unpaywall                 research_graph.json ─────► gap detection
     │
  library/pdfs/ + manifest.json + needs_library_access.csv
```

## 1. Harvest — `scripts/harvest_library.py`

Walks Q1 journals through Crossref and resolves each work to a **legally
downloadable** full text: Semantic Scholar `openAccessPdf` → Unpaywall →
publisher links that carry an open licence. Nothing is scraped past a
paywall; those works are queued in `needs_library_access.csv` for a human to
fetch through their institution.

```bash
python scripts/harvest_library.py --journals q1-entrepreneurship --from-year 2019
python scripts/harvest_library.py --dois-from some_reading_list.md
```

The crawl is resumable and never re-downloads a DOI. `manifest.json` records
each work's licence, the PDF's origin and its SHA-256, so any pattern mined
downstream traces back to a specific file. PDFs are gitignored (they are
copyrighted, and the manifest rebuilds the library on demand).

## 2. Learn — `scripts/mine_library.py`

**`style_profile.json` — how Q1 papers write.** Per genre and section: word
budgets, citation density, integral-vs-parenthetical citation share, sentence
and paragraph shape, hedge/booster balance, attested discourse connectors and
phrasings, plus real gap, contribution and discussion-opening sentences with
their DOIs. Injected into every section prompt the writer sends.

**`style_norms.json` — how Q1 prose lints.** Runs the same deslop and
prose_check linters over published papers and records how often each finding
category fires per 1,000 words. This exists because the style gate failed
every generated paper while 21 of 22 published papers use "inflation" words
and half contain findings the linter marks HIGH. The gate now flags only what
exceeds the published 90th percentile.

**`research_graph.json` — what the field has studied.** Reduces each paper to
theory × method × population × outcome, crosses those into a matrix, and reads
the holes: empty cells whose row *and* column are both well populated,
outcomes measured by only one method, and topics whose evidence stops before a
cutoff.

## The rule that makes this usable

A pattern is reported only if it appears in at least `min_papers` **different**
papers, and every entry carries its support count. A gap is claimed only where
both sides of the empty cell have real support. On a thin corpus these
analyses return *nothing* rather than something plausible — that is the
intended behaviour, and the reason the output can be trusted when it is not
empty.

## Two things this has already caught

- **The style gate was measuring nothing.** Recalibrated against published
  practice, it passes 9 of 10 real Q1 papers — and fails the tenth, which
  opens "In today's rapidly evolving business landscape" and carries 4.6× the
  peer AI-vocabulary rate. It reads as machine-written, and it is published.
- **Norms must not be learned from papers like that one.** `build_norms`
  trims the noisiest decile before computing percentiles, or the standard
  drifts toward the style the gate exists to catch.
