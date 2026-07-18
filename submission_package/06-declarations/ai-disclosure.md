# AI Use Disclosure

In accordance with journal and COPE guidance on the use of generative AI in research and manuscript preparation, the author discloses the following.

## Scope of AI assistance

This manuscript was produced with **substantial AI assistance** using a custom research pipeline built on Claude (Anthropic), under the direction and review of the human author. AI systems were used for:

1. **Literature retrieval and screening** — programmatic search of Semantic Scholar, arXiv, Crossref, and OpenAlex; deduplication; domain-relevance filtering.
2. **Extractive synthesis** — identification of findings, gaps, and methodological patterns from the abstracts (and, where openly available, full texts) of the included studies.
3. **Drafting** — the Abstract, Introduction, Results (thematic synthesis), and Discussion sections were drafted by a large language model constrained to a locked set of verified sources, a locked theoretical lens, and locked corpus statistics.
4. **Verification** — automated checks verified (i) every literature-review claim against the cited source's abstract, (ii) citation existence against the verified source library, (iii) population fidelity of interpretive claims, and (iv) statistical/count consistency. Three independent AI referee personas reviewed the manuscript adversarially before release.

## What AI did NOT do

- No data were fabricated, simulated, or imputed; the review synthesizes published literature only.
- No citations were invented: every reference in the manuscript corresponds to a real, database-verified publication (see `04-references/references.bib` and `05-data/corpus-review-matrix.csv`).
- The human author directed the research question, methodology, and revision decisions, and takes full responsibility for the content.

## Transparency artifacts included with this submission

| Artifact | Purpose |
|---|---|
| `05-data/corpus-review-matrix.csv` | Full list of included studies with bibliographic records |
| `05-data/consensus-analysis.json` | Stance classification of every study + verbatim evidence |
| `05-data/quality-gates-report.json` | Machine-verification scores (claim verification, grounding, style, adversarial review) |
| `provenance.json` | Manuscript SHA-256, generation timestamp, producing code commit, environment |

## Author responsibility statement

The author has reviewed the manuscript, verified the accuracy of the claims and citations to the best of their ability, and accepts full responsibility for its content, in line with authorship criteria that AI systems cannot satisfy.
