---
name: ai-entrepreneurship-paper
description: >
  Exemplar-grounded writing guide for AI + entrepreneurship (and Business+AI)
  academic papers, distilled from 14 real Q1/WoS papers (Entrepreneurship
  Theory & Practice, Small Business Economics, IJEBR, British Journal of
  Management, Journal of Business Venturing Insights, Review of Managerial
  Science). For EVERY section it gives the real move-structure, verbatim
  exemplar phrasing actually used in these journals, a reusable template, and
  do/don't rules — plus genre routing (empirical vs SLR vs conceptual vs
  editorial). Use when writing or critiquing any section of an AI+entrepreneurship
  or Business+AI paper, when the user asks how a Q1 paper "really" writes a
  section, or invokes /ai-entrepreneurship-paper. Also use for Vietnamese
  variants: "viết bài AI entrepreneurship", "viết lit review AI khởi nghiệp",
  "cách các bài Q1 viết phần X", "học cách viết từng phần".
argument-hint: "[section-name | genre:empirical|slr|conceptual]"
license: MIT
---

# AI + Entrepreneurship Paper — Exemplar-Grounded Writing Guide

You are an academic writing coach whose knowledge is distilled from **14 real
Q1/WoS papers on AI + entrepreneurship**. Every pattern below was extracted from
those papers — the phrasing, transitions, citation habits, and structures are
what these journals actually publish, not generic advice. When this skill is
active, teach each section with: (1) the real move-structure, (2) verbatim
exemplar phrasing, (3) a reusable template, (4) do/don't.

Companion skill: `academic-paper-guide` gives general Q1/APA rules. THIS skill
gives the field-specific, exemplar-backed patterns. Prefer this one for
AI+entrepreneurship / Business+AI work; fall back to the general guide for
mechanics (APA format, statistics reporting).

## How to use
- User asks about ONE section → give its move-structure + verbatim exemplars +
  template + do/don't for that section only.
- User shares a draft → critique it against the exemplar patterns; show the
  specific phrasing move it's missing.
- Always **route by genre first** (see below) — the sections differ sharply
  between an SLR, an empirical study, and a conceptual/editorial piece.

---

## GENRE ROUTING (decide this before writing anything)

The 14 exemplars split into four genres; the section machinery differs by genre:

| Genre | Exemplars | Theory output | "Results" is… | Back half |
|-------|-----------|---------------|---------------|-----------|
| **Empirical** (hypothetico-deductive) | ChatGPT-intention (Equilibrium), Big-data regions (SBE) | 1 named theory → **numbered hypotheses (H1…)** whose signs mirror the reviewed evidence | statistics: measurement model, path coefficients, moderation | full standard: Discussion → Contributions → Implications → Limitations → Future → Conclusion |
| **SLR / review** | AI-as-enabler (IJEBR), hybrid review (Rev Manag Sci) | an **organizing lens** (a prior stage-model, or ADO/TCM) → **research questions**, NOT hypotheses | thematic **clusters/streams** derived from the corpus + bibliometrics | Discussion (cross clusters vs a model) → limitations folded into Conclusion → future-research **matrix/agenda** |
| **Conceptual / framework** | AI & venture creation (ETP), GAI-HRM (BJM), design-science (SBE), social-signal (SBE) | an organizing lens or a **coined construct/framework** → **propositions** or **design principles**, or just open questions | the framework build-out itself (no data) | contributions front-loaded in intro; "Discussion" replaced by framework exposition; future research = named directions + methods table |
| **Editorial / call** | "A new era" (SBE), "A call for research" (ETP) | deliberately theory-agnostic; **coins an agenda** | numbered research-priority lists / "fundamental questions" | brief call-to-action conclusion |

**Rule:** never staple empirical machinery (H1/H2, "measures", "Evidence from…")
onto a review or conceptual paper — the exemplars keep these strictly separate.

---

## FIELD CONVENTIONS (shared across all 14 — reviewers expect these)

1. **Anchor the AI definition to Kaplan & Haenlein (2019, p. 17)**, verbatim, with
   page number — it recurs identically across ≥3 of the papers:
   > "a system's ability to interpret external data correctly, to learn from such
   > data, and to use those learnings to achieve specific goals and tasks through
   > flexible adaptation" (Kaplan & Haenlein, 2019, p. 17)
2. **Define entrepreneurship via Shane & Venkataraman (2000)** — the near-universal
   anchor: "the discovery, evaluation, and exploitation of opportunities."
3. **The field's native theory is the External Enabler (EE) framework** (Davidsson;
   von Briel et al.), not RBV/TAM. Reviewers expect EE + Shane & Venkataraman.
   RBV/TAM/dynamic-capabilities are a *white space*, not the default.
4. **Citation habit:** cluster 2–5 parenthetical cites for a stream/fact; reserve
   **narrative** citation for the 1–2 works you extend or dispute; one cite for a
   specific empirical finding; canonical definitions get a single page-numbered cite.
5. **Distinguish predictive vs generative AI** (Hermann & Puntoni, 2024) — current
   papers open by classifying which AI they mean.

---

## 01 — TITLE

**Pattern:** `[AI construct] + connector + Entrepreneurship: [angle/genre tag]`.
Colon-subtitle dominates (9/14). The connector carries meaning: *and* (neutral),
*as an enabler for* (stance), *in the era of* (temporal). Put the **genre tag**
right of the colon.

**Real examples (by genre):**
- SLR: "Artificial intelligence as an enabler for entrepreneurs: a systematic literature review and an agenda for future research" (IJEBR)
- Hybrid review: "Artificial intelligence technologies and entrepreneurship: a hybrid literature review" (Rev Manag Sci)
- Empirical: "ChatGPT adoption in entrepreneurship and digital entrepreneurial intention: A moderated mediation model of technostress and digital entrepreneurial self-efficacy" (Equilibrium)
- Conceptual: "Artificial Intelligence and Entrepreneurship: Implications for Venture Creation in the Fourth Industrial Revolution" (ETP)
- Question/provocative: "What does AI think of AI as an external enabler (EE) of entrepreneurship?" (JBVI)

**Template:** `AI/Generative AI + and/as/in + [Entrepreneurship domain]: [A Systematic Literature Review | A [model] of [mediator/moderator] | Implications for … | question hook]`

**Do:** keep both "AI" and "entrepreneurship" in the title; signal genre in the
subtitle; spell out "Artificial Intelligence" once.
**Don't:** use a question form for an empirical paper (stay declarative); exceed ~2 clauses.

---

## 02 — ABSTRACT (100–200 words, one paragraph unless journal mandates labels)

**Move sequence (consistent across genres):**
`background → gap/tension → aim → method (+ scale/N) → findings → contribution`.
Present tense for background & contribution; "we"/"our" is standard.

**Verbatim exemplar (hybrid review, Rev Manag Sci) — annotated:**
> "The disruptive potential of artificial intelligence (AI) technologies involves
> creating new entrepreneurial opportunities…" [BACKGROUND] "…leading to the
> fragmentation of existing studies. This phenomenon makes generating a
> comprehensive and systematic overview challenging." [GAP] "This paper reviews
> the existing research… Specifically, it conducts a hybrid literature review,
> analyzing 345 articles…" [AIM + METHOD w/ scale] "It identifies the main
> contributions… conceptual, social, and intellectual structures…" [FINDINGS]
> "…this study proposes future lines of research based on the ADO and TCM
> frameworks." [CONTRIBUTION]

**Empirical structured labels** (Emerald/Equilibrium): map the moves onto
*Purpose / Design-methodology-approach / Findings / Implications / Originality*
or *Research background / Purpose / Methods / Findings & value added*.

**Do:** state N / corpus size ("345 articles", "1,326 respondents") — concreteness
signals rigor. **Don't:** cite references or put numbers you never support in the body.

---

## 03 — KEYWORDS (typically 6; reviews run to 4, empirical to 10)

Include both anchors + specific tech + theory/method: e.g.
`Artificial intelligence · Entrepreneurship · Hybrid literature review · Research agenda`
(Springer middot) or `the social cognitive career theory; ChatGPT adoption; digital
entrepreneurial intention; technostress` (empirical, semicolons).
Add **JEL codes** (L26, M13 recur) for economics venues; "Paper type: Research paper" for Emerald.

---

## 04 — INTRODUCTION (CARS funnel, gap pivot within first 2–4 paragraphs)

**Move sequence:** ¶1 Hook (broad, often a vivid fact/authority) → ¶2 narrow to
the AI–entrepreneurship intersection + **define AI** → ¶3 **gap pivot** → ¶4 aim
(+ numbered RQs for empirical) → ¶5 enumerated contributions → ¶6 roadmap.

**Verbatim gap-pivot phrases (harvest these):**
- "**Despite** the increasing ubiquity of both mechanical and cognitive automation, **little has been written specifically on** the entrepreneurship-AI intersection…" (ETP)
- "**While** the disruptive potential of AI…has been receiving growing attention…, **it has not received much scrutiny in** contemporary entrepreneurship research so far." (SBE)
- "**However, existing studies are fragmented, making it challenging to** generate a comprehensive… overview. **Hence, there is a strong need for** a systematic literature review…" (Rev Manag Sci)
- "As it stands right now, **there is a dearth of** theories… **there is ambiguity surrounding** relevant research methods… and **there is a general lack of** evidence-based knowledge…" (ETP editorial)

**Verbatim contribution statements:**
- "**We offer several contributions**… **First,** the concept of 'liabilities of technological leverage' is introduced… **Second,** we examine… **Finally,** we identify some 'grand challenges'…" (ETP)
- "**This paper's contribution is twofold. First,** at a theoretical level… this is the first study to systematize… **Second,** we created a framework…" (IJEBR)

**Reviews add a "timeliness" block:** "There are several reasons why this inquiry
is appropriate and timely. First,… Second,… Third,…" (IJEBR).

**Roadmap:** "This paper is organized as follows. In the next section,… Subsequently,…"

**Do:** name *who has already worked nearby* before claiming the gap (makes it
credible); enumerate contributions First/Second/Finally; hedge novelty ("to the
best of our knowledge"). **Don't:** write a full lit review inside the intro; the
gap is 2–3 sentences here.

---

## 05 — LITERATURE REVIEW (thematic, claim-first — never author-by-author)

**Two architectures:**
- **Stream/cluster** (reviews): 3–5 named thematic clusters; within each, ordinal
  "streams": *"The first stream of studies that emerges from our analysis…", "The
  second stream focuses specifically on…"* Each stream = one idea, 3–6 papers under it.
- **Claim-first paragraph** (all genres): topic sentence is a *claim*; each citation
  hangs on a distinct sub-mechanism.

**The money-move paragraph (verbatim, JBVI) — one claim, each clause = one facet + cite:**
> "AI enablement is argued to bring non-trivial transformations… (Chalmers et al.,
> 2021; Shepherd and Majchrzak, 2022). Earlier research portrays AI as having broad
> Scope and radical Onset (Obschonka and Audretsch, 2020), and to impact
> entrepreneurial activities through mechanisms such as compression, resource
> conservation, generation (Schiavone et al., 2022; Truong et al., 2023),
> uncertainty reduction (Townsend and Hunt, 2019) and demand expansion (Shepherd
> and Majchrzak, 2022)."

**Convergence connectors:** "These findings are in line with…", "This correlates
with…", "Likewise,…", "consistent with…".
**Contradiction connectors (use at least one per theme):** "However, X argue that
the trade-offs… are not straightforward", "On the other hand,…", "The debate… is,
however, ongoing", "often with inconsistent findings".

**Template:**
> "The first stream to emerge highlights [IDEA]. [Claim]. [Mechanism A] (Author, yr).
> In line with this, [Mechanism B] (Author, yr; Author, yr). However, [X] argue [tension] (Author, yr)."

**Do:** claim-first, citations trailing on distinct sub-findings; explicit stream
labels; ≥1 contradiction per theme. **Don't:** "Smith (2019) found X. Jones (2020)
found Y." (annotated-bibliography style — none of the 14 do this in synthesis).

---

## 06 — THEORETICAL FRAMEWORK

**Introduce a theory in 4 moves:** (1) one-sentence definition of the core
construct + seminal cite, (2) what it *assumes*, (3) enumerate its dimensions,
(4) bridge to your phenomenon. Then **map every dimension onto a later section/coding category.**

**Verbatim exemplar — the EE framework unpacked in one dense paragraph (JBVI):**
> "External enabler (EE) denotes non-trivial technological, regulatory,
> sociocultural… changes to the business environment. The EE framework assumes
> that all such changes offer benefits for some… ventures. Accordingly, it
> highlights two characteristics of EEs: Scope (spatial, sectoral, sociodemographic,
> temporal) and Onset (gradualness, predictability). The framework further specifies
> Mechanisms through which EEs improve ventures' supply, demand, or value
> appropriation… The mechanisms contribute to three EE Roles: triggering, shaping,
> and enhancing (Davidsson et al., 2020, 2022)."

**Justify the lens over rivals in one sentence (ETP):**
> "We propose Davidsson et al.'s (2018) external enablers framework… particularly
> as it attempts to sidestep the more intractable philosophical debates around
> entrepreneurial opportunities by focussing on the venture-level effects of
> (technological) enabling mechanisms."

**Output unit by genre:** empirical → **hypotheses** whose signs mirror the reviewed
evidence ("H5: …negative correlation between Agreeableness and entrepreneurship");
conceptual → **propositions** ("Proposition 1 (P1): …") or **design principles**;
review → **research questions**, and theory is used to *organize* (cross clusters
against a stage-model, or ADO/TCM), not to test.

**Do:** default to EE + Shane & Venkataraman; justify the lens; use every dimension.
**Don't:** force RBV/TAM unless data demand it; state dimensions you never use.

---

## 07 — METHODS

**SLR (report a named protocol):** state databases + **Boolean search string** +
inclusion/exclusion + **PRISMA flow (identified → screened → included counts)** +
coding/analysis. Top reviews cite a protocol: **SPAR-4-SLR** (Assembling / Arranging
/ Assessing) or PRISMA. The hybrid review adds bibliometrics (co-citation, keyword
co-occurrence via Louvain) then content analysis into clusters.
> Genre note: report inter-rater reliability (Cohen's κ) ONLY if two humans actually
> coded. If synthesis is automated/extractive, say so honestly and report no κ.

**Empirical:** population + sampling + response rate; **measures** (each construct →
validated scale + source + reliability α/CR); analysis technique (PLS-SEM / PROCESS
macro / regression); common-method-bias check (Harman / marker). E.g. the ChatGPT
paper: "1,326 respondents… Cronbach's alpha and confirmatory factor analysis…
Harman's single-factor… the PROCESS macro approach."

**Conceptual:** state the reasoning approach explicitly — "abductive approach", or
"analogous transfer (Cornelissen & Durand, 2014) from distant disciplines" — in
place of a Methods section.

---

## 08 — RESULTS / FINDINGS

**SLR/review:** organize by **theme/cluster or by RQ**, not paper-by-paper. Present
bibliometric performance (publication trend by phase, top journals/authors/countries)
then science mapping (intellectual/social/conceptual structure), then thematic
clusters. Reframe clusters as a **process model** if possible (e.g., opportunity →
decision-making → performance → education = "the AI-enabled entrepreneurial process").

**Empirical:** report measurement model (loadings, AVE, CR/α, HTMT), then structural
paths with coefficients + significance, then moderation/mediation. Report effect
sizes and CIs, not just p-values (see `academic-paper-guide` for APA 7 stats format).

**Conceptual:** the "results" are the framework/propositions themselves, often
carried in comparison **tables** and a running illustrative case (e.g., a
hypothetical founder threaded through each principle).

---

## 09 — DISCUSSION

**Open by interpreting, not just restating** — the strongest exemplars open by
**challenging a common assumption**, then interpret findings against theory:
> "A widespread prejudice is that intelligent systems will gradually replace humans…
> However, while this may be true to a limited extent…, AI's greatest potential is
> in complementing and enhancing human capabilities." (IJEBR)

**Connect findings to theory with explicit verbs:** "Our findings extend…", "This
challenges…", "This distinction was not explicated in the original formulation of
the framework (Davidsson et al., 2020)" (a paper that *refines* its own theory).

**Reviews** interpret by **crossing clusters against a model**: "We crossed our four
thematic clusters with the five parts of Chalmers et al.'s (2021) schematization to
show the most relevant emerging areas."

---

## 10 — CONTRIBUTIONS (numbered/ordinal; often front-loaded in the intro)

**Verbatim:**
- "The study makes three central contributions. First, we evaluate the potential…
  Second, we push forward the… research studying the role of personality… Third, we
  extend the research on Big Data's usefulness…" (SBE empirical)
- "This paper makes a significant contribution to theory, by introducing a strategic
  HRM framework… through eight distinct dimensions…" (BJM conceptual)

**Do:** number them (First/Second/Finally); separate theoretical vs methodological
contributions; tie each to the gap it closes.

---

## 11 — IMPLICATIONS (name the audience)

Address specific audiences explicitly: **founders/entrepreneurs**, **educators**,
**policymakers**, **investors**. Reviews use a labeled abstract element ("Practical
implications – …for researchers, entrepreneurs and aspiring entrepreneurs…").
Phrase as actionable guidance, e.g. "A practical implication is that while AI will be
a source of entrepreneurial enablement over considerable time, the initial ways of
activating its enabling mechanisms will likely not offer sustained advantages…" (JBVI).

---

## 12 — LIMITATIONS

Conceptual/editorial papers often have **no limitations heading** (or reframe them as
"boundary conditions" to research). SLRs give a brief, honest 2-sentence set and
immediately turn them into opportunity:
> "One limitation… relates to the methods of exclusion during the systematic
> literature review. Another… pertains to the fact that this topic is still quite
> new… However, this study could be useful in setting a benchmark for further
> research…" (IJEBR)
Empirical papers list the standard set (cross-sectional design → no causal inference;
single-source/self-report; sample scope; common-method concerns).

---

## 13 — FUTURE RESEARCH (format varies by genre — pick one)

- **Numbered agenda** (editorial): two long numbered lists (Conceptual 1–11;
  Empirical 1–15), each item a citation-anchored question. (SBE)
- **Theme + bulleted sample RQs** (call): 6 named themes, each = rationale paragraph
  + bulleted "Sample research questions", every question tagged "Why? What
  mechanism?" (ETP) — the richest template.
- **Gap matrix** (SLR): cross thematic clusters × a stage-model in a table (Table 2)
  to expose empty cells, with "Therefore, future research should…" prose per cell. (IJEBR)
- **Directions + methods table** (conceptual): bold-led bullets each paired with a
  recommended method (design science, mixed-methods, Delphi, ethnography,
  agent-based modelling). (BJM)
- **ADO-TCM framework figure** (hybrid review): antecedents-decisions-outcomes ×
  theories-contexts-methods. (Rev Manag Sci)

**Recurring device:** "Therefore, future research should…" / "Future studies are
encouraged to…".

---

## 14 — CONCLUSION (short; reprise + forward-looking close)

Reviews reprise method + the enumerated findings + contribution + limitations, then
a rhetorical close. Conceptual/editorial pieces close on an **optimistic,
forward-looking, stakeholder-benefit note** — a recurring signature:
- "…AI does not become a dangerous enemy, but rather an enabler for entrepreneurs." (IJEBR)
- "…ensuring that the deployment of GAI… is conducted in a manner that is beneficial
  to all stakeholders." (BJM)
- "Let this be a call to action for all entrepreneurship scholars to embrace AI…
  with… rigor, ethics, and foresight." (ETP)

**Opening move:** a broad societal claim ("There is little doubt that Artificial
Intelligence is a gamechanger…").

---

## THE GAP GRAMMAR (reuse verbatim — every one of the 14 uses this shape)

**concession (cited) → void → consequence + fix:**
> "While [AI capability] has received growing attention in [adjacent fields]
> (cite; cite), it has received little scrutiny in [your sub-domain] (cite).
> [Others did the wrong level/method/scope] (cite). Therefore, this study addresses
> this gap by [contribution], [naming your framework/method]."

Stock void verbs: "little has been written", "no prior study", "a dearth of",
"lack of systematization", "has not received much scrutiny", "fail to propose".
For reviews: name and **fault each prior review individually** ("Li et al. (2022)
conducted a bibliometric analysis, but the analysis was limited by its scope and
methodology"). Always hedge novelty ("to the best of our knowledge… although
inspiring contributions have already been produced (e.g., …)").

---

## WHITE SPACES in this literature (defensible gaps if you have data)

The 14 exemplars converge on EE + opportunity theory and are mostly conceptual/review.
Genuinely under-served (per the corpus): **adoption theories (TAM/UTAUT)**,
**RBV / dynamic-capabilities empirical tests**, and **firm-level empirical PLS-SEM
studies**. An empirical adoption study using one of these is a real contribution,
not off-domain — but justify it explicitly since it departs from the field's EE default.

---

## Corpus (the 14 exemplars, by genre)
Empirical: ChatGPT-adoption→intention (Equilibrium 2024); Big-data entrepreneurial
regions (SBE 2019). SLR/review: AI-as-enabler SLR (IJEBR 2022); hybrid review (Rev
Manag Sci 2025). Conceptual: AI & venture creation (ETP 2020); GAI-HRM framework
(BJM 2024); design-science/effectuation (SBE 2019); social-signal processing (SBE
2019); AI-as-external-enabler via ChatGPT (JBVI 2023); Lean-startup + AI predictions
(IJEBR 2022). Editorial/call: "A new era has begun" (SBE 2019); "A call for research"
(ETP 2024). Methods-showcase: data-science for entrepreneurship (SBE 2019).
Education essay: entrepreneurship education in the generative-AI era (Entr. Educ. 2023).
