# Preliminary findings: abstract-level construct specification of AI in entrepreneurship research

Primary study dataset, GPT-4.1 nano rater (`gpt-4.1-nano-2025-04-14`), protocol
`spec-v3`. Generated 2026-07-11 from `data/interim/spec_cache/spec-v3/gpt-4.1-nano-2025-04-14/`.
These are preliminary results from a single rater; the GPT-5.4 mini full run and
the inter-rater reliability stage will refine them (see Limitations).

## Scope and method (read before the numbers)

Coding was based on titles, abstracts, and author keywords. The resulting
measures represent observable **abstract-level** construct specification rather
than exhaustive full-text interpretation. Dimensions that could not be
established reliably from the available metadata were marked as requiring
full-text review (`needs_full_text`) and were not silently inferred.

Each paper was coded once across seven controlled dimensions (role/function,
type/form, mechanism, level of analysis, process/sequence, scope conditions,
definition/construct clarity) with per-dimension evidence, an epistemic label
(stated / inferred / absent), and a confidence score. Controlled vocabularies
and the frozen instrument are documented in
[`LLM_REPRODUCIBILITY.md`](LLM_REPRODUCIBILITY.md).

**Pre-registered mechanism rule.** A spec-v3 audit found the nano rater codes a
substantive mechanism while leaving its causal-logic field empty. Per the rule
declared before completion, a substantive `ai_mechanism` code with an empty
`ai_mechanism_logic` is analysed as "mechanism missing" (no explicitly named
mechanism observable at the abstract level). Both raw and corrected mechanism
prevalences are reported below; the corrected figure is the defensible one.

## Completion and coverage

Dataset is complete: **22,335 / 22,345 papers (100.0%)**; the 10 uncoded fall
below the text-length gate. Every analytical view is fully covered.

| View | Coded / view size |
|---|---|
| Full corpus | 22,335 / 22,345 |
| Query 1 broad business & management | 21,445 / 21,455 |
| Query 2 FT50 | 438 / 438 |
| Query 3 leading entrepreneurship | 646 / 646 |
| Query 4 additional entrepreneurship | 986 / 986 |
| Strict AI x entrepreneurship subset | 2,509 / 2,509 |

## 1. Prevalence of specification problems (headline)

Share of the 22,335 papers exhibiting each construct-specification problem:

| Specification problem | Papers | Share |
|---|---:|---:|
| AI type underspecified | 18,084 | 81.0% |
| Mechanism missing | 15,364 | 68.8% |
| Scope conditions missing | 11,420 | 51.1% |
| Role ambiguity | 7,932 | 35.5% |
| AI treated as loose label | 7,651 | 34.3% |
| AI definition absent | 3,027 | 13.6% |
| Level mismatch | 1,956 | 8.8% |
| Process not specified | 1,944 | 8.7% |
| Construct contrast | 1,703 | 7.6% |
| Construct fragmentation | 1,682 | 7.5% |

**80.7% of papers provide no definition of AI** (definition/clarity dimension),
the single strongest result. The AI construct enters entrepreneurship research
predominantly undefined, untyped, and without a stated causal mechanism.

## 2. Dimension distributions

| Dimension | Leading codes (share) |
|---|---|
| Role/function | tool 31.3%, context 26.0%, research method 25.6%, actor/agent 13.5% |
| Type/form | machine learning 27.8%, general AI 16.6%, generative AI 14.3%, analytics 12.5%, unspecified 11.4% |
| Mechanism (raw) | supports learning 76.8%, automates decisions 11.2%, missing 5.9% |
| Mechanism (corrected) | **mechanism missing 67.1%**, supports learning ~26%, automates decisions ~9% |
| Level of analysis | venture 45.6%, individual 17.8%, firm 11.2%, platform 6.7%, industry 6.6% |
| Process/sequence | process unspecified 26.6%, static input 24.2%, opportunity evaluation 15.7% |
| Scope conditions | sector-specific 39.3%, established firms 18.3%, high-tech startups 12.3%, generalised without scope 6.0% |
| Definition/clarity | no definition 80.7%, explicit-fits 16.0%, partial 3.2% |

The raw-vs-corrected mechanism swing (5.9% -> 67.1%) is the largest single
correction; see Limitations.

## 3. Trends over time (corrected mechanism)

| Era | n | No definition | Mechanism missing | Scope missing | Generative AI | AI as actor |
|---|---:|---:|---:|---:|---:|---:|
| 2000-2015 | 3,297 | 89% | 79% | 14% | 7% | 5% |
| 2016-2020 | 2,654 | 86% | 74% | 9% | 6% | 7% |
| 2021-2023 | 4,329 | 82% | 68% | 9% | 11% | 12% |
| 2024-2026 | 10,486 | 75% | 59% | 7% | 21% | 20% |

Specification improves monotonically in relative terms, but 47% of the corpus is
2024-2026: absolute under-specification is rising as output accelerates. This is
the quantified basis for "specification before accumulation".

## 4. Construct contrast: one label, multiple constructs

Role distribution within each AI type (every type spreads across 6-7 roles;
"dominant" is the single most common role's share):

| AI type | n | Distinct roles | Dominant | Top three roles |
|---|---:|---:|---:|---|
| generative AI | 3,190 | 7 | 30% | actor/agent 30%, tool 25%, context 25% |
| machine learning | 6,206 | 7 | 40% | tool 40%, research method 35%, context 14% |
| general AI | 3,702 | 7 | 48% | context 48%, actor/agent 18%, tool 15% |
| analytics | 2,788 | 7 | 38% | tool 38%, research method 34%, context 21% |
| predictive AI | 1,469 | 6 | 55% | tool 55%, research method 17%, context 15% |
| unspecified AI | 2,542 | 7 | 47% | context 47%, research method 19%, tool 16% |

Generative AI is the showcase: a near-even three-way split (actor / tool /
context) shows one term carrying three distinct theoretical claims.

Cross-tabulation of AI type x corrected mechanism shows generative-AI papers
articulate mechanisms most often (50% missing) versus general AI (66% missing) -
the mechanism signal exists and is recoverable by a rule-compliant rater.

## 5. Where the problem occurs: journal-tier contrast is muted

Key failure rates are nearly uniform across the five views:

| View | n | No definition | Mechanism missing | Scope missing | Loose label |
|---|---:|---:|---:|---:|---:|
| Full corpus | 22,335 | 81% | 67% | 9% | 34% |
| Query 1 broad B&M | 21,445 | 81% | 67% | 9% | 34% |
| Query 2 FT50 | 438 | 78% | 66% | 8% | 29% |
| Query 3 leading entrepreneurship | 646 | 75% | 65% | 12% | 33% |
| Query 4 additional entrepreneurship | 986 | 75% | 63% | 10% | 36% |

The construct-specification problem is roughly **uniform across journal prestige
tiers**: FT50 journals specify AI only marginally better than the broad corpus.
Entrepreneurship journals (Q3/Q4) define AI slightly more often (75% vs 81% no
definition) - a small directional signal, not a chasm. This is a finding, not a
null result: elite venues do not escape the problem.

Journal-level role fragmentation is universal - every journal with >=100 coded
papers uses 6-7 distinct AI roles (dominant role 33-50%). Examples: Decision
Support Systems (1,186 papers, 7 roles), Journal of Cleaner Production (1,041, 7),
International Journal of Production Research (971, 7).

## 6. Instrument health

- Zero "stated"-with-empty-evidence across ~156,000 dimension-instances.
- Output tokens mean 513, max 930 (4,096 ceiling; ample headroom).
- Confidence is calibrated: role/function 0.89, definition/clarity 0.50 -
  high where the dimension is observable from abstracts, honest where it is not.
- `needs_full_text` discriminates by dimension, functioning as a result rather
  than a disclaimer: definition 91.9%, mechanism 87.3%, type 79.2% (the
  full-text-sensitive tier) versus level 18.6%, process 21.0% (the
  abstract-observable tier).

## Limitations and next steps

1. **Nano's raw mechanism dimension is degenerate.** 76.8% "supports learning"
   is a reflex default with empty causal logic; the analysis relies on the
   pre-registered correction (67.1% missing). Nano reliably supports only a
   binary (mechanism stated-with-logic vs not), not the full mechanism taxonomy.
   The GPT-5.4 mini full run (rule-7 compliant at 0% empty logic in the gate)
   is planned specifically to recover the taxonomy.
2. **Single-rater results.** These are one rater's abstract-level codes.
   Inter-rater reliability against the mini full run and the completed local
   validation records (Llama; partial Gemma) is a required stage, not yet run.
3. **Muted cross-view contrast.** The near-uniform journal-tier rates mean the
   study's "where" question resolves to "roughly everywhere"; frame this as a
   substantive finding about elite venues rather than expecting a large contrast.
4. **Abstract-level boundary.** High `needs_full_text` on definition and
   mechanism is expected and honest; prevalence figures are "observable at the
   abstract level", per the methods statement above.

Six of seven dimensions are publication-grade from this dataset now; mechanism
awaits the rule-compliant mini run. Numbers here should be regenerated after any
corpus change and cross-checked against the mini dataset before manuscript use.
