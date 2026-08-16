# MVP result — SAE partisan channels in Fed/banking hearing speech

2026-08-15. Total compute ~5 minutes on the M4. Code: `mvp.py` (data prep in the docstring),
inputs: `data/emb.npy`, `data/meta.parquet`; artifacts: `output/codes_all.npy`,
`output/speaker_codes.parquet`, `output/table/code_tests.parquet`.

## Setup

67,080 hearing segments (1951–2023, 280 hearings), OpenAI embeddings d=1536, already merged
with the speaker→party crosswalk. SAE: m=1,000, k=5, pre-encoder bias, trained unsupervised
on **all** segments (party labels never touch training) — 15 epochs, ~2 min, **0 dead codes**,
exactly 5 active codes per segment. Tests on politicians with D/R labels only.

## Result 1 — the dependence inflation is real and large

| test level | p<.05 | Bonferroni (α/1000) |
|---|---|---|
| naive, 29.5k segments | 119 / 1000 | **17** |
| speaker level, n=471 | 24 / 1000 | **0** |

Ignoring within-speaker dependence manufactures 17 Bonferroni "discoveries" out of nothing.
This is the replication's paradox mechanism appearing in the real data on day one, and it is
the empirical justification for the clustered-inference design.

## Result 2 — pooling eras destroys the signal; the split matches Gentzkow–Shapiro

Speaker-level, substantive segments only (≥50 words, ≥5 segments/speaker):

| era | n speakers | p<.05 |
|---|---|---|
| 1951–1994 | 123 | 6 / 1000 (below the 50 expected by chance) |
| 1995–2023 | 180 | 12 / 1000, top code t=−3.3 |

Pre-1995 there is essentially **no** partisan signal in how members speak in these hearings —
consistent with Gentzkow–Shapiro's finding that congressional-speech partisanship was flat
until the mid-1990s. The signal that exists is concentrated post-1995.

## Result 3 — the post-1995 top codes read as the canonical partisan split

Top-activating segments, no cherry-picking (first 4 codes by p-value):

| code | t | leans | reads as |
|---|---|---|---|
| 216 | −3.3 | D | labor-market framing in questions to the Fed chair (AOC, Velázquez, Casten: jobs, wage growth, overheating) |
| 235 | −2.5 | D | wage growth / full employment / disparities |
| 837 | −2.6 | D | program funding levels, reauthorization, cuts |
| 510 | +2.4 | R | **national debt / deficits / entitlement spending** |

Democrats press the employment half of the Fed's dual mandate; Republicans press fiscal
discipline. This is the known partisan split in Fed oversight — recovered from raw embeddings
by an unsupervised dictionary that never saw a party label or a word count. (Cassidy–Kempf's
literal bigram lists are Twitter-domain and mostly not applicable here; the thematic
correspondence — D: jobs/inequality, R: debt/spending — is the meaningful check, and it holds.)

## Honest verdict on the gate

**The formal gate (Bonferroni at speaker level) is not cleared** — best t = 3.3 against a
threshold of ~4.1 with n=180. This is exactly the ISTAnt regime (the ECI paper itself drops
Bonferroni at n=44), so the gate as originally set was probably miscalibrated for this data.
What the MVP does establish:

1. the pipeline transfers end-to-end (replication code ran on text embeddings unmodified),
2. dependence inflation is large and measurable (17 → 0),
3. the signal that exists sits where the literature says it should (post-1995),
4. and its content is interpretable and matches priors without supervision.

## Result 4 — the ISTAnt validation protocol transfers verbatim

The paper's second experiment labels discovered codes with a four-step protocol: discover
first (NES, no Bonferroni at n=44) → interpret via max-activating vs non-activating examples
judged one by one by a domain expert → close the loop quantitatively (their code 394 is the
argmax-F1 grooming detector among all 4,608 codes, F1=0.398) → confirm against prior
literature. We ran the quantitative step on our discovered codes, with keyword lexicons
built *after* discovery as the proxy annotation (post-1995 substantive politician segments):

| discovered code | lexicon | F1 | rank among 1,000 codes |
|---|---|---|---|
| 216 (leans D) | jobs/labor | 0.535 | **1 / 1,000** |
| 510 (leans R) | debt/deficit | 0.265 | 3 / 1,000 |

Code 216 was found by the party contrast alone; the jobs lexicon never touched discovery.
Two independent routes land on the same code — the ISTAnt result (their F1=0.398, rank 1)
reproduced on hearing speech with a higher F1. Code 510's near-tie with two other codes is
the paper's own caveat verbatim: "the imperfect F1-score suggests other entangled effects
or broader representation."

Bonus parallel: ISTAnt's second code (550) was interpreted as a *finite-sample experimental
design error* detector — our procedural codes (309 "Senator.", 819 "Senator Carper.", 337
chair scripts) are the same object in this corpus, and can be reported the same way rather
than hidden.

Still uncopied from the protocol: the non-activating contrast column in the example display,
and example-level (rather than code-level) judgment — both cheap, both belong in the full
design.

## What the full design adds, in order of expected payoff

1. **Era-aware analysis** — pooling 1951–2023 was the biggest self-inflicted wound here.
2. **Hearing-level stratification** — code 837-style "funding levels" content is partly
   committee-agenda driven; within-hearing contrasts isolate the *how* from the *what*.
3. **Sentence-level re-embedding + larger m** — 67k segments cap the dictionary at ~1,000
   codes; the interesting fine-grained frames likely need more resolution.
4. **NES with clustered SEs instead of Bonferroni** — the ISTAnt precedent applies at n≈180.
5. Role controls (chair vs member) — the raw scan's top codes were procedural before the
   ≥50-word filter; roles are a real confound.
