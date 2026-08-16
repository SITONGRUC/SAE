# Reproduction notes — what does not match, and what has been ruled out

Running record of the gap between this replication and *Exploratory Causal Inference in
SAEnce* (ICLR 2026). Written to stop the same hypotheses being re-tested. Everything here is
from the run of 2026-08-15.

Pipeline: `encode_siglip.py` → `train_sae.py` → `figure5.py` → `plot_figure5.py`.

---

## 0. The headline

**Figure 3's claim reproduces. Figure 5 does not.** The baselines fail exactly as the paper
says they will, but NES does not recover the ground-truth codes, and the SAE that feeds it is
weaker than the paper's.

---

## 1. What reproduces

**The paradox, cleanly.** Mean codes selected out of 9,216, averaged over ATE and 10 seeds:

| method | n=30 | n=250 | n=500 | n=5000 |
|---|---|---|---|---|
| NES | 0.6 | 5.8 | 7.1 | **10.1** |
| Bonferroni | 0.6 | 30.5 | 99.5 | 1,266 |
| FDR | 1.0 | 140 | 461 | 3,048 |
| t-test | 124 | 875 | 1,384 | **3,700** |

At n=5000 the t-test calls 40% of the dictionary significant; recall → 1.0 while precision → 0.
NES holds at ~10 codes across the whole grid. The mechanism the paper describes is visibly
present in the real pipeline, not just in the synthetic Figure 3 setup.

**Per-patch training fixes dictionary starvation.** Dead codes went 8,427/9,216 → **0/9,216**.

**SigLIP is sufficient for Eyeglasses.** Logistic regression on pooled features, fit on val,
evaluated on test: precision 0.9846 at the paper's own recall of 0.9747, against their 0.9712.

## 2. What does not reproduce

| quantity | ours | paper |
|---|---|---|
| best F1, Eyeglasses | **0.578** (code 1914) | 0.748 (neuron 6051) |
| best F1, Wearing_Hat | **0.478** (code 2202) | 0.841 (neuron 38) |
| NES hit rate on G | **0 / 2** everywhere | recovers G |
| sufficiency, Wearing_Hat | P 0.8505 / R 0.8272 | P 0.9639 / R 0.9947 |

Best F1 is below the 0.6 checkpoint set in the replication plan. Precision is ~0 for every method in
Figure 5, so the figure carries no signal.

## 3. Hypotheses tested

### 3.1 REFUTED — the concept is split across near-tied codes

If a concept were spread over many codes, `G` would be an arbitrary argmax among near-ties and
NES could find the concept while missing the specific code. Checked: only **1** code for
Eyeglasses and **2** for Wearing_Hat sit within 90% of the best F1. `G` is well separated.

### 3.2 REAL BUT NOT THE CAUSE — code scale dominates NES selection

NES's second criterion is the largest `|τ̂|`, unstandardised. After mean-pooling, code
magnitudes span three orders of magnitude:

| code | mean | τ̂ | p |
|---|---|---|---|
| 8352 | 4.64 | −3.92 | ~0 |
| 5114 | 25.10 | −3.62 | ~0 |
| **1914 (Eyeglasses)** | **0.032** | **0.026** | 1e-34 |

Code 1914 is overwhelmingly significant yet ranks **582nd of 3,134** significant codes by
`|τ̂|`. The effect is real and worth knowing about. But z-scoring every code before NES changes
*which* codes get picked and still returns **0/2**, so scale is not what breaks Figure 5.

### 3.3 CONSISTENT, NOT ISOLATED — entangled codes outrank clean ones

The codes NES keeps selecting (6795, 8201) score only 0.13 / 0.10 F1 on either attribute. A
code encoding *both* glasses and hats collects treatment signal from both outcomes at once, so
it beats either principal code on effect size while being mediocre on both. This fits every
observation so far and is the paper's own entanglement mechanism — our SAE just has it worse.
Not yet demonstrated directly.

### 3.4 NOT A BUG — pooled codes are 9% dense

`Z` has 9.22% of entries above zero: 849 active codes per image, not 20. That is the arithmetic
of the paper's own recipe — 196 patches × top-20 each, unioned by mean pooling, then thresholded
at `Z_j > 0`. Code 1914 fires on 9.48% of images against a 6.46% Eyeglasses prevalence, which is
where its precision of 0.486 comes from. Faithful to the paper; not the explanation.

## 4. Reconstruction and alignment move in opposite directions

**Training longer makes alignment worse, monotonically.** Per-epoch F1 from the run with a
pre-encoder bias (`logs/train_log_bias.txt`):

| epoch | MSE | F1 Eyeglasses | F1 Wearing_Hat | dead |
|---|---|---|---|---|
| 1 | 2.879 | **0.722** | **0.667** | 0 |
| 2 | 2.388 | 0.658 | 0.594 | 0 |
| 3 | 2.143 | 0.575 | 0.521 | 0 |
| 4 | 1.999 | 0.535 | 0.458 | 0 |
| 5 | 1.899 | 0.513 | 0.366 | 0 |
| 6 | 1.820 | 0.457 | 0.315 | 0 |

Every epoch improves reconstruction and degrades alignment, with no dead codes at any point.
Stopped at epoch 6 — the trend is monotone and the remaining 14 epochs cost 48 minutes for no
new information.

### 4.1 REFUTED — the missing pre-encoder bias

The vendored `TopKEncoder` has no bias term, and the shared mean of the SigLIP features is 45%
of a typical vector's norm. On the no-bias run this showed up exactly as predicted: **51 codes
fired on 100% of images**, and the 274 codes firing on more than half of images carried **66%
of all activation mass**. Codes 5114 (mean 27.2), 478 (10.3) and 8352 (7.2) all fire on 100% of
images — mean carriers, not concepts. NES kept selecting from this group.

Adding a bias initialised to the data mean fixed the reconstruction (MSE at epoch 1: 3.46 →
2.88) and raised F1 at epoch 1 (0.722 / 0.667 vs paper 0.748 / 0.841 — nearly there). **But the
degradation is unchanged.** The mean-carrier problem was real and is now fixed; it was not what
breaks the replication.

### 4.2 The remaining explanation — pooled codes are not sparse enough

The paper calls a code active when `Z_j > 0`, and `Z` is the mean over 196 patches. Per-patch
training with k=20 therefore yields **849 active codes per image**, not 20 — the union of 196
separate top-20 sets. Picking one code out of 849 to predict a 6.5%-prevalence attribute caps
precision, which is where F1 0.486 comes from.

Training on FM-pooled vectors instead gives exactly **20** non-zero entries per image, two
orders of magnitude more selective. That path was rejected earlier because it killed 91% of the
dictionary — **but that test ran without the bias**. With mean carriers hogging the top-20 slots,
most codes could never be selected and starved. Once the competition happens on centred signal,
the starvation may not recur.

| | dead codes | best F1 |
|---|---|---|
| pooled, no bias | 91% | 0.32 / 0.27 |
| per-patch, no bias | 0 | 0.578 / 0.478 |
| per-patch, bias | 0 | 0.722 / 0.667 at epoch 1, falling |
| pooled, bias | never run — superseded by §5-6 | — |

## 5. RESOLVED — the evaluation prevalence was wrong, not the SAE

The decisive detail is in the Evaluation paragraph's own notation:

> *"we induce predictions ŷ_ik := I{Z_j(X_i) > 0} and compute the F1-score of
> **{ŷ_ik}_{i=1}^n** against the ground-truth labels {y_ik}_{i=1}^n"*

The index runs to **n — the simulated experiment's sample size** (the DGP paragraph's
`n ≪ 200k`), not the 19,962-image raw test split. The paper computes F1 and identifies G **on
the resampled experimental sample**, where prevalence is 0.35–0.5. Every F1 in this note's §2
was computed on raw test prevalence (4–6%), so none of them were comparable to 0.748 / 0.841.

Verified zero-cost with the existing k=20 codes on a simulated sample (n=5000, ATE=0.5):
best Eyeglasses code F1 **0.795** (paper 0.748), best Wearing_Hat **0.848** (paper 0.841).
The k=20 SAE had matched the paper all along. Consequences:

- G identified at raw prevalence ({1914, 2202}) is **not** the experimental-prevalence
  argmax; NES "missing G" was partly scoring against the wrong answer key.
- NES's picks (6795, 8201) score 0.65–0.70 on the simulated sample — mid-tier aligned,
  not garbage. §3.3's "entangled codes" reading overstated the problem.
- The 0.6-checkpoint alarms, and the whole bias/k=5 escalation they triggered, were set off
  by a metric artifact. (The mean-carrier fix of §4.1 remains real and worth keeping.)
- The "degradation" of §4 was measured in the wrong metric; at balanced prevalence the k=5
  run (§6) holds a flat 0.94+ across all 20 epochs.

`train_sae.py` now logs both `F1raw` (trend only) and `F1bal` (comparable to the paper), and
`figure5.py` must identify G on the simulated sample rather than loading the raw argmax.

## 6. The k=5 run — checkpoint cleared

Table-1-faithful run (`--tag k5`): k=5, per-patch, pre-encoder bias, 20 epochs. Full-test
results:

| | ours (balanced) | paper | ours (raw) |
|---|---|---|---|
| Eyeglasses | **0.9378** (code 8654) | 0.748 | 0.7406 |
| Wearing_Hat | **0.9360** (code 6795) | 0.841 | 0.6707 |

Balanced-prevalence checkpoint: worst = 0.936, clears 0.6 decisively. Both attributes now
**exceed** the paper's reported alignment.

Dynamics worth recording:

- **Dead codes grow then plateau** under k=5: 600 → ~4,700 during training (measured on a
  5,000-image eval subset), but only 2,539 (27.5%) fire nowhere on the full 19,962-image
  test set. Winner-take-all is real but self-limiting here; the paper says nothing about
  dead-code handling (no aux loss, no resampling), and none was used.
- **F1bal is flat across epochs** (0.972 → 0.935 over 20 epochs) — the catastrophic-looking
  decline of §4 does not exist in the correct metric. The mild residual drift is consistent
  with slow feature splitting but never threatens the checkpoint.
- k=5 raw-prevalence F1 (0.741) lands nearly on the paper's number (0.748) by coincidence —
  a reminder that two incomparable metrics can look deceptively similar.

Remaining known deviations (documented in `train_sae.py`'s docstring, none blocking): tied
encoder/decoder with unit-norm rows and a single pre-encoder bias versus Eq. (3)'s untied
E, D, b_e, b_d; the k=5 (Table 1) vs k=20 (Appendix E.1/E.2.2) contradiction is unresolved in
the paper itself — both readings now have runs on record, and both clear the corrected
checkpoint.

## 7. Figure 5 with per-sample G — the paradox reproduces, two gaps remain

Grid rerun with `codes_test_k5.npy` and G identified on each simulated sample (§5 protocol).
Slices matching the paper's figure layout:

**Across N at ATE=0.5** (paper's top row):

| n | NES P/R | Bonferroni P/R | FDR P/R | t-test P/R | top-k P/R |
|---|---|---|---|---|---|
| 30 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.03 / 0.5 | 0.40 / 0.4 |
| 250 | **0.58 / 0.5** | 0.13 / 0.9 | 0.04 / 1.0 | 0.01 / 1.0 | 0.50 / 0.5 |
| 500 | **0.41 / 0.6** | 0.04 / 1.0 | 0.01 / 1.0 | 0.00 / 1.0 | 0.60 / 0.6 |
| 5000 | **0.13 / 0.7** | 0.00 / 1.0 | 0.00 / 1.0 | 0.00 / 1.0 | 0.50 / 0.5 |

**Qualitatively reproduced:** every baseline's recall → 1 while its precision collapses to ~0
(the significance collapse), NES holds the best precision of any statistical procedure at every
n ≥ 250, the paradox is absent at n=30 exactly as the paper notes, and G is stable at large n
(1-2 distinct codes across 90 runs) while noisy at n=30 (7 distinct — the protocol's own
small-sample behaviour). NES returns the empty set at ATE=0, which is correct null behaviour.

**Two quantitative gaps against the paper's bars:**

1. **NES recall plateaus at 0.5-0.75 instead of reaching 1.0.** The paper's NES finds both
   effects "most of the time" at n ≥ 250; ours misses one of the two G codes in 30-50% of
   runs. Consistent with §3.2's scale problem: selection is by largest raw `|τ̂|` among
   significant codes (reference implementation's criterion), and code scales after mean
   pooling span orders of magnitude, so an entangled high-magnitude code can be picked ahead
   of the second principal code.
2. **At n=5000 NES over-selects** (hits our 12-code `max_cells` cap; precision decays to
   ~0.13 at high ATE vs the paper's ~0.5). Suspect protocol difference: the shipped reference
   keeps strata with a single unit per arm, where `var(ddof=1)` = NaN propagates and kills the
   whole neuron (a known gotcha in the shipped reference code) — an accidental conservativeness that stops NES earlier.
   Our cleaner ≥2-per-arm guard keeps more neurons testable, so NES keeps finding significant
   leakage at n=5000. Untested.

Both gaps are about NES's selection/stopping details, not the SAE and not the metric — the
representation and evaluation questions are closed.

## 5. Deviations and unspecified details

**No intentional deviations.** The protocol matches Appendix D.1, which states: *"We follow the
authors' official train/val/test split, and we employ the validation data for training SAEs and
the test data to interpret them."* The 162,770-image train split is unused here, as in the
paper. Training on it would be a deviation, not a fix — noted because it was briefly and
wrongly proposed as one.

Things the paper leaves open, where a choice had to be made:

| detail | paper | here |
|---|---|---|
| SigLIP checkpoint | cites Zhai et al. 2023 only | `siglip-base-patch16-224`, inferred from d=768 + 196 patches |
| input normalization | unspecified | none — SAE literature usually normalizes; untested A/B |
| dictionary init | unspecified | `randn`, rows not unit-normed at init |
| `k` in the top-k baseline | unspecified | 2, matching \|G\| |
| split for the sufficiency check | unspecified | fit val, evaluate test |
| prevalence for the F1 spectrum | unspecified | raw CelebA test (4.2% / 6.5%) |

Two of these are worth flagging as possible sources of the gap:

- **Input normalization** is the one unspecified knob most likely to matter for a top-k SAE,
  and it is free to test.
- **Prevalence for F1.** The same linear probe scores F1 0.8387 on raw CelebA test and 0.9026
  on a balanced resample. If the paper computed its F1 on the resampled experimental sample
  (where P(Y) ≈ 0.5) rather than raw CelebA, part of the 0.748 / 0.841 gap is a base-rate
  difference rather than a worse dictionary. Suggestive: the paper's single neuron 38 scores
  0.841, essentially matching what our *entire 768-d linear probe* gets on raw CelebA.

**Unaccounted for.** D.1 also says *"labels have been doubled (we pass from Beard to Has Beard
and Has notBeard)"* — 40 attributes become 80. This has no effect on SAE training, but it
changes the F1 labelling step, since a code aligned with the *negation* of an attribute becomes
scorable. Not implemented here.


---

## Appendix — step 2 notes: SigLIP encoding and the sufficiency check

Merged from the former `step2_encoding.md` (2026-08-16); content unchanged apart from
heading levels.

Steps 2 / 2b of the pipeline. Run:

```bash
cd reproduce_experiment1
python encode_siglip.py --limit 8    # smoke test: shapes + fp16 vs fp32 deviation
python encode_siglip.py              # full run, val + test
python sufficiency_check.py
```

Use the Anaconda `python`, not `/usr/bin/python3` — the system interpreter has no numpy.

---

### 1. The model

The paper cites only "SigLIP (Zhai et al., 2023)" and never names a checkpoint. It does give
`d=768` and `196 patches`, and those two numbers pin it:

```
224 / 16 = 14 patches per side     14**2 = 196 tokens     hidden size 768
```

| checkpoint | d | patches | |
|---|---|---|---|
| **base-patch16-224** | **768** | **196** | ✅ |
| base-patch16-256 | 768 | 256 | patch count wrong |
| base-patch16-384 | 768 | 576 | patch count wrong |
| large-patch16-256 | 1024 | 256 | both wrong |
| so400m-patch14-384 | 1152 | 729 | both wrong |

`d=768` fixes the size class, `196` fixes the resolution. Unique among the SigLIP releases.

**This is inferred, not quoted** — recorded because if it is wrong, the features are wrong, and
the symptom will appear much later as SAE F1 scores that miss the paper. One residual
ambiguity: SigLIP 2's `siglip2-base-patch16-224` has identical geometry. The citation year
(2023) is what rules it out, not the arithmetic.

**Benign warning on load.** `SiglipVisionModel` loading this checkpoint reports ~200
`UNEXPECTED text_model.*` keys. The checkpoint holds both towers and we take only the vision
one, so the text weights are discarded. Vision weights are all present — `MISSING` would be
the problem, `UNEXPECTED` is not.

### 2. What gets stored

Per-patch features are written and the pooled view is derived from them:

| file | shape | size |
|---|---|---|
| `features/patches_val.f16` | (19867, 196, 768) fp16 | 5.98 GB |
| `features/patches_test.f16` | (19962, 196, 768) fp16 | 6.01 GB |
| `features/pooled_{split}.npy` | (n, 768) fp16 | 31 MB each |
| `features/index_{split}.parquet` | file order | tiny |

Appendix D.1 says patches are averaged on the FM side *and* that codes are mean-pooled after
the SAE, which only both make sense if the SAE runs per patch. Keeping patches means one
encoding pass serves both readings — pooling is a mean over axis 1. Given the encoding cost
(~35 min, and 18 of those were downloading weights at 733 kB/s), not having to redo it is
worth the 12 GB.

Verified after the run: shapes correct, **row order matches the metadata exactly** (otherwise
attribute joins would silently misalign), all values finite — fp16 did not overflow, max
absolute value ~52 against a 65504 ceiling — and `pooled == mean(patches)` to within 0.008,
which is fp16 rounding.

### 3. The sufficiency check

Vanilla logistic regression on the pooled features, fit on val, evaluated on test. The paper
does not say which split it used.

| attribute | prevalence | precision | recall | F1 | AUC | paper P / R |
|---|---|---|---|---|---|---|
| Eyeglasses | 0.065 | 0.9593 | 0.9503 | 0.9548 | 0.9963 | 0.9712 / 0.9747 |
| Wearing_Hat | 0.042 | 0.8505 | 0.8272 | 0.8387 | **0.9899** | 0.9639 / 0.9947 |

**Eyeglasses reproduces.** At the paper's own recall of 0.9747 our precision is 0.9846 against
their 0.9712, so the operating points agree and ours is marginally better.

**Wearing_Hat does not, and it is not a threshold artefact.** Sweeping every threshold:

| protocol | Wearing_Hat | Eyeglasses |
|---|---|---|
| out-of-sample, threshold 0.5 | P 0.8505 / R 0.8272 | P 0.9593 / R 0.9503 |
| out-of-sample, balanced class weights | P 0.8148 / R 0.8546 | P 0.9408 / R 0.9620 |
| balanced prevalence (0.5), threshold 0.5 | P 0.9933 / R 0.8272 | P 0.9973 / R 0.9503 |
| balanced, at the paper's recall | **P 0.7410** | P 0.9846 |
| **in-sample (fit = eval = test)** | **P 1.0000 / R 1.0000** | **P 1.0000 / R 1.0000** |

The paper's hat operating point is unreachable out-of-sample at any threshold. But in-sample
both attributes separate perfectly, and the paper reports neither 1.0 nor our numbers — its
values sit between the two. So its protocol differs from ours in some way the text does not
specify: plausibly fitting on the unused 162k train split, or evaluating on the resampled
experimental sample rather than raw CelebA.

**This does not block step 3.** What Assumption A.1 requires is that the attribute information
survives into the representation, and AUC 0.9899 says it does — the ranking is nearly perfect
even though the calibrated operating point is not. Hats are simply harder than glasses here:
4.2% prevalence, and hats vary far more in shape and colour than eyeglasses do.

### 4. The finding that matters for step 3

**F1 depends strongly on prevalence, so the paper's F1 = 0.841 checkpoint is only meaningful
against a matched prevalence.** Our linear probe on Wearing_Hat scores

- F1 **0.8387** on raw CelebA test (4.2% positives)
- F1 **0.9026** on a balanced resample (50% positives)

from the exact same classifier. Same model, same features, different denominator.

Two consequences:

1. The paper's SAE neuron 38 scores F1 = 0.841 on Wearing_Hat — essentially identical to what
   our *entire 768-dimensional linear probe* gets on raw CelebA. A single SAE code matching a
   full linear probe is suspicious, and is more consistent with the paper evaluating on the
   resampled experimental sample (where P(Y) ≈ 0.5) than on raw CelebA.
2. So when step 3 checks whether the best code clears F1 = 0.6, **evaluate on both prevalences
   before concluding the SAE failed.** A code that looks like 0.55 on raw CelebA could be
   comfortably above 0.6 on the experimental sample. Blaming the SAE for what is a base-rate
   difference would send the work down the per-patch retraining path for no reason.
