# Experiment 1 replication — *Exploratory Causal Inference in SAEnce* (ICLR 2026)

Self-contained replication of the paper's CelebA experiment (Section 6.1 / Appendix D.1):
CelebA → SigLIP → top-k SAE → simulated RCT → NES vs. baselines (Figure 5), plus the
F1-spectrum plot (Figure 8). Everything needed to rerun lives in this folder; nothing
imports from outside it.

**Verdict.** The SAE reproduces and exceeds the paper's code–concept alignment
(best single-code F1 at experimental prevalence: **0.938** Eyeglasses / **0.936**
Wearing_Hat vs. the paper's 0.748 / 0.841). Figure 5's qualitative claim reproduces:
every baseline's recall → 1 while precision → 0 as n grows (the significance-collapse
paradox), while NES keeps the best precision of any statistical procedure at every
n ≥ 250. Two quantitative gaps against the paper's bars remain, both localized to NES's
selection/stopping rules — see `reproduce_note.md` §7.

`reproduce_note.md` is the debugging history: every hypothesis tested and its verdict.
**Read it before changing anything in the SAE** — the expensive lessons (evaluation
prevalence, pre-encoder bias, per-patch training, the val/test-swapping HF mirror) are
all recorded there so they don't get re-learned.

## Layout

| path | role |
|---|---|
| `data/fetch_celeba.py` | step 0a — download canonical CelebA from Google Drive, md5-checked |
| `data/build_splits.py` | step 0b — build `data/meta/celeba_meta.parquet`, hard-assert the official split sizes |
| `data/step1_download.md` | notes on the download, mirror pitfalls, and the joint-cell imbalance |
| `encode_siglip.py` | step 1 — SigLIP patch features for val+test (`features/*.f16` memmaps) |
| `sufficiency_check.py` | step 1b (optional) — logistic probe on pooled features; separates "encoding broken" from "SAE broken" |
| `train_sae.py` | step 2 — tied top-k SAE (768 → 9,216, per-patch, pre-encoder bias) + F1 labelling of codes |
| `nes.py` | Neural Effect Search + baselines (t-test, Bonferroni, FDR, top-k); pure numpy |
| `figure5.py` | step 3 — DGP + 360-cell grid (n × ATE × seed), writes `output/table/figure5.parquet` |
| `plot_figure5.py` | step 4 — renders `output/picture/figure5.png` and `figure8.png` |
| `reproduce_note.md` | **the debugging history** — what reproduces, what doesn't, and why; the encoding-step notes are its appendix |
| `output/picture/`, `output/table/` | figures (Figure 5, Figure 8, training curve) and the Figure 5 grid results |
| `logs/` | logs of the recorded runs (k=20, bias-ablation, final k=5) |
| `checkpoints/`, `features/`, `data/` | regenerable artifacts, gitignored (≈27 GB total) |

## Environment

M4 Mac, 16 GB RAM, MPS (CUDA works too; CPU-only is feasible but slow for encoding and
training). ~30 GB free disk. Python ≥ 3.10 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The 12 GB patch memmaps in `features/` must be **streamed, never loaded into RAM** —
`train_sae.py` already does this.

## Run order

All commands from inside this folder. Each step gates the next.

```bash
# 0. data (~1.4 GB download; Drive quota errors are common — just retry later)
python data/fetch_celeba.py
python data/build_splits.py
unzip -q data/img_align_celeba.zip -d data/            # ~21 s, 202,599 images

# 1. SigLIP features for val+test (~15–30 min on MPS; writes 12 GB of memmaps)
python encode_siglip.py --limit 100                    # optional smoke test first
python encode_siglip.py

# 1b. optional diagnostic: probe pooled features directly (paper's Appendix E.1 check)
python sufficiency_check.py

# 2. train the SAE and label the codes (~1 h on M4: 20 epochs × ~180 s)
python train_sae.py --tag k5

# 3. the Figure 5 grid: 4 n's × 9 ATEs × 10 seeds (~15 min, CPU only)
python figure5.py

# 4. plots
python plot_figure5.py                    # output/picture/figure5.png + figure8.png
```

`train_sae.py` defaults to Table 1 as printed (k=5). The paper contradicts itself on k
(Table 1 says 5; Appendix E.1/E.2.2 say 20); `--k 20 --tag k20` runs the other reading —
both clear the alignment checkpoint (see `reproduce_note.md` §6). `figure5.py --codes
codes_test.npy` points the grid at the k=20 codes.

## Expected results

After `train_sae.py --tag k5` (exact numbers vary slightly with hardware/seed):

```
attribute      argmax raw   F1raw argmax bal   F1bal   paper (experimental prevalence)
Eyeglasses           2455  0.7406       8654  0.9378   neuron 6051, F1=0.748
Wearing_Hat          7179  0.6707       6795  0.9360   neuron 38, F1=0.841
```

`F1bal` (balanced prevalence) is the number comparable to the paper; `F1raw` is trend-only.
This distinction matters — the paper scores F1 on the *simulated experiment sample*
(prevalence ~0.35–0.5), not the raw test split (4–6%). Scoring at raw prevalence made the
replication look broken for a full day (`reproduce_note.md` §5).

`figure5.py`, averaged over ATE and seeds (precision / recall):

| n | NES | Bonferroni | FDR | t-test |
|---|---|---|---|---|
| 30 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.03 / 0.5 |
| 250 | **0.58 / 0.5** | 0.13 / 0.9 | 0.04 / 1.0 | 0.01 / 1.0 |
| 500 | **0.41 / 0.6** | 0.04 / 1.0 | 0.01 / 1.0 | 0.00 / 1.0 |
| 5000 | **0.13 / 0.7** | 0.00 / 1.0 | 0.00 / 1.0 | 0.00 / 1.0 |

Known quantitative gaps vs. the paper (documented, with suspects, in
`reproduce_note.md` §7): NES recall plateaus at 0.5–0.75 instead of reaching 1.0, and at
n=5000 NES over-selects. Both trace to NES's selection/stopping details, not the SAE.

## Implementation notes

- **NES follows the authors' shipped reference code** (ICLR supplementary) where it
  diverges from the appendix pseudocode: strata binarized at `> 0`, cutpoints pooled over
  T, joint `2^|S|` stratification across rounds, strata missing an arm dropped with
  weights renormalized. Selection is Bonferroni filter first, then largest |τ̂|. The
  implementation was validated against analytic ground truth in the Figure 3 notebook
  (`../reproduce_fig3.ipynb`, outside this folder).
- **Do not use HuggingFace CelebA mirrors for splits** — `tpremoli/CelebA-attrs` swaps
  val and test. `build_splits.py` trusts only the raw `list_eval_partition.txt` and
  asserts the official sizes.
- **The pre-encoder bias is mandatory.** Without it, mean-carrier atoms hog the top-k
  slots (51 codes firing on 100% of images). `--no-bias` reproduces the failure.
- **Train per patch, not on pooled vectors** — pooled training starves the dictionary
  (91% dead codes).
- Full list of deviations and unspecified-detail choices: `reproduce_note.md` §5 (bottom)
  and the `train_sae.py` docstring.

The hearings pilot (`../partisan_congress_speech/`) carries its own vendored copy of the
SAE and NES code in `saelib/`; this folder's copies are the tested originals — fix here
first, then re-copy.
