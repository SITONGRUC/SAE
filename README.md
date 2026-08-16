# SAE — replicating *Exploratory Causal Inference in SAEnce*, then taking it to finance

This repository does two things. First, it **replicates** Mencattini, Cadei & Locatello's
*Exploratory Causal Inference in SAEnce* (ICLR 2026) — a framework for finding treatment
effects in unstructured outcomes by encoding them with a foundation model, disentangling
the embeddings with a top-k sparse autoencoder (SAE), and testing the resulting codes
with leakage-aware inference (Neural Effect Search). Second, it **pilots the transfer**
of that machinery to an econ/finance question: partisan divergence in Fed and
banking-committee hearing speech, 1951–2023, measured against the bigram paradigm of
Gentzkow–Shapiro–Taddy (2019). Each half is a self-contained sub-repo with its own
README, requirements, and outputs.

## Results at a glance

| claim | verdict |
|---|---|
| Figure 3 (the paradox: rejection rate → 1 as n or τ grows) | **fully reproduced** (`reproduce_fig3.ipynb`) |
| SAE code–concept alignment | **exceeds the paper**: best F1 0.938 / 0.936 vs 0.748 / 0.841 |
| Figure 5 (NES vs baselines) | **qualitative claim reproduced** — baselines' recall → 1 while precision → 0, NES holds; two quantitative gaps documented with suspects |
| Finance pilot | unsupervised SAE recovers the dual-mandate partisan split (D: jobs, R: deficits) with GST's post-1995 timing; discovery↔lexicon loop closes at rank 1/1,000 |

## Repository structure

```
.
├── README.md                    ← you are here
├── reading_note.md              the paper in my own words: the SAE, the leakage
│                                paradox, the NES algorithm, Experiment 1
├── reproduce_fig3.ipynb         Figure 3 replication — pure numpy, no data needed;
│                                doubles as the NES unit test (ground truth is analytic)
├── original paper/              the paper PDF + the authors' NES reference
│                                implementation from the ICLR supplementary ("nes 2.py")
│
├── reproduce_experiment1/       ★ SUB-REPO 1 — the Experiment 1 replication
│   ├── README.md                what it reproduces, full run order, expected numbers
│   ├── reproduce_note.md        the debugging history: every hypothesis tested and its
│   │                            verdict (read before touching the SAE); encoding notes
│   │                            merged as its appendix
│   ├── data/                    CelebA fetch + split-building scripts (md5-checked,
│   │                            HF-mirror pitfalls documented); images gitignored
│   ├── encode_siglip.py         CelebA → SigLIP patch features (12 GB memmaps)
│   ├── sufficiency_check.py     linear-probe diagnostic (paper's Appendix E.1)
│   ├── train_sae.py             tied top-k SAE, 768 → 9,216, per-patch + bias
│   ├── nes.py                   NES + baselines, pure numpy (the tested original)
│   ├── figure5.py               DGP + 360-cell grid → output/table/figure5.parquet
│   ├── plot_figure5.py          → output/picture/{figure5,figure8}.png
│   ├── output/picture|table/    figures and grid results
│   ├── logs/                    logs of the recorded runs (k=20, bias ablation, k=5)
│   └── checkpoints/ features/   trained SAEs + features (gitignored, regenerable)
│
└── partisan_congress_speech/    ★ SUB-REPO 2 — the finance pilot
    ├── README.md                written as a short paper: personal motivation, three
    │                            construction-level advantages over bigram methods,
    │                            figure, findings, caveats, full citations
    ├── MVP_RESULT.md            the pilot's lab notes (top-activating segments per code)
    ├── saelib/                  SAE + NES vendored from sub-repo 1 (provenance noted)
    ├── data/                    67k hearing-segment embeddings + speaker↔party
    │                            crosswalks; the 2.2 GB source CSV gitignored
    ├── mvp.py                   train SAE unsupervised → segment- vs speaker-level tests
    ├── plot_mvp.py              → output/picture/mvp_figure.png
    └── output/picture|table/    figure; per-code test tables + novelty screen
```

## Where to start

- **Understand the paper** → `reading_note.md`, then `reproduce_fig3.ipynb` for the
  paradox made concrete.
- **Rerun the replication** → `reproduce_experiment1/README.md` (data download → SigLIP →
  SAE → Figure 5; ~2–3 h wall clock, mostly waiting, on an M4/16 GB with MPS).
- **The research pitch** → `partisan_congress_speech/README.md` — written to be read on
  its own, figure and references included.

## Status and open items

Replication (stage 1) is complete. Open, in order:

1. Two ~20-minute experiments to close the Figure 5 quantitative gaps — SD-normalize τ̂
   in NES selection; replicate the reference code's NaN-kills-neuron stratum behaviour
   (suspects documented in `reproduce_experiment1/reproduce_note.md` §7).
2. The full hearings design (sentence-level re-embedding, hearing strata,
   speaker-clustered inference, role/majority controls) — sketched in the pilot README's
   "Next" section.
3. Codify the novelty screen (its artifacts exist in
   `partisan_congress_speech/output/table/`; the script does not yet).

## Environment

Apple M4, 16 GB, MPS; Anaconda Python (not the system interpreter). Each sub-repo has its
own `requirements.txt`. Nothing needs a discrete GPU; the heavy features are float16
memmaps that must be streamed, never loaded into RAM.
