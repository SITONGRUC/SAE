"""MVP — is there partisan signal in hearing speech that an SAE can carve into channels?

Pipeline: train a small SAE (m=1000, k=5, pre-encoder bias) on ALL 67k segment embeddings
(unsupervised — party labels never touch training), then test politicians' D vs R codes at
two levels:

  1. naive segment-level t-test on 29.5k segments  -> the paradox demo (dependence ignored,
     everything lights up)
  2. speaker-level test: one row per politician (mean of their segment codes), Welch t
     across 238 D vs 233 R independent people -> the honest existence test

Success gate: some codes clear Bonferroni at the SPEAKER level. Interpretation: print the
top-activating segments of the top codes and eyeball against Cassidy-Kempf's partisan
vocabulary. This is a 2-hour existence proof, not the paper protocol (no hearing strata,
no cluster SEs, no NES) — those belong to the real design.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_ind

from saelib.sae import encode, decode  # tied top-k SAE with pre-encoder bias

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "output"
M, K, EPOCHS, LR, BS = 1000, 5, 15, 5e-4, 64
ALPHA = 0.05


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    X = np.load(DATA / "emb.npy")  # (67080, 1536) float32
    meta = pd.read_parquet(DATA / "meta.parquet")
    n, d = X.shape
    print(f"{n:,} segments x {d} | SAE m={M} k={K} epochs={EPOCHS} device={device}")

    # ---- train on everything (unsupervised) ----
    Xt = torch.from_numpy(X).to(device)
    dictionary = torch.nn.Parameter(torch.randn(M, d, device=device))
    b_dec = torch.nn.Parameter(Xt.mean(0).clone())
    opt = torch.optim.Adam([dictionary, b_dec], lr=LR)
    steps = n // BS
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        tot, t0 = 0.0, time.time()
        for i in range(steps):
            x = Xt[perm[i * BS:(i + 1) * BS]]
            code, normed = encode(x, dictionary, K, b_dec)
            loss = F.mse_loss(x, decode(code, normed, b_dec))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1:2d}/{EPOCHS}  mse {tot/steps:.5f}  {time.time()-t0:.0f}s", flush=True)

    # ---- encode all segments ----
    with torch.no_grad():
        Z = torch.cat([encode(Xt[i:i + 4096], dictionary, K, b_dec)[0].cpu()
                       for i in range(0, n, 4096)]).numpy()
    dead = int((Z > 0).sum(0).min() == 0) and int(((Z > 0).sum(0) == 0).sum())
    print(f"codes: dead {((Z>0).sum(0)==0).sum()}/{M} | active per segment {(Z>0).sum(1).mean():.1f}")

    # ---- politicians with D/R labels ----
    pol = meta[(meta.ID == "politician") & meta.party.isin(["Democratic", "Republican"])]
    ix = pol.index.to_numpy()
    Zp, T = Z[ix], (pol.party == "Republican").to_numpy()  # T=1 Republican
    print(f"\npoliticians: {len(pol):,} segments, {pol.full_name.nunique()} people "
          f"({(~T).sum():,} D / {T.sum():,} R segments)")

    # ---- level 1: naive segment-level scan (the paradox demo) ----
    tstat, pval = ttest_ind(Zp[T], Zp[~T], axis=0, equal_var=False)
    naive_bonf = int((pval < ALPHA / M).sum())
    print(f"[naive segment-level]  t-test p<.05: {(pval < ALPHA).sum()} / {M}   "
          f"Bonferroni: {naive_bonf} / {M}")

    # ---- level 2: one row per politician ----
    g = pd.DataFrame(Zp).assign(name=pol.full_name.values, party=pol.party.values)
    spk = g.groupby("name").agg({**{c: "mean" for c in range(M)}, "party": "first"})
    Zs = spk[list(range(M))].to_numpy()
    Ts = (spk.party == "Republican").to_numpy()
    tstat_s, pval_s = ttest_ind(Zs[Ts], Zs[~Ts], axis=0, equal_var=False)
    sig = np.where(pval_s < ALPHA / M)[0]
    print(f"[speaker-level, n={len(spk)}]  t-test p<.05: {(pval_s < ALPHA).sum()} / {M}   "
          f"Bonferroni: {len(sig)} / {M}")

    # ---- interpret the top codes ----
    order = sig[np.argsort(pval_s[sig])] if len(sig) else np.argsort(pval_s)[:5]
    texts = meta.text.fillna("")
    print("\n==== top party-separating codes (speaker level) ====")
    for j in order[:5]:
        lean = "R" if tstat_s[j] > 0 else "D"
        print(f"\n--- code {j}  t={tstat_s[j]:+.1f}  p={pval_s[j]:.1e}  leans {lean} ---")
        top = np.argsort(-Z[:, j])[:6]
        for i in top:
            row = meta.iloc[i]
            snip = " ".join(str(texts.iloc[i]).split())[:220]
            print(f"  [{row.year} {row.party if isinstance(row.party,str) else '?':12s} "
                  f"{str(row.full_name)[:22]:22s}] {snip}")

    (OUT / "table").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "codes_all.npy", Z.astype(np.float16))
    spk.to_parquet(OUT / "speaker_codes.parquet")
    tests = pd.DataFrame({"code": range(M), "t_speaker": tstat_s, "p_speaker": pval_s,
                          "t_naive": tstat, "p_naive": pval})
    tests.to_parquet(OUT / "table" / "code_tests.parquet")
    tests.to_csv(OUT / "table" / "code_tests.csv", index=False)
    print("\nsaved output/codes_all.npy, output/speaker_codes.parquet, "
          "output/table/code_tests.{parquet,csv}")


if __name__ == "__main__":
    main()
