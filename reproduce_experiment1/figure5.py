"""Step 4 — Figure 5: Precision / Recall / IoU for NES against the baselines.

DGP from Appendix D.1. Sample (T, W, Y1, Y2), then retrieve a real CelebA test image whose
attributes match the realized (Y1, Y2, W):

    T, W ~ Bernoulli(0.5)          Y1 = Eyeglasses, Y2 = Wearing Hat, W = Smiling
    Y2 | T=1 : 0.5 + ATE/2         Y2 | T=0 : 0.5 - ATE/2
    Y1 | T=1, W=1 : 0.5 + ATE/2    Y1 | T=0, W=1 : 0.5 - ATE/2
    Y1 | T=1, W=0 : 0.2 + ATE      Y1 | T=0, W=0 : 0.2

W modifies Y1 only and is a deliberate distractor: it is not in the ground truth G, so
selecting its code counts as a false positive.

Retrieval is with replacement. It has to be — the (Eyeglasses=1, Hat=1) cells hold only 41
and 82 images in the whole test split, and at n=5000 they are drawn hundreds of times.

**G is identified on each simulated sample**, per the paper's Evaluation paragraph: the
predictions ŷ_ik = I{Z_j(X_i) > 0} are scored over {i=1..n} — the n simulated units, where
prevalence is ~0.35-0.5 — and g_k is the argmax-F1 code on that sample. G is NOT the argmax
on the raw test split (prevalence 4-6%): that reading cost this replication a full day of
false alarms (see reproduce_note.md §5).

Codes default to the Table-1-faithful k=5 run (`codes_test_k5.npy`).
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from nes import baselines, nes, score

ROOT = Path(__file__).resolve().parent
METHODS = ["NES", "t-test", "Bonferroni", "FDR", "top-k"]


def load(codes="codes_test_k5.npy"):
    Z = np.load(ROOT / "features" / codes).astype(np.float32)
    idx = pd.read_parquet(ROOT / "features" / "index_test.parquet")
    meta = pd.read_parquet(ROOT / "data" / "meta" / "celeba_meta.parquet")
    te = meta[meta.split == "test"].reset_index(drop=True)
    assert (idx.file.values == te.file.values).all()

    # pools of row indices for each realized (Y1, Y2, W) cell
    pools = {}
    for y1, y2, w in itertools.product((0, 1), repeat=3):
        sel = np.where(
            (te.Eyeglasses.values == y1)
            & (te.Wearing_Hat.values == y2)
            & (te.Smiling.values == w)
        )[0]
        pools[(y1, y2, w)] = sel
    return Z, pools


def simulate(n, ate, rng, pools):
    """Draw (T, W, Y1, Y2) from the DGP, then retrieve matching images (with replacement)."""
    T = rng.integers(0, 2, n)
    W = rng.integers(0, 2, n)
    p2 = np.where(T == 1, 0.5 + ate / 2, 0.5 - ate / 2)
    p1 = np.where(
        W == 1,
        np.where(T == 1, 0.5 + ate / 2, 0.5 - ate / 2),
        np.where(T == 1, 0.2 + ate, 0.2),
    )
    Y2 = (rng.random(n) < p2).astype(int)
    Y1 = (rng.random(n) < np.clip(p1, 0, 1)).astype(int)

    rows = np.empty(n, dtype=np.int64)
    for key, pool in pools.items():
        mask = (Y1 == key[0]) & (Y2 == key[1]) & (W == key[2])
        k = int(mask.sum())
        if k and len(pool):
            rows[mask] = rng.choice(pool, k, replace=True)
        elif k:
            raise SystemExit(f"no test images for cell {key}")
    return T, rows, Y1, Y2


def identify_g(Zs, ys):
    """g_k = argmax_j F1( I{Z_j > 0}, y_k ) over the simulated sample — Evaluation, {i=1..n}.

    Duplicated draws count as many times as the DGP produced them, which is what the paper's
    notation implies. At very small n a concept can be absent from the sample; the argmax is
    then arbitrary, which is the protocol's own small-sample behaviour, not a bug here.
    """
    active = Zs > 0
    pred_pos = active.sum(0).astype(np.float64)
    G = []
    for y in ys:
        y = y.astype(np.float64)
        tp = active.T.astype(np.float64) @ y
        f1 = 2 * tp / np.maximum(pred_pos + y.sum(), 1e-9)
        G.append(int(np.argmax(f1)))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="codes_test_k5.npy")
    ap.add_argument("--n", type=int, nargs="+", default=[30, 250, 500, 5000])
    ap.add_argument("--ate", type=float, nargs="+",
                    default=[round(0.1 * i, 1) for i in range(9)])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="figure5")
    args = ap.parse_args()

    Z, pools = load(args.codes)
    print(f"codes {Z.shape} from {args.codes} | G identified per simulated sample | "
          f"{len(args.n)} x {len(args.ate)} x {args.seeds} = "
          f"{len(args.n)*len(args.ate)*args.seeds} cells")

    rec = []
    for n in args.n:
        for ate in args.ate:
            for seed in range(args.seeds):
                rng = np.random.default_rng(hash((n, ate, seed)) % (2 ** 32))
                T, rows, Y1, Y2 = simulate(n, ate, rng, pools)
                Zs = Z[rows]
                G = identify_g(Zs, [Y1, Y2])
                sets = baselines(T, Zs, top_k=len(set(G)))
                sets["NES"] = nes(T, Zs)
                for meth in METHODS:
                    p, r, iou = score(sets[meth], G)
                    rec.append(dict(n=n, ate=ate, seed=seed, method=meth,
                                    precision=p, recall=r, iou=iou,
                                    n_selected=len(sets[meth]),
                                    g1=G[0], g2=G[1]))
            print(f"  n={n:5d} ate={ate:.1f} done", flush=True)

    df = pd.DataFrame(rec)
    (ROOT / "output" / "table").mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "output" / "table" / f"{args.out}.parquet", index=False)

    summary = df.groupby(["n", "method"])[["precision", "recall", "iou"]].mean().round(3)
    print(f"\naveraged over ATE and seeds:\n{summary}")

    stab = df.groupby("n")[["g1", "g2"]].nunique()
    print(f"\ndistinct G codes per n (G stability across ATE x seeds):\n{stab}")
    print(f"\nwrote {args.out}.parquet")


if __name__ == "__main__":
    main()
