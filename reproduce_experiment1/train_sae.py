"""Step 3 — train the top-k SAE on SigLIP patch features, then label the codes by F1.

Defaults are Table 1 as printed: top-k with **k=5**, input 768, m=9216, Adam, lr 5e-4,
batch 20, 20 epochs, grad clip 1.0. (Appendix E.1/E.2.2 say k=20 for "the main
experiments"; the tables disagree with the text, so k stays a flag — `--k 20` to test the
other reading.) Trains on val, evaluates on test, per Appendix D.1: "we employ the
validation data for training SAEs and the test data to interpret them."

The SAE runs **per patch**: Table 1's batch=20 is read as 20 images = 20 x 196 = 3,920
patch vectors per step (993 steps/epoch). Codes are mean-pooled over the 196 patches into
one Z in R^9216 per image — D.1's "aggregate patchwise by mean pooling". Training on
FM-pooled vectors instead starves the dictionary: 43 activations per code, 91% dead.

Known deviations from the paper's Eq. (3), kept deliberately and documented here:

- Eq. (3) has untied encoder/decoder maps E, D with biases b_e, b_d. This implementation
  is **tied** (one dictionary, row-normalised, used for both directions) with a single
  pre-encoder bias (x - b, encode, decode, + b), following the vendored TopKEncoder plus
  the standard mean-centring fix. Without that bias, mean-carrier atoms hog the top-k
  slots: 51 codes fired on 100% of images and carried 66% of activation mass
  (`--no-bias` reproduces this).
- The vendored ReLU after top-k was removed: it is not in the paper, and it was verified
  to be a no-op here (all top-k scores positive on this data).

Evaluation prevalence, important: the paper's Evaluation paragraph scores F1 over
{i=1..n} — the *simulated experiment's sample*, prevalence ~0.35-0.5 — not the raw test
split (4-6%). Per-epoch logs therefore report both `F1raw` (raw test prevalence, internal
trend only) and `F1bal` (balanced-prevalence subsample, the number comparable to the
paper's 0.748 / 0.841). The definitive G for Figure 5 must be identified on the simulated
sample in figure5.py, not taken from the raw-prevalence argmax saved here.

Writes `checkpoints/train_history.csv`, `logs/train_log.txt` and
`output/picture/training_curve.png`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "checkpoints"
ATTRS = ["Eyeglasses", "Wearing_Hat", "Smiling"]
PAPER = {"Wearing_Hat": (38, 0.841), "Eyeglasses": (6051, 0.748), "Smiling": (None, None)}


class Log:
    """Print to the terminal and append to train_log.txt at the same time."""

    def __init__(self, path):
        self.fh = open(path, "a", buffering=1)
        self.fh.write(f"\n{'='*70}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.fh.write(msg + "\n")


def patches(split):
    return np.load(ROOT / "features" / f"patches_{split}.f16", mmap_mode="r")


def load_labels(split):
    idx = pd.read_parquet(ROOT / "features" / f"index_{split}.parquet")
    meta = pd.read_parquet(ROOT / "data" / "meta" / "celeba_meta.parquet")
    sub = meta[meta.split == split].reset_index(drop=True)
    assert (idx.file.values == sub.file.values).all()
    return sub


def encode(x, dictionary, k, b_dec=None):
    normed = dictionary / dictionary.norm(dim=-1, keepdim=True)
    xc = x if b_dec is None else x - b_dec
    scores = xc @ normed.T
    top = scores.topk(k, dim=-1)
    # top-k is the nonlinearity (Table 1); no ReLU here — not in the paper, and verified
    # to be a no-op on this data anyway
    code = torch.zeros_like(scores).scatter_(-1, top.indices, top.values)
    return code, normed


def decode(code, normed, b_dec=None):
    out = code @ normed
    return out if b_dec is None else out + b_dec


def codes_for(P, rows, dictionary, k, b_dec, m, device, chunk=32):
    """Encode images per patch and mean-pool the codes to one Z per image."""
    d = P.shape[2]
    Z = np.empty((len(rows), m), dtype=np.float32)
    alive = torch.zeros(m, dtype=torch.bool, device=device)
    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            sel = rows[s : s + chunk]
            x = torch.from_numpy(np.asarray(P[sel]).reshape(-1, d).astype(np.float32)).to(device)
            c, _ = encode(x, dictionary, k, b_dec)
            alive |= (c > 0).any(0)
            Z[s : s + chunk] = c.view(len(sel), -1, m).mean(1).cpu().numpy()
    return Z, int((~alive).sum())


def f1_spectrum(Z, labels, attrs=ATTRS):
    active = Z > 0
    pred_pos = active.sum(0).astype(np.float64)
    out = {}
    for a in attrs:
        y = labels[a].values.astype(np.float64)
        tp = active.T.astype(np.float64) @ y
        out[a] = 2 * tp / np.maximum(pred_pos + y.sum(), 1e-9)
    return out


def balanced_rows(labels, attr, rng):
    """Row indices with prevalence forced to 0.5: every positive plus an equal number of
    sampled negatives. Approximates the paper's evaluation prevalence (the Evaluation
    paragraph scores F1 on the simulated sample, where P(Y) is 0.35-0.5, not raw CelebA's
    4-6%)."""
    y = labels[attr].values.astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    if len(pos) == 0 or len(neg) < len(pos):
        return None
    return np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])


def curve(history, path):
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(h.epoch, h.mse, color="#2b3a67", lw=1.8)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("reconstruction MSE")
    axes[0].grid(alpha=0.25, lw=0.6)

    if "f1bal_Eyeglasses" in h:
        ev = h.dropna(subset=["f1bal_Eyeglasses"])
        for a, c, ref in [("Eyeglasses", "#2b3a67", 0.748), ("Wearing_Hat", "#d6604d", 0.841)]:
            axes[1].plot(ev.epoch, ev[f"f1bal_{a}"], color=c, lw=1.8, label=f"{a} (balanced)")
            if f"f1raw_{a}" in ev:
                axes[1].plot(ev.epoch, ev[f"f1raw_{a}"], color=c, lw=1.0, alpha=0.35)
            axes[1].axhline(ref, color=c, ls=":", lw=1.2, alpha=0.7)
        axes[1].set_ylim(0, 1)
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("best single-code F1")
    axes[1].grid(alpha=0.25, lw=0.6)
    axes[1].set_title("dotted = paper's F1 (at experimental prevalence); faint = raw prevalence", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5,
                    help="Table 1 says 5; Appendix E.1/E.2.2 say 20 — use --k 20 for that reading")
    ap.add_argument("--m", type=int, default=9216)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--images-per-batch", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bias", dest="bias", action="store_false",
                    help="reproduce the vendored no-bias formulation")
    ap.add_argument("--eval-every", type=int, default=1, help="0 disables per-epoch F1")
    ap.add_argument("--eval-images", type=int, default=5000,
                    help="per-epoch F1 is on this subsample; the final number is on all 19,962")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    suffix = f"_{args.tag}" if args.tag else ""
    (ROOT / "logs").mkdir(exist_ok=True)
    log = Log(ROOT / "logs" / f"train_log{suffix}.txt")

    P = patches("val")
    n_img, n_patch, d = P.shape
    log(f"train on val: {n_img:,} images x {n_patch} patches = {n_img*n_patch:,} vectors x {d}")
    log(f"m={args.m}  k={args.k}  bias={args.bias}  epochs={args.epochs}  device={device}")

    dictionary = torch.nn.Parameter(torch.randn(args.m, d, device=device))
    params, b_dec = [dictionary], None
    if args.bias:
        sample = np.asarray(P[:: max(n_img // 2000, 1)]).reshape(-1, d).astype(np.float32)
        b_dec = torch.nn.Parameter(torch.from_numpy(sample.mean(0)).to(device))
        params.append(b_dec)
        log(f"pre-encoder bias initialised to the data mean (norm {b_dec.norm():.2f})")

    opt = torch.optim.Adam(params, lr=args.lr)
    Pte, te = patches("test"), load_labels("test")
    # separate generator: drawing eval rows from `rng` would advance it and change the
    # training permutation, so --eval-images would silently alter the trained model
    eval_rng = np.random.default_rng(12345)
    eval_rows = np.sort(eval_rng.choice(Pte.shape[0],
                                        min(args.eval_images, Pte.shape[0]), replace=False))
    te_eval = te.iloc[eval_rows].reset_index(drop=True)
    # balanced subsets fixed once, so the per-epoch curve compares like with like
    bal_sel = {a: balanced_rows(te_eval, a, eval_rng) for a in ("Eyeglasses", "Wearing_Hat")}

    history, steps = [], n_img // args.images_per_batch
    for ep in range(args.epochs):
        order = rng.permutation(n_img)
        total, t0 = 0.0, time.time()
        for i in range(steps):
            sel = np.sort(order[i * args.images_per_batch : (i + 1) * args.images_per_batch])
            x = torch.from_numpy(np.asarray(P[sel]).reshape(-1, d).astype(np.float32)).to(device)
            code, normed = encode(x, dictionary, args.k, b_dec)
            loss = F.mse_loss(x, decode(code, normed, b_dec))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            total += loss.item()

        row = {"epoch": ep + 1, "mse": total / steps, "seconds": time.time() - t0}
        msg = f"  epoch {ep+1:2d}/{args.epochs}  mse {row['mse']:.4f}  {row['seconds']:.0f}s"
        if args.eval_every and (ep + 1) % args.eval_every == 0:
            Ze, dead = codes_for(Pte, eval_rows, dictionary, args.k, b_dec, args.m, device)
            spec = f1_spectrum(Ze, te_eval)
            for a in ("Eyeglasses", "Wearing_Hat"):
                row[f"f1raw_{a}"] = float(np.nanmax(spec[a]))
                if bal_sel[a] is not None:
                    bspec = f1_spectrum(Ze[bal_sel[a]], te_eval.iloc[bal_sel[a]], attrs=[a])
                    row[f"f1bal_{a}"] = float(np.nanmax(bspec[a]))
            row["dead"] = dead
            msg += (f"  F1bal eye {row.get('f1bal_Eyeglasses', float('nan')):.3f}"
                    f" hat {row.get('f1bal_Wearing_Hat', float('nan')):.3f}"
                    f"  (raw {row['f1raw_Eyeglasses']:.3f}/{row['f1raw_Wearing_Hat']:.3f})"
                    f"  dead {dead}")
        history.append(row)
        log(msg)

    OUT.mkdir(exist_ok=True)
    torch.save({"dict": dictionary.detach().cpu(),
                "b_dec": None if b_dec is None else b_dec.detach().cpu(),
                "k": args.k}, OUT / f"sae_topk{suffix}.pt")
    pd.DataFrame(history).to_csv(OUT / f"train_history{suffix}.csv", index=False)
    (ROOT / "output" / "picture").mkdir(parents=True, exist_ok=True)
    curve(history, ROOT / "output" / "picture" / f"training_curve{suffix}.png")
    log(f"wrote output/picture/training_curve{suffix}.png and checkpoints/train_history{suffix}.csv")

    # full test encode, the codes Figure 5 runs on
    Z, dead = codes_for(Pte, np.arange(Pte.shape[0]), dictionary, args.k, b_dec, args.m, device)
    np.save(ROOT / f"features/codes_test{suffix}.npy", Z.astype(np.float16))
    log(f"\ntest codes {Z.shape} | dead codes: {dead:,} / {args.m} ({dead/args.m:.1%})")

    spec = f1_spectrum(Z, te)
    bal_full_rng = np.random.default_rng(6789)
    save = dict(spec)
    log(f"\n{'attribute':14s} {'argmax raw':>10s} {'F1raw':>7s} {'argmax bal':>10s} {'F1bal':>7s}   paper (experimental prevalence)")
    for a in ATTRS:
        j = int(np.nanargmax(spec[a]))
        line = f"{a:14s} {j:>10d} {spec[a][j]:7.4f}"
        sel = balanced_rows(te, a, bal_full_rng)
        if sel is not None:
            bspec = f1_spectrum(Z[sel], te.iloc[sel].reset_index(drop=True), attrs=[a])[a]
            save[f"{a}_balanced"] = bspec
            jb = int(np.nanargmax(bspec))
            line += f" {jb:>10d} {bspec[jb]:7.4f}"
        ref = PAPER[a]
        log(line + ("   " + f"neuron {ref[0]}, F1={ref[1]}" if ref[1] else "   (distractor)"))
    np.savez(ROOT / f"features/f1_spectrum{suffix}.npz", **save)

    worst = min(float(np.nanmax(save[f"{a}_balanced"])) for a in ("Eyeglasses", "Wearing_Hat")
                if f"{a}_balanced" in save)
    log("\nNOTE: the raw-prevalence argmax is NOT the paper's G. The Evaluation paragraph")
    log("scores F1 over {i=1..n}, the simulated experiment's sample (prevalence ~0.35-0.5).")
    log("figure5.py must identify G on the simulated sample, not from this file's argmax.")
    log(f"balanced-prevalence checkpoint: worst = {worst:.3f} "
        + ("(clears 0.6)" if worst >= 0.6 else "(below 0.6 — investigate before Figure 5)"))


if __name__ == "__main__":
    main()
