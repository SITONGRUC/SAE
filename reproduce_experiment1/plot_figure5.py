"""Plot Figure 5 (and Figure 8) from the grid that figure5.py wrote.

Figure 5 is a 3 x 4 panel: one row per metric (Precision, Recall, IoU), one column per sample
size, with ATE on the x-axis and one curve per method. Bands are +-1 s.d. over the 10 seeds.

Figure 8 comes free from the F1 spectrum saved during training: sort every code's F1 against
an attribute and plot it. Assumption A.2 wants a single code far above a flat tail — one
principal neuron, everything else only marginally aligned. Read the shape rather than assume
it: a gradual decline across the leading codes is entanglement, which is the condition the
paper's whole argument is about, not a plotting artefact.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = ["precision", "recall", "iou"]
LABELS = {"precision": "Precision", "recall": "Recall", "iou": "IoU"}
# NES emphasised; the baselines share the warm end of the ramp
COLORS = {
    "NES": "#2b3a67",
    "Bonferroni": "#4f9d9d",
    "FDR": "#f4a582",
    "t-test": "#d6604d",
    "top-k": "#9970ab",
}
ORDER = ["NES", "Bonferroni", "FDR", "t-test", "top-k"]


def figure5(df, out, ate_fixed=0.5, n_fixed=500, taus=(0.1, 0.2, 0.5, 0.8)):
    """Grouped bar chart in the paper's layout: top row sweeps N at a fixed ATE, bottom row
    sweeps ATE at a fixed n. Bars are means over the 10 seeds, whiskers +-1 s.d.

    The paper does not print which slice each row fixes; ATE=0.5 / n=500 are the midpoints
    of its grids and match the bar heights best.
    """
    ns = sorted(df.n.unique())
    rows = [
        (df[df.ate == ate_fixed], "n", ns, f"N   (ATE = {ate_fixed})"),
        (df[(df.n == n_fixed) & (df.ate.isin(taus))], "ate", list(taus), f"ATE   (n = {n_fixed})"),
    ]

    fig, axes = plt.subplots(2, len(METRICS), figsize=(3.6 * len(METRICS), 5.6), sharey=True)
    width = 0.15

    for r, (sub, xcol, xvals, xlabel) in enumerate(rows):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            x = np.arange(len(xvals))
            for j, meth in enumerate(ORDER):
                g = sub[sub.method == meth].groupby(xcol)[metric]
                mu = g.mean().reindex(xvals).values
                sd = g.std().reindex(xvals).fillna(0).values
                ax.bar(x + (j - 2) * width, mu, width * 0.92, yerr=sd,
                       color=COLORS[meth], label=meth if (r, c) == (0, 0) else None,
                       error_kw=dict(lw=0.8, capsize=2, capthick=0.8, ecolor="#555555"))
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in xvals])
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(LABELS[metric], fontsize=11)
            if c == 0:
                ax.set_ylabel("score")
            ax.set_xlabel(xlabel)

    axes[0, 0].legend(frameon=False, fontsize=8, ncol=1, loc="upper right")
    fig.suptitle("Figure 5 — NES vs. baselines on the CelebA semi-synthetic RCT", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def figure8(out, spectrum="f1_spectrum_k5.npz"):
    path = ROOT / "features" / spectrum
    if not path.exists():
        print("no f1_spectrum.npz — skipping Figure 8")
        return
    spec = np.load(path)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for attr, color in [("Eyeglasses", "#2b3a67"), ("Wearing_Hat", "#d6604d"),
                        ("Smiling", "#bbbbbb")]:
        if attr not in spec:
            continue
        f1 = np.sort(np.nan_to_num(spec[attr]))[::-1]
        ax.plot(f1, color=color, lw=1.6,
                label=f"{attr}  (best {f1[0]:.3f}, code {int(np.nanargmax(spec[attr]))})")
    ax.set_xscale("log")
    ax.set_xlabel("codes, sorted by alignment")
    ax.set_ylabel("F1 vs. attribute")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure 8 — code alignment spectrum per concept")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="figure5.parquet")
    args = ap.parse_args()

    (ROOT / "output" / "picture").mkdir(parents=True, exist_ok=True)
    path = ROOT / "output" / "table" / args.grid
    if path.exists():
        df = pd.read_parquet(path)
        figure5(df, ROOT / "output" / "picture" / "figure5.png")
        summary = df.groupby(["method", "n"])[METRICS].mean().round(3)
        print(f"\naveraged over ATE and seeds:\n{summary}")
    else:
        print(f"no {args.grid} — run figure5.py first")
    figure8(ROOT / "output" / "picture" / "figure8.png")


if __name__ == "__main__":
    main()
