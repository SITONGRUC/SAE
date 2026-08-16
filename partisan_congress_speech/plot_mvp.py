"""One figure for the MVP: dependence inflation, era timing, and the top partisan codes.

Panel A  the paradox demo — Bonferroni discoveries at segment vs speaker level
Panel B  speaker-level t distribution across all 1,000 codes, pre- vs post-1995
Panel C  the four strongest post-1995 codes: one dot per politician, D vs R

Party colors are semantic (D blue, R red); pair validated for CVD separation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "output"
BLUE, RED = "#2b5fa8", "#c0392b"
INK, MUTED = "#333333", "#767676"
M = 1000

CODE_LABELS = {216: "labor market / jobs", 235: "wages & full employment",
               837: "program funding & cuts", 510: "debt & deficits"}


def speaker_table(meta, Z, lo, hi):
    pol = meta[(meta.ID == "politician") & meta.party.isin(["Democratic", "Republican"])
               & (meta.word_cnt >= 50) & meta.year.between(lo, hi)]
    g = pd.DataFrame(Z[pol.index.to_numpy()]).assign(name=pol.full_name.values,
                                                     party=pol.party.values)
    cnt = g.groupby("name").size()
    g = g[g.name.isin(cnt[cnt >= 5].index)]
    spk = g.groupby("name").agg({**{c: "mean" for c in range(M)}, "party": "first"})
    return spk[list(range(M))].to_numpy(), (spk.party == "Republican").to_numpy()


def main():
    Z = np.load(OUT / "codes_all.npy").astype(np.float32)
    meta = pd.read_parquet(DATA / "meta.parquet")

    # ---- panel A numbers: naive segment level vs speaker level (all years) ----
    pol = meta[(meta.ID == "politician") & meta.party.isin(["Democratic", "Republican"])]
    Zp, T = Z[pol.index.to_numpy()], (pol.party == "Republican").to_numpy()
    _, p_naive = ttest_ind(Zp[T], Zp[~T], axis=0, equal_var=False)
    g = pd.DataFrame(Zp).assign(name=pol.full_name.values, party=pol.party.values)
    spk = g.groupby("name").agg({**{c: "mean" for c in range(M)}, "party": "first"})
    Zs_all, Ts_all = spk[list(range(M))].to_numpy(), (spk.party == "Republican").to_numpy()
    _, p_spk = ttest_ind(Zs_all[Ts_all], Zs_all[~Ts_all], axis=0, equal_var=False)

    # ---- panel B/C: era split, substantive segments ----
    Z1, T1 = speaker_table(meta, Z, 1951, 1994)
    Z2, T2 = speaker_table(meta, Z, 1995, 2023)
    t_pre, _ = ttest_ind(Z1[T1], Z1[~T1], axis=0, equal_var=False)
    t_post, _ = ttest_ind(Z2[T2], Z2[~T2], axis=0, equal_var=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9),
                             gridspec_kw={"width_ratios": [0.8, 1.1, 1.3]})
    fig.patch.set_facecolor("white")

    # ---------- A ----------
    ax = axes[0]
    levels = ["segment level\n(29.5k segments)", "speaker level\n(471 people)"]
    p05 = [(p_naive < 0.05).sum(), (p_spk < 0.05).sum()]
    bonf = [(p_naive < 0.05 / M).sum(), (p_spk < 0.05 / M).sum()]
    x = np.arange(2)
    ax.bar(x - 0.17, p05, 0.30, color="#b9c3d6", label="p < .05")
    ax.bar(x + 0.17, bonf, 0.30, color="#2b3a67", label="Bonferroni")
    ax.axhline(50, ls=":", lw=1.2, color=MUTED)
    ax.text(1.35, 52, "chance (50)", fontsize=7.5, color=MUTED, ha="right")
    for xi, v in zip(x - 0.17, p05):
        ax.text(xi, v + 2, str(v), ha="center", fontsize=8.5, color=INK)
    for xi, v in zip(x + 0.17, bonf):
        ax.text(xi, v + 2, str(v), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=8.5)
    ax.set_ylabel("codes flagged (of 1,000)")
    ax.set_title("A — ignoring dependence inflates discoveries", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- B ----------
    ax = axes[1]
    bins = np.linspace(-4.5, 4.5, 37)
    ax.hist(t_pre, bins=bins, histtype="step", lw=1.8, color=MUTED,
            label="1951–1994  (n=123)")
    ax.hist(t_post, bins=bins, histtype="step", lw=1.8, color="#2b3a67",
            label="1995–2023  (n=180)")
    for v in (-1.96, 1.96):
        ax.axvline(v, ls=":", lw=1.0, color=MUTED)
    ax.text(2.1, 46, "|t| = 1.96", fontsize=7.5, color=MUTED)
    for j, ytxt in ((216, 22), (510, 32)):
        ax.annotate(f"code {j}", xy=(t_post[j], 1.5), xytext=(t_post[j], ytxt),
                    fontsize=7.5, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=INK))
    ax.set_xlabel("speaker-level t statistic (R − D), per code")
    ax.set_ylabel("codes")
    ax.set_title("B — partisan separation exists only post-1995", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- C ----------
    ax = axes[2]
    rng = np.random.default_rng(0)
    order = [216, 235, 837, 510]
    for row, j in enumerate(order):
        z = (Z2[:, j] - Z2[:, j].mean()) / (Z2[:, j].std() + 1e-12)
        for pty, mask, color, dy in [("D", ~T2, BLUE, +0.16), ("R", T2, RED, -0.16)]:
            y = row + dy + rng.uniform(-0.08, 0.08, mask.sum())
            ax.scatter(z[mask], y, s=9, color=color, alpha=0.45, lw=0)
            ax.scatter(z[mask].mean(), row + dy, s=90, color=color, zorder=3,
                       edgecolor="white", lw=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"code {j}\n{CODE_LABELS[j]}" for j in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, lw=0.8, color="#cccccc", zorder=0)
    ax.set_xlabel("speaker mean activation (z-scored per code)")
    ax.set_title("C — top codes, 1995–2023: one dot per politician", fontsize=10, loc="left")
    hD = ax.scatter([], [], s=40, color=BLUE, label="Democrat")
    hR = ax.scatter([], [], s=40, color=RED, label="Republican")
    ax.legend(handles=[hD, hR], frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("SAE partisan channels in Fed & banking hearings — MVP", y=1.02, fontsize=12)
    fig.tight_layout()
    (OUT / "picture").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "picture" / "mvp_figure.png", dpi=170, bbox_inches="tight")
    print("wrote output/picture/mvp_figure.png")


if __name__ == "__main__":
    main()
