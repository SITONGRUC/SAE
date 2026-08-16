"""Step 2b — is SigLIP alone sufficient for the two affected outcomes?

Appendix E.1 tests Assumption A.1 by classifying the outcomes straight off the FM features
with vanilla logistic regression, before any SAE is involved:

    Wearing Hat   Precision = 0.9639   Recall = 0.9947
    Eyeglasses    Precision = 0.9712   Recall = 0.9747

The point is diagnostic. If these reproduce, the encoding is sound, and any later failure of
the SAE is the SAE's fault rather than the foundation model's. The paper does not say which
split it fits on; fitting on val and evaluating on test keeps the two disjoint and matches the
roles the splits play everywhere else in the replication.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PAPER = {"Wearing_Hat": (0.9639, 0.9947), "Eyeglasses": (0.9712, 0.9747)}


def load(split):
    Z = np.load(ROOT / "features" / f"pooled_{split}.npy").astype(np.float32)
    idx = pd.read_parquet(ROOT / "features" / f"index_{split}.parquet")
    meta = pd.read_parquet(ROOT / "data" / "meta" / "celeba_meta.parquet")
    sub = meta[meta.split == split].reset_index(drop=True)
    if not (idx.file.values == sub.file.values).all():
        raise SystemExit(f"{split}: feature rows do not line up with the metadata")
    return Z, sub


def main():
    Ztr, tr = load("val")
    Zte, te = load("test")
    scaler = StandardScaler().fit(Ztr)
    Xtr, Xte = scaler.transform(Ztr), scaler.transform(Zte)

    print(f"fit on val (n={len(tr):,}), evaluate on test (n={len(te):,})\n")
    print(f"{'attribute':14s} {'prev':>6s} {'prec':>7s} {'rec':>7s} {'F1':>7s} {'AUC':>7s}   paper (P/R)")
    rows = []
    for attr, (p_paper, r_paper) in PAPER.items():
        ytr, yte = tr[attr].values, te[attr].values
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        p = precision_score(yte, pred, zero_division=0)
        r = recall_score(yte, pred, zero_division=0)
        print(f"{attr:14s} {yte.mean():6.3f} {p:7.4f} {r:7.4f} "
              f"{f1_score(yte, pred, zero_division=0):7.4f} {roc_auc_score(yte, prob):7.4f}   "
              f"{p_paper:.4f} / {r_paper:.4f}")
        rows.append((attr, p, r, p_paper, r_paper))

    print()
    for attr, p, r, pp, rp in rows:
        verdict = "matches" if abs(p - pp) < 0.05 and abs(r - rp) < 0.05 else "DIVERGES"
        print(f"  {attr:14s} dP={p-pp:+.4f} dR={r-rp:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
