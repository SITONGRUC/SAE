"""Neural Effect Search and the baseline selection procedures.

Copied from `../reproduce_experiment1/nes.py` (2026-08-16); that copy is the tested
original, validated against analytic ground truth in the Figure 3 notebook and used for
the Figure 5 replication. Three details follow the authors' shipped code rather than the appendix pseudocode:

- cutpoints are computed pooled over T, so the strata cannot encode the treatment
- multiple rounds stratify jointly on everything in S, giving 2**|S| cells
- strata missing an arm are dropped and the weights renormalised over what remains

Selection is two criteria in order: keep whatever clears alpha/m, then take the largest
|tau_hat| among the survivors. Stop when nothing clears.
"""

import numpy as np
from scipy.stats import t as student_t

ALPHA = 0.05


def stratified_test(T, Z, S=(), threshold=0.0):
    """Weighted stratified treated-vs-control contrast for every column of Z.

    Returns (ate, pval) of length m. Columns with no usable stratum come back as NaN.
    """
    n, m = Z.shape
    S = list(S)
    if not S:
        gid = np.zeros(n, dtype=np.int64)
    else:
        Sbin = (Z[:, S] > threshold).astype(np.int64)
        gid = Sbin @ (1 << np.arange(len(S), dtype=np.int64))

    num = np.zeros(m)
    var = np.zeros(m)
    wsum = 0.0
    for g in np.unique(gid):
        cell = gid == g
        t1, t0 = cell & (T == 1), cell & (T == 0)
        n1, n0 = t1.sum(), t0.sum()
        if n1 < 2 or n0 < 2:          # appendix guard; ddof=1 needs two per arm
            continue
        w = cell.sum() / n
        d = Z[t1].mean(0) - Z[t0].mean(0)
        v = Z[t1].var(0, ddof=1) / n1 + Z[t0].var(0, ddof=1) / n0
        num += w * d
        var += (w ** 2) * v
        wsum += w

    if wsum == 0:
        return np.full(m, np.nan), np.full(m, np.nan)
    ate = num / wsum
    se = np.sqrt(var) / wsum
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = ate / se
    p = 2 * student_t.sf(np.abs(tstat), df=max(n - 2, 1))
    p[~np.isfinite(tstat)] = np.nan
    return ate, p


def bh_reject(p, alpha=ALPHA):
    """Benjamini-Hochberg step-up. NaN p-values never reject."""
    ok = np.isfinite(p)
    rej = np.zeros(len(p), dtype=bool)
    if not ok.any():
        return rej
    idx = np.where(ok)[0]
    q = p[idx]
    order = np.argsort(q)
    m = len(q)
    passed = q[order] <= alpha * np.arange(1, m + 1) / m
    if passed.any():
        rej[idx[order[: np.max(np.where(passed)[0]) + 1]]] = True
    return rej


def nes(T, Z, alpha=ALPHA, max_cells=4096, max_steps=25):
    """Neural Effect Search. Returns the list of selected column indices."""
    m = Z.shape[1]
    S = []
    for _ in range(max_steps):
        if 2 ** (len(S) + 1) > max_cells:
            break
        ate, p = stratified_test(T, Z, S)
        ok = np.isfinite(p) & (p < alpha / m)
        ok[S] = False
        if not ok.any():
            break
        cand = np.where(ok)[0]
        S.append(int(cand[np.argmax(np.abs(ate[cand]))]))
    return S


def baselines(T, Z, alpha=ALPHA, top_k=2):
    """The four comparison procedures, all off one pass of unstratified tests."""
    ate, p = stratified_test(T, Z)
    finite = np.isfinite(p)
    out = {
        "t-test": np.where(finite & (p < alpha))[0],
        "Bonferroni": np.where(finite & (p < alpha / Z.shape[1]))[0],
        "FDR": np.where(bh_reject(p, alpha))[0],
    }
    score = np.where(finite, np.abs(ate), -np.inf)
    out["top-k"] = np.argsort(score)[::-1][:top_k]
    return out


def score(selected, truth):
    """Precision / Recall / IoU of a discovered set against the ground truth set."""
    S, G = set(int(s) for s in selected), set(int(g) for g in truth)
    tp = len(S & G)
    return (
        tp / len(S) if S else 0.0,
        tp / len(G) if G else 0.0,
        tp / len(S | G) if (S | G) else 0.0,
    )
