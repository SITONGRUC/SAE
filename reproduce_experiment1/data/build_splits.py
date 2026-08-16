"""Turn the raw CelebA metadata into one tidy table keyed by the official split.

Produces `data/meta/celeba_meta.parquet` with columns:
    file, split ('train'|'val'|'test'), and the 40 attributes as 0/1 (not -1/+1).

The split is taken from `list_eval_partition.txt`, not inferred. Several HuggingFace
mirrors of CelebA swap val and test, so the raw partition file is the only thing trusted
here. Codes: 0=train, 1=val (19,867), 2=test (19,962).

Per the replication plan the SAE trains on **val** and the causal experiment runs on **test**.
"""

from pathlib import Path

import pandas as pd

META = Path(__file__).resolve().parent / "meta"
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
EXPECTED = {"train": 162770, "val": 19867, "test": 19962}
# the DGP in Appendix D.1 uses exactly these three
KEY_ATTRS = ["Eyeglasses", "Wearing_Hat", "Smiling"]


def build():
    part = pd.read_csv(
        META / "list_eval_partition.txt", sep=r"\s+", header=None, names=["file", "code"]
    )
    attr = pd.read_csv(META / "list_attr_celeba.txt", sep=r"\s+", header=1)
    attr.index.name = "file"
    attr = attr.reset_index()

    if not (attr["file"].values == part["file"].values).all():
        raise SystemExit("attribute and partition files disagree on image order")

    df = attr.copy()
    df.insert(1, "split", part["code"].map(SPLIT_NAMES).values)
    attr_cols = [c for c in df.columns if c not in ("file", "split")]
    df[attr_cols] = (df[attr_cols] > 0).astype("int8")

    counts = df["split"].value_counts().to_dict()
    if counts != EXPECTED:
        raise SystemExit(f"split sizes {counts} != official {EXPECTED}")

    out = META / "celeba_meta.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  {df.shape[0]:,} rows x {df.shape[1]} cols")
    for s in ("val", "test"):
        sub = df[df.split == s]
        rates = "  ".join(f"P({a})={sub[a].mean():.4f}" for a in KEY_ATTRS)
        print(f"  {s:5s} n={len(sub):,}  {rates}")

    # joint cells the Figure 5 DGP has to sample from, on the causal-experiment split
    test = df[df.split == "test"]
    cells = test.groupby(KEY_ATTRS).size().rename("n").reset_index()
    print("\njoint (Eyeglasses, Wearing_Hat, Smiling) cells on test:")
    print(cells.to_string(index=False))
    return df


if __name__ == "__main__":
    build()
