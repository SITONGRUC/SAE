# Step 1 — CelebA acquisition

What this step produced, where every byte came from, and how to rebuild it from nothing.
Step 0 of the pipeline.

---

## 1. Where the data comes from

CelebA has no official HTTP endpoint. The dataset page
([mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
links to a Google Drive folder, and that Drive folder is what every library — torchvision
included — actually downloads from. So the canonical source is a set of Drive file IDs.

Those IDs are stable and are the same ones hardcoded in
`torchvision.datasets.CelebA`:

| File | Drive file ID | Size |
|---|---|---|
| `img_align_celeba.zip` | `0B7EVK8r0v71pZjFTYXZWM3FlRnM` | 1.443 GB |
| `list_attr_celeba.txt` | `0B7EVK8r0v71pblRyaVFSWGxPY0U` | 27 MB |
| `list_eval_partition.txt` | `0B7EVK8r0v71pY0NSMzRuSXJEVkk` | 2.8 MB |

`fetch_celeba.py` requests each as
`https://drive.google.com/uc?export=download&id=<ID>`.

**The one wrinkle.** The two text files stream straight back. The 1.4 GB zip does not — Drive
cannot virus-scan a file that size, so it returns an HTML interstitial instead of bytes. The
fix is to parse that page's form (`action`, plus the hidden `id` / `export` / `confirm` /
`uuid` fields) and resubmit it, which redirects to `drive.usercontent.google.com/download`.
That logic lives in `fetch_celeba.py:resolve()`.

Two failure modes look similar and are worth telling apart:

- **Interstitial** — the recoverable case handled above.
- **Quota exceeded** — Drive refusing service because the file has been pulled too many times
  globally that day. `resolve()` detects this and exits rather than writing an HTML file to
  disk under a `.zip` name. The fix is to wait, or use a mirror (see the warning in §3).

## 2. How to reproduce

From nothing, two commands, ~12 minutes:

```bash
python3 reproduce_experiment1/data/fetch_celeba.py     # downloads all three files, checks the archive md5
python3 reproduce_experiment1/data/build_splits.py     # builds the tidy metadata table, asserts split sizes
unzip -q reproduce_experiment1/data/img_align_celeba.zip -d reproduce_experiment1/data/   # ~21 s
```

`fetch_celeba.py` skips any file already complete on disk, so it is safe to re-run.

## 3. The mirror trap — read before "optimising" this

The obvious shortcut is a HuggingFace mirror, which would avoid the Drive dance entirely and
let us pull only the ~40k images we need instead of all 202,599. **It was rejected, and the
reason is not fussiness.**

`tpremoli/CelebA-attrs` is the best-looking candidate: images plus all 40 attributes, already
split three ways, and the counts match the official partition exactly. But:

| | official | tpremoli |
|---|---|---|
| train | 162,770 | 162,770 ✓ |
| validation | 19,867 | **19,962** |
| test | 19,962 | **19,867** |

**Validation and test are swapped.** Probably a `{train:0, validation:2, test:1}` mismap during
conversion. Since the plan trains the SAE on val and runs the causal experiment on test,
accepting those labels would have inverted the paper's setup — and nothing downstream would
have looked wrong. The SAE would still train, the F1 scores would still be plausible, the
figures would still render. It would just quietly not be the paper's experiment.

Hence the rule now baked into `build_splits.py`: **the split comes from
`list_eval_partition.txt` and nowhere else**, and the resulting sizes are hard-asserted against
`{train: 162770, val: 19867, test: 19962}`. Any mirror that disagrees fails loudly.

The other mirrors were rejected for simpler reasons — `nielsr/CelebA-faces` and
`huggan/CelebA-faces-with-attributes` ship a single `train` split with no partition at all, and
`flwrlabs/celeba` uses a federated-learning split that is not the official one.

## 4. What was verified

| Check | Result |
|---|---|
| Archive md5 vs torchvision's published value | `00d2c5bc6d35e252742224ab0c1e8fcb` ✓ byte-identical to canonical |
| Attribute file and partition file agree on image order | ✓ all 202,599 rows |
| Split sizes vs official partition | ✓ 162,770 / 19,867 / 19,962 |
| Partition contiguous by image index | ✓ see §5 |
| Extracted files vs manifest | ✓ exact set match, 0 missing, 0 extra |
| Image geometry (300 sampled) | ✓ all 178×218 RGB, all open cleanly |
| Attribute values, raw file → parquet | ✓ spot-checked |

The md5 match matters more than it looks: it proves these are the original JPEGs, not
re-encoded copies. A mirror that decoded and re-saved the images would shift every SigLIP
feature slightly, which is exactly the kind of thing that is invisible until results disagree
with the paper by a few percent and there is no way to tell why.

## 5. The official split is contiguous

Confirmed empirically, not assumed:

| Split | Image range | n | Used for |
|---|---|---|---|
| train | `000001.jpg` – `162770.jpg` | 162,770 | **nothing** (see §7) |
| val | `162771.jpg` – `182637.jpg` | 19,867 | SAE training |
| test | `182638.jpg` – `202599.jpg` | 19,962 | the causal experiment |

Useful property: a split can be recovered from filenames alone if the metadata is ever lost.

## 6. Layout on disk

```
reproduce_experiment1/data/
├── fetch_celeba.py            downloader (tracked in git)
├── build_splits.py            metadata builder (tracked in git)
├── step1_download.md          this file (tracked in git)
├── img_align_celeba.zip       1.443 GB   raw archive
├── img_align_celeba/          1.410 GB   202,599 JPEGs, 178×218 RGB
└── meta/
    ├── list_attr_celeba.txt        27 MB
    ├── list_eval_partition.txt    2.8 MB
    └── celeba_meta.parquet        ~1 MB   file + split + 40 attrs as 0/1
```

Everything except the three scripts/docs is gitignored — the data is reproducible, and 2.9 GB
does not belong in a git history.

`celeba_meta.parquet` is the thing downstream code should read. It carries `file`, `split`, and
the 40 attributes recoded from `{-1, +1}` to `{0, 1}`.

## 7. Can we delete the train split?

**Yes eventually, but not yet.** Here is the accounting.

| Item | Size | Status |
|---|---|---|
| `img_align_celeba.zip` | 1.443 GB | **redundant — delete now** |
| train images (162,770) | 1.132 GB | keep until step 3 clears |
| val + test images (39,829) | 0.278 GB | **required** |
| `meta/` | 0.030 GB | **required** |
| | | |
| current total | 2.88 GB | |
| strictly required | 0.31 GB | |
| reclaimable | 2.58 GB | |

**Delete the zip now.** It is pure duplication: extraction is deterministic, the extracted
files have been verified against the manifest, and if it is ever needed again `fetch_celeba.py`
restores it with an md5 check. Saves 1.443 GB at zero cost in optionality.

Note the asymmetry — keeping the zip "in case we need train later" is strictly worse than
keeping the extracted train split, because the zip (1.443 GB) is *larger* than the train images
it contains (1.132 GB) and needs a 21-second extraction on top.

**Keep the train split until the SAE clears its F1 checkpoint.** This is the part worth being
careful about. The replication plan set a stop rule: if the best F1 against the attribute labels lands
below 0.6, do not tune hyperparameters — instead pull one of exactly two levers. One is
per-patch features. **The other is the unused train split, which is 8× more data.**

So "unused" in the plan means *not used on the intended path*, not *useless*. Deleting it now
would foreclose one of the two documented escape hatches, and the SAE has not been trained yet
— we do not know whether we will need it.

Cost of holding it: 1.132 GB out of ~249 GB free, about 0.45% of free space.
Cost of deleting it and being wrong: a 12-minute re-download plus re-extraction.

That trade only goes one way. Keep it until step 3 passes, then delete.

**After step 3 passes**, if F1 is comfortably above the threshold:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
meta = pd.read_parquet("reproduce_experiment1/data/meta/celeba_meta.parquet")
for f in meta[meta.split == "train"].file:
    (Path("reproduce_experiment1/data/img_align_celeba") / f).unlink(missing_ok=True)
PY
```

That leaves 0.31 GB, and `build_splits.py` still works — it reads only the metadata, never the
image directory.

**Do not delete `meta/`.** It is 30 MB and it is the only thing standing between this
replication and the split-swap bug in §3.

## 8. What comes next

Derived artifacts are small enough that none of this matters much once step 2 runs:

| Artifact | Size |
|---|---|
| SigLIP pooled features, 40k × 768 fp16 | 61 MB |
| SAE codes, test split, 19,962 × 9,216 fp32 | 736 MB |
| SigLIP per-patch features (fallback only) | **12 GB**, must be memmapped |

The 12 GB per-patch array is the only future item that needs planning, and it only materialises
if the F1 checkpoint fails.
