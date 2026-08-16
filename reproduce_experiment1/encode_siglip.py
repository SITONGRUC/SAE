"""Step 2 — encode CelebA val+test with SigLIP.

Appendix D.1: "Each image x is encoded with SigLIP into a patch-level representation; we use
the final-layer token features (dim d=768, 196 patches/token positions averaged)."

`d=768` with 196 patches pins google/siglip-base-patch16-224 (224/16 = 14, 14**2 = 196).
The paper only cites Zhai et al. 2023, so the checkpoint is inferred, not quoted.

Patch-level features are written to a memmap and the mean-pooled view is derived from them.
The paper is ambiguous about where pooling happens — D.1 says patches are averaged on the FM
side *and* that codes are mean-pooled after the SAE, which only both make sense if the SAE
runs per patch. Storing patches keeps both readings available off one encoding pass; pooling
is a mean over axis 1 and costs nothing.

Outputs (features/):
    patches_{split}.f16      (n, 196, 768)  memmap
    pooled_{split}.npy       (n, 768)
    index_{split}.parquet    file order, so rows can be joined back to attributes
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "img_align_celeba"
META = ROOT / "data" / "meta" / "celeba_meta.parquet"
OUT = ROOT / "features"
MODEL_ID = "google/siglip-base-patch16-224"
N_PATCHES, D_MODEL = 196, 768


class CelebAImages(Dataset):
    """Preprocessing done here rather than in the processor so DataLoader workers can decode
    JPEGs in parallel. Parameters are read off the processor config, not hardcoded."""

    def __init__(self, files, size, mean, std, resample):
        self.files = list(files)
        self.size, self.resample = size, resample
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(IMAGES / self.files[i]) as im:
            im = im.convert("RGB").resize(self.size, self.resample)
            x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((x - self.mean) / self.std)


def encode(split, model, device, ds_kwargs, batch_size, workers, dtype):
    meta = pd.read_parquet(META)
    sub = meta[meta.split == split].reset_index(drop=True)
    n = len(sub)

    OUT.mkdir(exist_ok=True)
    sub[["file"]].to_parquet(OUT / f"index_{split}.parquet", index=False)
    path = OUT / f"patches_{split}.f16"
    patches = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(n, N_PATCHES, D_MODEL)
    )

    loader = DataLoader(
        CelebAImages(sub.file, **ds_kwargs),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False,
    )

    done = 0
    for batch in loader:
        with torch.no_grad():
            h = model(pixel_values=batch.to(device, dtype)).last_hidden_state
        if h.shape[1:] != (N_PATCHES, D_MODEL):
            raise SystemExit(f"unexpected feature shape {tuple(h.shape)}")
        patches[done : done + len(h)] = h.float().cpu().numpy().astype(np.float16)
        done += len(h)
        if done % (batch_size * 40) < batch_size:
            print(f"  {split}: {done:,}/{n:,}", flush=True)

    patches.flush()
    pooled = np.asarray(patches).astype(np.float32).mean(axis=1).astype(np.float16)
    np.save(OUT / f"pooled_{split}.npy", pooled)
    print(f"{split}: {n:,} images -> {path.name} "
          f"({path.stat().st_size/1e9:.2f} GB) + pooled_{split}.npy ({pooled.nbytes/1e6:.0f} MB)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N images")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    cfg = proc.to_dict()
    size = (cfg["size"]["width"], cfg["size"]["height"])
    ds_kwargs = dict(
        size=size,
        mean=cfg["image_mean"],
        std=cfg["image_std"],
        resample=Image.BICUBIC,
    )

    model = SiglipVisionModel.from_pretrained(MODEL_ID).to(device, dtype).eval()
    c = model.config
    assert c.hidden_size == D_MODEL and (c.image_size // c.patch_size) ** 2 == N_PATCHES
    print(f"{MODEL_ID} on {device}/{dtype} | {size} | mean={cfg['image_mean']} std={cfg['image_std']}")

    if args.limit:
        smoke(model, device, dtype, ds_kwargs, args.limit)
        return
    for split in args.splits:
        encode(split, model, device, ds_kwargs, args.batch_size, args.workers, dtype)


def smoke(model, device, dtype, ds_kwargs, limit):
    """Encode a few images and check fp16 on MPS agrees with fp32 on CPU."""
    files = pd.read_parquet(META).query("split == 'val'").file.head(limit)
    ds = CelebAImages(files, **ds_kwargs)
    x = torch.stack([ds[i] for i in range(len(ds))])
    with torch.no_grad():
        fast = model(pixel_values=x.to(device, dtype)).last_hidden_state.float().cpu()
        ref = model.float().cpu()(pixel_values=x).last_hidden_state
    err = (fast - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"shape {tuple(fast.shape)} | max abs dev {err:.4f} | relative {rel:.2%}")


if __name__ == "__main__":
    main()
