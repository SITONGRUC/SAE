"""Tied top-k SAE, encode/decode only.

Copied from `../reproduce_experiment1/train_sae.py` (2026-08-16), where this exact code
trained the SAE that exceeded the ECI paper's CelebA alignment (balanced-prevalence F1
0.938/0.936 vs the paper's 0.748/0.841). Design notes inherited from there:

- One dictionary with unit-normalised rows serves as both encoder and decoder (tied),
  with a single pre-encoder bias: encode(x - b), decode(code) + b. The bias is mandatory —
  without it, mean-carrier atoms hog the top-k slots and the dictionary degenerates.
- Top-k is the nonlinearity; no ReLU after it (not in the paper, and verified to be a
  no-op on both CelebA/SigLIP and the hearing embeddings).
"""

import torch


def encode(x, dictionary, k, b_dec=None):
    normed = dictionary / dictionary.norm(dim=-1, keepdim=True)
    xc = x if b_dec is None else x - b_dec
    scores = xc @ normed.T
    top = scores.topk(k, dim=-1)
    code = torch.zeros_like(scores).scatter_(-1, top.indices, top.values)
    return code, normed


def decode(code, normed, b_dec=None):
    out = code @ normed
    return out if b_dec is None else out + b_dec
