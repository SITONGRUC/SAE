"""saelib — vendored SAE + NES code for the partisan hearing-speech pilot.

Both modules are verbatim copies from `../reproduce_experiment1/` (copied 2026-08-16),
where they were validated against the ECI paper's Figures 3 and 5. That folder is the
tested original; fix bugs there first, then re-copy. Copied rather than imported so this
subrepo stands alone.

    sae.py  — tied top-k sparse autoencoder (encode/decode with pre-encoder bias)
    nes.py  — Neural Effect Search + baseline selection procedures (pure numpy)
"""

from saelib.sae import encode, decode
from saelib.nes import nes, baselines, stratified_test, score
