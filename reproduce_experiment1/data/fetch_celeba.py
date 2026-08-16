"""Fetch the canonical CelebA aligned-image archive plus its metadata from Google Drive.

The two small text files download directly. The 1.4 GB image zip hits Drive's virus-scan
interstitial first, so the confirm form has to be parsed and resubmitted.
"""

import hashlib
import html
import re
import sys
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent
FILES = {
    "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    "meta/list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
    "meta/list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
}
# md5 of img_align_celeba.zip as published by torchvision
KNOWN_MD5 = {"img_align_celeba.zip": "00d2c5bc6d35e252742224ab0c1e8fcb"}


def resolve(session, fid):
    """Return (url, params) that actually serve bytes, clearing the confirm page if shown."""
    url, params = "https://drive.google.com/uc?export=download", {"id": fid}
    r = session.get(url, params=params, stream=True, timeout=30)
    if "text/html" not in r.headers.get("Content-Type", ""):
        return r
    body = session.get(url, params=params, timeout=30).text
    if "quota" in body.lower():
        sys.exit("Drive quota exceeded — retry later or use a mirror")
    action = html.unescape(re.search(r'action="([^"]+)"', body).group(1))
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', body))
    return session.get(action, params=fields, stream=True, timeout=60)


def fetch(name, fid):
    dest = DATA / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    r = resolve(session, fid)
    total = int(r.headers.get("Content-Length", 0))
    if dest.exists() and dest.stat().st_size == total:
        print(f"{name}: already complete ({total/1e6:.0f} MB)")
        return dest
    got, md5 = 0, hashlib.md5()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            md5.update(chunk)
            got += len(chunk)
            if total and got % (50 << 20) < (1 << 20):
                print(f"{name}: {got/1e6:7.0f} / {total/1e6:.0f} MB", flush=True)
    print(f"{name}: done {got/1e6:.0f} MB  md5={md5.hexdigest()}")
    if name in KNOWN_MD5 and md5.hexdigest() != KNOWN_MD5[name]:
        print(f"  WARNING: md5 mismatch, expected {KNOWN_MD5[name]}")
    return dest


if __name__ == "__main__":
    for name, fid in FILES.items():
        fetch(name, fid)
