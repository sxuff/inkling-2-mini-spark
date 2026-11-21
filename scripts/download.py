#!/usr/bin/env python3
"""Download the pinned Inkling 2 Mini checkpoint and verify every file (size, and SHA-256 for LFS files).

usage: python3 scripts/download.py [DEST]   (default ~/models/Inkling-2-Mini-ed40e7d)
Needs curl. Resumes partial files. About 20.2 GB.
"""
import hashlib, json, os, pathlib, subprocess, sys, urllib.parse, urllib.request

REPO = "thinkingmachines/Inkling-2-Mini"
REV = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
BASE = "https://huggingface.co"
dest = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / f"models/Inkling-2-Mini-{REV[:7]}")
dest.mkdir(parents=True, exist_ok=True)
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}

req = urllib.request.Request(f"{BASE}/api/models/{REPO}/revision/{REV}?blobs=true", headers=headers)
with urllib.request.urlopen(req, timeout=60) as r:
    info = json.load(r)
assert info["sha"] == REV, info["sha"]
files = [s for s in info["siblings"] if s["rfilename"] != ".gitattributes"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(path, spec):
    if not path.exists() or path.stat().st_size != spec["size"]:
        return False
    return "lfs" not in spec or sha256(path) == spec["lfs"]["sha256"]


for spec in files:
    name = spec["rfilename"]
    path = dest / name
    if ok(path, spec):
        print(f"ok   {name}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    url = f"{BASE}/{REPO}/resolve/{REV}/{urllib.parse.quote(name)}?download=true"
    print(f"get  {name} ({spec['size'] / 1e9:.2f} GB)", flush=True)
    auth = ["-H", f"Authorization: {headers['Authorization']}"] if headers else []
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "3", "--continue-at", "-", *auth,
                    "--output", str(part), url], check=True)
    part.replace(path)
    if not ok(path, spec):
        sys.exit(f"verification failed for {name}")
    print(f"ok   {name}")

(dest / "DOWNLOAD_VERIFIED.json").write_text(json.dumps(
    {"repo": REPO, "revision": REV, "files": {s["rfilename"]: s["size"] for s in files}}, indent=1) + "\n")
print(f"verified {len(files)} files in {dest}")
