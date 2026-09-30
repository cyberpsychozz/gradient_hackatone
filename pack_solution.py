"""Pack a solution folder into a platform-ready ZIP.

The platform requires Dockerfile, train.py and README.md at the ARCHIVE ROOT
(no top-level folder), total size <= 32 MiB, and source files only: no
artifacts/, checkpoints (.pt/.onnx), __pycache__, .git or egg-info.

Usage:
    python pack_solution.py solutions/rover_best_v1 solutions/rover_best_v1.zip
"""

from __future__ import annotations

import os
import sys
import zipfile

SKIP_DIRS = {"artifacts", "__pycache__", ".git", ".egg-info", ".pytest_cache", "build", "dist"}
SKIP_EXTS = {".pyc", ".pt", ".onnx", ".pyd", ".so"}


def pack(source: str, destination: str) -> None:
    source = os.path.abspath(source)
    if not os.path.isdir(source):
        raise SystemExit(f"not a directory: {source}")
    files = []
    for root, dirs, names in os.walk(source):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, source).replace(os.sep, "/")
            if os.path.splitext(name)[1] in SKIP_EXTS:
                continue
            files.append((rel, path))
    files.sort()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel, path in files:
            zf.write(path, rel)
    size_mb = os.path.getsize(destination) / (1024 * 1024)
    with zipfile.ZipFile(destination) as zf:
        root_entries = [n for n in zf.namelist() if "/" not in n]
        required = ["Dockerfile", "train.py", "README.md"]
        missing = [r for r in required if r not in root_entries]
        weights = [n for n in zf.namelist() if n.endswith((".pt", ".onnx"))]
    status = "OK" if not missing and not weights and size_mb <= 32 else "FAIL"
    print(f"[{status}] {destination}  {size_mb:.2f} MiB, {len(files)} files")
    if missing:
        print("  missing at root:", missing)
    if weights:
        print("  weight files found:", weights)
    print("  root entries:", root_entries)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    pack(sys.argv[1], sys.argv[2])
