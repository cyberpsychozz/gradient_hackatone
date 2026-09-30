"""Build source-only submission ZIPs for the two selected PPO variants.

Ready-made ONNX models are deliberately excluded: the platform runs train.py
inside the submitted Docker image and accepts only its /output/policy.onnx.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXCLUDED_PARTS = {"artifacts", "build", "__pycache__", ".pytest_cache", "tests"}
EXTENSIONS = {".py", ".hpp", ".cpp", ".yaml", ".toml", ".md", ".txt", ".inc"}


def source_files() -> list[Path]:
    files = []
    for path in HERE.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(HERE)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.suffix in EXTENSIONS or path.name in {"Dockerfile", "CMakeLists.txt"}:
            files.append(path)
    return sorted(files)


def dockerfile(variant: str) -> str:
    source = (HERE / "Dockerfile").read_text()
    if variant == "shared_ppo":
        args = ["python", "/submission/train.py", "--algorithm", "shared_ppo",
                "--seed", "2027", "--max-seconds", "6300",
                "--eval-interval", "1200"]
    elif variant == "wide_ppo":
        args = ["python", "/submission/train.py", "--algorithm", "separate_ppo",
                "--seed", "2028", "--extra-actions", "5,6,7,12,14,15,16,17,23,24,26,28",
                "--max-seconds", "6300", "--eval-interval", "1200"]
    elif variant == "context_ppo":
        args = ["python", "/submission/train.py", "--algorithm", "separate_ppo",
                "--seed", "2029", "--extra-actions", "12,14,16,17,23,24,28",
                "--extra-logit-bias", "-4.0", "--max-seconds", "6300",
                "--eval-interval", "1200"]
    else:
        raise ValueError(variant)
    lines = [line for line in source.splitlines() if not line.startswith("CMD ")]
    return "\n".join(lines + ["CMD " + json.dumps(args)]) + "\n"


def package(variant: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            name = source.relative_to(HERE).as_posix()
            if name == "Dockerfile":
                archive.writestr(name, dockerfile(variant))
            else:
                archive.write(source, name)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        required = {"Dockerfile", "train.py", "README.md"}
        if not required.issubset(names):
            raise RuntimeError(f"missing required files: {required - set(names)}")
        if len(names) > 512 or sum(item.file_size for item in archive.infolist()) > 128 * 1024**2:
            raise RuntimeError("unpacked source exceeds platform limits")
        forbidden = (".onnx", ".pt", ".zip", ".so", ".log")
        if any(name.endswith(forbidden) or name.startswith("artifacts/") for name in names):
            raise RuntimeError("archive contains binary artifacts or nested ZIP")
    if output.stat().st_size > 32 * 1024**2:
        raise RuntimeError("ZIP exceeds platform size limit")
    print(f"{variant}: {output} ({len(names)} source files, {output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "best_solutions")
    args = parser.parse_args()
    for variant in ("context_ppo", "wide_ppo"):
        package(variant, args.output_dir / f"rover_{variant}.zip")


if __name__ == "__main__":
    main()
