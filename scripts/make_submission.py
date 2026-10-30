"""Build the capstone submission zip (source code + documentation).

Usage:  python scripts/make_submission.py [output.zip]

Excludes virtual environments, the local vector store, caches and secrets.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {".venv", "venv", ".git", "data", "__pycache__", ".pytest_cache", "dist"}
EXCLUDED_FILES = {".env", "secrets.toml"}


def included_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_dir() or any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.name in EXCLUDED_FILES or path.suffix in {".zip", ".pyc"}:
            continue
        yield path, relative


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "Edureka_Capstone_submission.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, relative in included_files():
            archive.write(path, Path("Edureka_Capstone") / relative)
            count += 1
    print(f"Wrote {count} files to {output}")


if __name__ == "__main__":
    main()
