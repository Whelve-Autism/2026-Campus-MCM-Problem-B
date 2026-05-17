"""Run problem_1 -> problem_2 -> problem_3 from repo root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    cmds = [
        [sys.executable, "-m", "problem_1.run"],
        [sys.executable, "-m", "problem_2.run"],
        [sys.executable, "-m", "problem_3.run"],
    ]
    for c in cmds:
        print("+", " ".join(c))
        subprocess.run(c, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
