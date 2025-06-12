#!/usr/bin/env python3
"""Update the benchmark commit badge in results/README.md."""
from pathlib import Path
import subprocess


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    readme = Path("results/README.md")
    lines = readme.read_text().splitlines()
    badge = f"![Benchmarks](https://img.shields.io/badge/benchmarks-{sha}-blue)"

    if lines and lines[0].startswith("![Benchmarks]"):
        lines[0] = badge
    else:
        lines.insert(0, badge)
        lines.insert(1, "")

    readme.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
