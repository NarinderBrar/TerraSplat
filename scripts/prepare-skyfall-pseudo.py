"""Prepare the separate, recorded pseudo-depth adapter without altering basic Stage 1."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / ".external/skyfall-gs-pseudo"
    if target.exists():
        raise ValueError("Use a new pseudo-depth checkout")
    subprocess.run(["git", "clone", "--no-hardlinks", str(ROOT / ".external/skyfall-gs"), str(target)], check=True)
    for name in ("stage1-pseudo-depth.patch", "prepared-enu.patch"):
        subprocess.run(["git", "-C", str(target), "apply", str(ROOT / "adapters/skyfall" / name)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "submodules/diff-gaussian-rasterization-depth",
            "submodules/simple-knn",
            "submodules/fused-ssim",
        ],
        check=True,
    )
    print(target)


if __name__ == "__main__":
    main()
