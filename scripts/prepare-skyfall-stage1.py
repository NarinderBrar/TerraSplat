"""Prepare an isolated pinned Skyfall checkout and apply the basic Stage 1 patch."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "73194842a404910c5bbff006359eedfc56a98d60"
EXTENSIONS = ["submodules/diff-gaussian-rasterization-depth", "submodules/simple-knn", "submodules/fused-ssim"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "Ref" / "Skyfall-GS")
    args = parser.parse_args()
    target = ROOT / ".external" / "skyfall-gs"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(["git", "clone", "--no-hardlinks", str(args.reference.resolve()), str(target)], check=True)
    def git(*arguments, **kwargs):
        return subprocess.run(["git", "-C", str(target), *arguments], check=True, **kwargs)
    actual = git("rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected Skyfall {COMMIT}, got {actual}; checkout left unchanged")
    patch = ROOT / "adapters" / "skyfall" / "stage1-only.patch"
    dirty = git("diff", "HEAD", "--", "train.py", "utils/camera_utils.py", capture_output=True, text=True).stdout
    if dirty:
        if dirty != patch.read_text():
            raise RuntimeError("Isolated train.py has unexpected changes; refusing to overwrite")
    else:
        git("apply", "--check", str(patch))
        git("apply", str(patch))
    git("submodule", "update", "--init", "--recursive", *EXTENSIONS)
    rasterizer = target / "submodules/diff-gaussian-rasterization-depth"
    header_patch = ROOT / "adapters/skyfall/rasterizer-cstdint.patch"
    changes = subprocess.check_output(["git", "-C", str(rasterizer), "diff", "HEAD", "--", "cuda_rasterizer/rasterizer_impl.h"], text=True)
    if changes:
        if changes != header_patch.read_text():
            raise RuntimeError("Rasterizer header has unexpected changes; refusing to overwrite")
    else:
        subprocess.run(["git", "-C", str(rasterizer), "apply", "--check", str(header_patch)], check=True)
        subprocess.run(["git", "-C", str(rasterizer), "apply", str(header_patch)], check=True)
    print(target)


if __name__ == "__main__":
    main()
