"""Fetch the pinned SatelliteSfM calibration source in the isolated workspace."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "908c61937f6a16d6e1d50e98898db997e5e257bd"
target = ROOT / ".external/satellitesfm"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "https://github.com/jayin92/SatelliteSfM.git",
            str(target),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", COMMIT], check=True
    )
actual = subprocess.check_output(
    ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
).strip()
if actual != COMMIT:
    raise SystemExit(
        "Existing SatelliteSfM checkout differs from the pinned revision; it was left unchanged"
    )
print(target)
