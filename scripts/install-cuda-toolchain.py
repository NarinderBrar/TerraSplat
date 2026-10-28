"""Install checksum-verified CUDA compiler components inside this workspace."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = {
    "cuda_nvcc": "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3",
    "cuda_cudart": "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68",
    "cuda_cccl": "0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b",
}


def main():
    cache = ROOT / ".cache" / "cuda-redist"
    target = ROOT / ".toolchains" / "cuda-12.8"
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + "redistrib_12.8.1.json", timeout=60) as response:
        manifest = json.load(response)
    for name, expected in COMPONENTS.items():
        component = manifest[name]["linux-x86_64"]
        if component["sha256"] != expected:
            raise ValueError(f"Upstream manifest checksum changed for {name}")
        archive = cache / Path(component["relative_path"]).name
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            print(f"Downloading {name}: {int(component['size']) / 1024**2:.1f} MiB", flush=True)
            with urllib.request.urlopen(BASE + component["relative_path"], timeout=120) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.rename(archive)
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            with tarfile.open(archive) as package:
                package.extractall(temporary, filter="data")
            folders = list(Path(temporary).iterdir())
            if len(folders) != 1 or not folders[0].is_dir():
                raise ValueError(f"Unexpected archive layout: {archive}")
            shutil.copytree(folders[0], target, dirs_exist_ok=True, symlinks=True)
        print(f"Installed {name} ({expected[:12]})", flush=True)
    (target / "redistrib.json").write_text(json.dumps({name: manifest[name] for name in COMPONENTS}, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
