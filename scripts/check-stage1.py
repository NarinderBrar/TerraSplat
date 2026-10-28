"""Execute the three CUDA extensions used by Skyfall Stage 1."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".external" / "skyfall-gs"
sys.path.insert(0, str(SOURCE))
from utils.graphics_utils import getProjectionMatrix


def main():
    assert torch.cuda.is_available(), "Run with host GPU access"
    torch.manual_seed(0)
    points = torch.tensor([[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]], dtype=torch.float32, device="cuda")
    distances = distCUDA2(points)
    assert torch.isfinite(distances).all() and (distances > 0).all()
    image = torch.rand(1, 3, 32, 32, device="cuda", requires_grad=True)
    ssim = fused_ssim(image, image.detach().clone())
    ssim.backward()
    assert abs(ssim.item() - 1) < 1e-5 and torch.isfinite(image.grad).all()
    projection = getProjectionMatrix(0.01, 100, 1.0, 1.0, 0, 0).transpose(0, 1).cuda()
    settings = GaussianRasterizationSettings(
        image_height=32, image_width=32, tanfovx=0.5463024898, tanfovy=0.5463024898,
        kernel_size=0.1, subpixel_offset=torch.zeros(32, 32, 2, device="cuda"),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"), projmatrix=projection,
        sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False,
    )
    means = torch.tensor([[0., 0., 2.]], device="cuda", requires_grad=True)
    color = torch.tensor([[1., 0.2, 0.1]], device="cuda", requires_grad=True)
    rendered, depth, normal, alpha, radii, extra = GaussianRasterizer(settings)(
        means3D=means, means2D=torch.zeros_like(means, requires_grad=True),
        opacities=torch.tensor([[0.9]], device="cuda"), colors_precomp=color,
        scales=torch.tensor([[0.2, 0.2, 0.2]], device="cuda"),
        rotations=torch.tensor([[1., 0., 0., 0.]], device="cuda"),
    )
    rendered.sum().backward()
    assert rendered.max() > 0 and (radii > 0).any()
    assert means.grad is not None and color.grad is not None
    assert torch.isfinite(means.grad).all() and torch.isfinite(color.grad).all()
    torch.cuda.synchronize()
    result = subprocess.run([sys.executable, "train.py", "--help"], cwd=SOURCE, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for arguments in (["--iterative_datasets_update"], ["--lambda_pseudo_depth", "0.5"]):
        rejected = subprocess.run([sys.executable, "train.py", *arguments], cwd=SOURCE, capture_output=True, text=True)
        assert rejected.returncode == 2 and "requires IDU off" in rejected.stderr
    report = {
        "status": "passed", "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "capability": torch.cuda.get_device_capability(0),
        "checks": ["CUDA nearest neighbours", "SSIM forward/backward", "rasterizer forward/backward", "patched Stage 1 CLI import", "optional model modes rejected"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
        "patch_sha256": hashlib.sha256((ROOT / "adapters/skyfall/stage1-only.patch").read_bytes()).hexdigest(),
    }
    output = ROOT / "artifacts/stage1/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
