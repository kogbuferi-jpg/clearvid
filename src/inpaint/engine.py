"""Video inpainting engine wrappers.

MVP strategy: E2FGVI (fast, free tier) is invoked via its official repo
(clone into `third_party/E2FGVI`). ProPainter (paid tier) plugs in through
the same interface.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"


class E2FGVIEngine:
    """Wraps the official E2FGVI inference script."""

    name = "e2fgvi"

    def __init__(self, repo: Path | None = None, ckpt: str = "release_model/E2FGVI-HQ-CVPR22.pth"):
        self.repo = Path(repo or THIRD_PARTY / "E2FGVI")
        self.ckpt = ckpt
        if not self.repo.exists():
            raise FileNotFoundError(
                f"E2FGVI repo not found at {self.repo}. "
                "Run: git clone https://github.com/MCG-NKU/E2FGVI third_party/E2FGVI")

    def inpaint(self, frames_dir: Path, masks_dir: Path, output_dir: Path,
                max_clip_len: int = 60, neighbor_stride: int = 5) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(self.repo / "test.py"),
            "--model", "e2fgvi_hq",
            "--video", str(frames_dir),
            "--mask", str(masks_dir),
            "--ckpt", str(self.repo / self.ckpt),
            "--save_path", str(output_dir),
            "--num_local_frames", str(max_clip_len),
            "--neighbor_stride", str(neighbor_stride),
        ]
        subprocess.run(cmd, check=True, cwd=self.repo)
        return output_dir


class ProPainterEngine:
    """Wraps ProPainter inference (higher quality, paid tier)."""

    name = "propainter"

    def __init__(self, repo: Path | None = None):
        self.repo = Path(repo or THIRD_PARTY / "ProPainter")
        if not self.repo.exists():
            raise FileNotFoundError(
                f"ProPainter repo not found at {self.repo}. "
                "Run: git clone https://github.com/sczhou/ProPainter third_party/ProPainter")

    def inpaint(self, frames_dir: Path, masks_dir: Path, output_dir: Path,
                max_clip_len: int = 60, neighbor_stride: int = 5) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(self.repo / "inference_propainter.py"),
            "--video", str(frames_dir),
            "--mask", str(masks_dir),
            "--output", str(output_dir),
            "--neighbor_length", str(max_clip_len),
            "--ref_stride", str(neighbor_stride),
            "--fp16",
        ]
        subprocess.run(cmd, check=True, cwd=self.repo)
        return output_dir


ENGINES = {"e2fgvi": E2FGVIEngine, "propainter": ProPainterEngine}


def get_engine(name: str, **kwargs):
    return ENGINES[name](**kwargs)
