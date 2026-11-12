"""Video IO: decode to frames, encode frames back to video with audio passthrough."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


def probe_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def decode_frames(video: Path, out_dir: Path) -> float:
    """Extract frames as PNG sequence, return fps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = probe_fps(video)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video),
        "-qscale:v", "1", str(out_dir / "%06d.png"),
    ], check=True, capture_output=True)
    return fps


def encode_video(frames_dir: Path, source_video: Path, output: Path,
                 fps: float, codec: str = "libx264") -> Path:
    """Mux restored frames with the original audio track."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".silent.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", codec, "-crf", "16", "-pix_fmt", "yuv420p", str(tmp),
    ], check=True, capture_output=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp), "-i", str(source_video),
        "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a?",
        "-shortest", str(output),
    ], check=True, capture_output=True)
    tmp.unlink(missing_ok=True)
    return output


def copy_frames(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.png")):
        shutil.copy2(f, dst_dir / f.name)
