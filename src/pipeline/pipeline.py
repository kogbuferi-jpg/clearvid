"""End-to-end pipeline: detect subtitles -> mask -> inpaint -> mux."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from src.detect.subtitle import SubtitleDetector
from src.detect.mask import boxes_to_mask, merge_masks_temporal
from src.inpaint.engine import get_engine
from src.pipeline.video_io import decode_frames, encode_video


def remove_subtitles(input_video: str | Path, output: str | Path,
                     config_path: str | Path = "configs/default.yaml",
                     engine_override: str | None = None,
                     keep_temp: bool = False) -> Path:
    input_video, output = Path(input_video), Path(output)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    workdir = Path(tempfile.mkdtemp(prefix="clearvid_"))
    frames_dir, masks_dir, results_dir = (workdir / d for d in
                                          ("frames", "masks", "results"))
    try:
        print(f"[1/4] Decoding frames: {input_video.name}")
        fps = decode_frames(input_video, frames_dir)

        print("[2/4] Detecting subtitle regions")
        det_cfg = cfg["detect"]
        detector = SubtitleDetector()
        frames = sorted(frames_dir.glob("*.png"))
        raw_masks = []
        for f in tqdm(frames):
            img = cv2.imread(str(f))
            boxes = detector.detect_boxes(img)
            raw_masks.append(boxes_to_mask(img.shape[:2], boxes,
                                           dilate_px=det_cfg["dilate_px"]))
        masks = merge_masks_temporal(raw_masks, gap=det_cfg["temporal_merge_gap"])
        masks_dir.mkdir(parents=True, exist_ok=True)
        for f, m in zip(frames, masks):
            cv2.imwrite(str(masks_dir / f.name), m)

        engine_name = engine_override or cfg["inpaint"]["engine"]
        print(f"[3/4] Inpainting with {engine_name}")
        engine = get_engine(engine_name)
        engine.inpaint(frames_dir, masks_dir, results_dir,
                       max_clip_len=cfg["inpaint"]["max_clip_len"],
                       neighbor_stride=cfg["inpaint"]["neighbor_stride"])

        print("[4/4] Encoding output")
        encode_video(results_dir, input_video, output, fps,
                     codec=cfg["io"]["codec"])
        print(f"Done -> {output}")
        return output
    finally:
        if not keep_temp:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
