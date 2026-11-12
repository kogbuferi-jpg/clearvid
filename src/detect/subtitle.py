"""Subtitle region detection via PaddleOCR."""
from __future__ import annotations

import cv2
import numpy as np


class SubtitleDetector:
    """Detect hardcoded subtitle regions frame by frame.

    Uses PaddleOCR text detection boxes, restricted to a search zone
    (default: bottom 35% of the frame where subtitles usually live).
    """

    def __init__(self, lang: str = "ch", bottom_ratio: float = 1.0,
                 use_gpu: bool = True):
        from paddleocr import PaddleOCR  # lazy import, heavy dep
        self.ocr = PaddleOCR(use_angle_cls=False, lang=lang,
                             use_gpu=use_gpu, show_log=False,
                             det=True, rec=False)
        self.bottom_ratio = bottom_ratio

    def detect_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) boxes."""
        h, w = frame.shape[:2]
        y_start = int(h * (1.0 - self.bottom_ratio))
        roi = frame[y_start:, :]
        result = self.ocr.ocr(roi, cls=False)
        boxes = []
        for line in (result[0] or []):
            pts = np.array(line[0], dtype=np.int32)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min() + y_start
            x2, y2 = pts[:, 0].max(), pts[:, 1].max() + y_start
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
