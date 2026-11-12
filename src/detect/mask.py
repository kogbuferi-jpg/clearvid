"""Mask generation: dilate boxes, merge across time."""
from __future__ import annotations

import cv2
import numpy as np


def boxes_to_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]],
                  dilate_px: int = 8) -> np.ndarray:
    """Render boxes into a binary mask (H, W), 255 = to-inpaint."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1 - dilate_px), max(0, y1 - dilate_px)
        x2, y2 = min(w, x2 + dilate_px), min(h, y2 + dilate_px)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def merge_masks_temporal(masks: list[np.ndarray], gap: int = 10) -> list[np.ndarray]:
    """Stabilize masks over time: a region missing for <= `gap` frames
    (detection flicker) gets filled with the union of its neighbors."""
    n = len(masks)
    out = [m.copy() for m in masks]
    for i in range(n):
        if masks[i].any():
            continue
        prev = next((masks[j] for j in range(i - 1, max(-1, i - gap - 1), -1)
                     if masks[j].any()), None)
        nxt = next((masks[j] for j in range(i + 1, min(n, i + gap + 1))
                    if masks[j].any()), None)
        if prev is not None and nxt is not None:
            out[i] = cv2.bitwise_or(prev, nxt)
    return out
