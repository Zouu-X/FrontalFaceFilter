from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import cv2
    import mediapipe as mp
except Exception as e:  # pragma: no cover - environment import guard
    cv2 = None  # type: ignore
    mp = None  # type: ignore

from .types import ImageMeta, FaceLandmarks, DetectionInfo, BBox

logger = logging.getLogger(__name__)


@dataclass
class FaceMeshConfig:
    static_image_mode: bool = True
    refine_landmarks: bool = False
    max_faces: int = 2


def _landmarks_to_arrays(lms, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = [pt.x for pt in lms.landmark]
    ys = [pt.y for pt in lms.landmark]
    zs = [getattr(pt, "z", 0.0) for pt in lms.landmark]
    normalized = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    px = np.clip(np.round(normalized[:, 0] * width), 0, width - 1)
    py = np.clip(np.round(normalized[:, 1] * height), 0, height - 1)
    pixel = np.stack([px, py], axis=1).astype(np.int32)
    return normalized, pixel


def _bbox_from_pixels(pixel: np.ndarray) -> BBox:
    x_min = int(np.min(pixel[:, 0]))
    y_min = int(np.min(pixel[:, 1]))
    x_max = int(np.max(pixel[:, 0]))
    y_max = int(np.max(pixel[:, 1]))
    return BBox(x=x_min, y=y_min, w=int(x_max - x_min + 1), h=int(y_max - y_min + 1))


class FaceMeshDetector:
    """Reusable wrapper around MediaPipe FaceMesh for static images.

    Usage:
        with FaceMeshDetector(FaceMeshConfig()) as det:
            result = det.detect(image, meta)
    """

    def __init__(self, cfg: Optional[FaceMeshConfig] = None):
        if mp is None or cv2 is None:
            raise ImportError("mediapipe and opencv-python must be installed to use FaceMeshDetector")
        self.cfg = cfg or FaceMeshConfig()
        self._mesh = None

    def __enter__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.cfg.static_image_mode,
            refine_landmarks=self.cfg.refine_landmarks,
            max_num_faces=self.cfg.max_faces,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    def _ensure_open(self):
        if self._mesh is None:
            # Allow use without context manager by lazy init
            self.__enter__()

    def detect(self, image_bgr: np.ndarray, meta: ImageMeta) -> Tuple[Optional[FaceLandmarks], Optional[DetectionInfo]]:
        self._ensure_open()
        assert self._mesh is not None

        # Convert BGR -> RGB as required by MediaPipe
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(img_rgb)
        if not results or not results.multi_face_landmarks:
            return None, None

        faces = results.multi_face_landmarks
        width, height = meta.width, meta.height

        # Convert all faces to arrays and compute bbox area for selection
        converted: List[Tuple[np.ndarray, np.ndarray, BBox]] = []
        for flm in faces:
            norm, pix = _landmarks_to_arrays(flm, width, height)
            bbox = _bbox_from_pixels(pix)
            converted.append((norm, pix, bbox))

        # Select primary face: largest bbox area
        areas = [c[2].w * c[2].h for c in converted]
        idx = int(np.argmax(areas)) if len(areas) > 1 else 0
        norm, pix, bbox = converted[idx]

        fl = FaceLandmarks(normalized=norm, pixel=pix, indices=None)
        det = DetectionInfo(bbox=bbox, score=None, face_index=idx)
        return fl, det

__all__ = ["FaceMeshDetector", "FaceMeshConfig"]

