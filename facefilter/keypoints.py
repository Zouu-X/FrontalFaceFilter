from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

from .types import FaceLandmarks


# MediaPipe FaceMesh landmark indices for key points
# Order: nose, chin, left_eye, right_eye, left_mouth, right_mouth
PNP_LANDMARKS: Dict[str, int] = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

PNP_ORDER: List[str] = ["nose", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth"]


def select_pnp_image_points(landmarks: FaceLandmarks) -> np.ndarray:
    """Return 2D image points (shape [6,2]) in PnP order.

    Expects `landmarks.pixel` to be shape [N,2]. Indexes correspond to MediaPipe FaceMesh.
    """
    pix = landmarks.pixel
    idxs = [PNP_LANDMARKS[name] for name in PNP_ORDER]
    if np.max(idxs) >= pix.shape[0]:
        raise ValueError("Landmark array too small for required PnP indices")
    pts2d = pix[idxs, :].astype(np.float32)
    return pts2d


def get_pnp_object_points(cfg: Optional[dict] = None) -> np.ndarray:
    """Return 3D template points (shape [6,3]) matching PnP order.

    If cfg contains `pnp_template.points`, use it; otherwise, use defaults
    from config.DEFAULTS mirrored here to avoid circular import.
    """
    default_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1],
        ],
        dtype=np.float32,
    )
    if cfg and isinstance(cfg.get("pnp_template"), dict):
        pts = cfg["pnp_template"].get("points")
        if isinstance(pts, (list, tuple)):
            arr = np.asarray(pts, dtype=np.float32)
            if arr.shape == (6, 3):
                return arr
    return default_points


__all__ = [
    "PNP_LANDMARKS",
    "PNP_ORDER",
    "select_pnp_image_points",
    "get_pnp_object_points",
]

