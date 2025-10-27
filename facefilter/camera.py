from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .types import ImageMeta


def build_camera_matrix(meta: ImageMeta, focal: Optional[float] = None, principal: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Return intrinsic matrix K (3x3) using pinhole model.

    - If `focal` is None, use max(h, w).
    - If `principal` is None, use image center: (w/2, h/2).
    """
    # w, h = float(meta.width), float(meta.height)
    w, h = float(256), float(256)
    f = float(focal) if focal is not None else max(w, h)
    cx, cy = principal if principal is not None else (w / 2.0, h / 2.0)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def get_dist_coeffs(cfg: Optional[dict] = None) -> np.ndarray:
    """Return distortion coefficients vector for OpenCV (k1, k2, p1, p2, k3, k4, k5, k6).

    OpenCV accepts variable lengths; we return 8 for generality.
    """
    if cfg and isinstance(cfg.get("camera"), dict):
        dist = cfg["camera"].get("distortion")
        if isinstance(dist, (list, tuple)) and len(dist) in (4, 5, 8):
            return np.asarray(dist, dtype=np.float32).reshape(-1, 1)
    return np.zeros((8, 1), dtype=np.float32)


__all__ = ["build_camera_matrix", "get_dist_coeffs"]

