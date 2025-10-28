from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class ImageMeta:
    path: str
    width: int
    height: int
    channels: Optional[int] = None
    ext: Optional[str] = None


@dataclass
class DetectionInfo:
    bbox: Optional[BBox]
    score: Optional[float]
    face_index: int = 0


@dataclass
class FaceLandmarks:
    # Normalized (x, y, z) in [0,1] for x,y; z is relative depth from MediaPipe
    normalized: np.ndarray  # shape (N, 3)
    # Pixel coordinates (x, y), shape (N, 2)
    pixel: np.ndarray
    # Optional subset of landmark indices used for pose estimation
    indices: Optional[List[int]] = None


@dataclass
class PoseResult:
    success: bool
    rvec: Optional[np.ndarray] = None  # shape (3, 1)
    tvec: Optional[np.ndarray] = None  # shape (3, 1)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    reproj_error: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class FrontalDecision:
    is_frontal: bool
    maybe_frontal: bool
    reasons: List[str]
