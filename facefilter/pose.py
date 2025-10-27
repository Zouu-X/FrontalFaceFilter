"""Pose estimation and Euler conversion (Task 5).

Provides solvePnP (EPnP) integration and conversion of rotation vectors
to yaw/pitch/roll angles using an OpenCV camera convention:
  - Axes: X right, Y down, Z forward
  - Angles: yaw (Y), pitch (X), roll (Z)
  - Rotation composition: R = Rz(roll) * Ry(yaw) * Rx(pitch)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2

from .types import PoseResult


def _euler_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """Compute yaw(Y), pitch(X), roll(Z) using ZYX (roll→yaw→pitch) convention.

    Stable extraction that keeps frontal faces near zero angles:
      R = Rz(roll) * Ry(yaw) * Rx(pitch)
    Returns angles in radians.
    """
    assert R.shape == (3, 3)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(-R[2, 0], sy)
        pitch = np.arctan2(R[2, 1], R[2, 2])
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: sy ~ 0
        yaw = np.arctan2(-R[2, 0], sy)
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def rvec_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Convert OpenCV Rodrigues rvec to yaw/pitch/roll in degrees.

    Returns (yaw_deg, pitch_deg, roll_deg).
    """
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _euler_from_R(R)
    ypr = np.degrees([yaw, pitch, roll])
    return float(ypr[0]), float(ypr[1]), float(ypr[2])


def estimate_pose(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
    method: int = cv2.SOLVEPNP_EPNP,
) -> PoseResult:
    """Estimate head pose from 2D-3D correspondences using solvePnP.

    - pts3d: shape (N,3) float32/float64
    - pts2d: shape (N,2) float32/float64
    - K: camera intrinsics 3x3
    - dist: distortion coefficients (None or (n,1))
    - method: OpenCV solvePnP flag; defaults to EPnP
    """
    pts3d = np.asarray(pts3d, dtype=np.float32)
    pts2d = np.asarray(pts2d, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    if dist is None:
        dist = np.zeros((8, 1), dtype=np.float32)
    else:
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)

    success, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, dist, flags=method)
    if not success:
        # Fallback to ITERATIVE which can be more stable in some cases
        success, rvec, tvec = cv2.solvePnP(
            pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not success:
        return PoseResult(success=False, reason="solvePnP_failed")

    yaw_deg, pitch_deg, roll_deg = rvec_to_euler(rvec)

    # Reprojection error
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - pts2d, axis=1)
    reproj_error = float(np.mean(err))

    return PoseResult(
        success=True,
        rvec=rvec,
        tvec=tvec,
        yaw=float(yaw_deg),
        pitch=float(pitch_deg),
        roll=float(roll_deg),
        reproj_error=reproj_error,
        reason=None,
    )


__all__ = ["estimate_pose", "rvec_to_euler"]
