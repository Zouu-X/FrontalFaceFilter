"""Quality and frontal angle filters (Task 7).

Implements frontal classification using absolute yaw/pitch/roll thresholds
and a soft-band to mark maybe_frontal cases.
"""

from __future__ import annotations

from typing import Dict, List

from .types import PoseResult, FrontalDecision


def apply_quality_filters(*args, **kwargs):
    """Placeholder since Task 6 is skipped; always passes.

    Returns True (accepted) and empty reasons list.
    """
    return True, []


def classify_frontal(pose: PoseResult, cfg: Dict) -> FrontalDecision:
    th = cfg.get("thresholds", {})
    yaw_th = float(th.get("yaw", 15.0))
    pitch_th = float(th.get("pitch", 10.0))
    roll_th = float(th.get("roll", 10.0))
    soft_band = float(th.get("soft_band", 5.0))
    near_zero = float(th.get("near_zero", 3.0))

    if not pose.success or pose.yaw is None or pose.pitch is None or pose.roll is None:
        return FrontalDecision(False, False, [pose.reason or "pose_invalid"])

    ay, ap, ar = abs(pose.yaw), abs(pose.pitch), abs(pose.roll)

    reasons: List[str] = []
    if ay > yaw_th:
        reasons.append("pose_yaw")
    if ap > pitch_th:
        reasons.append("pose_pitch")
    if ar > roll_th:
        reasons.append("pose_roll")

    is_front = len(reasons) == 0

    maybe = False
    if not is_front:
        exceeded = [
            ("yaw", ay - yaw_th),
            ("pitch", ap - pitch_th),
            ("roll", ar - roll_th),
        ]
        # angles within threshold count as negative exceed => clamp at 0
        exceeded = [(n, max(0.0, v)) for (n, v) in exceeded]
        n_exceed = sum(1 for _, v in exceeded if v > 0)
        if n_exceed == 1:
            # Only one angle slightly over threshold and others near zero
            name, over = max(exceeded, key=lambda x: x[1])
            others_ok = (ay if name != "yaw" else 0.0) <= near_zero and (ap if name != "pitch" else 0.0) <= near_zero and (ar if name != "roll" else 0.0) <= near_zero
            if over <= soft_band and others_ok:
                maybe = True

    return FrontalDecision(is_front, maybe, reasons)


# Backward-compatible alias
is_frontal = classify_frontal

__all__ = ["apply_quality_filters", "classify_frontal", "is_frontal"]
