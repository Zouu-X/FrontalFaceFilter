"""Output writers (Task 8).

Implements writers for frontal/rejected indices (JSON only) and a summary
YAML. Supports optional copying of accepted files to an output folder
with idempotent behavior.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .types import DetectionInfo, FrontalDecision, ImageMeta, PoseResult
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _bbox_to_dict(bbox) -> Optional[Dict[str, int]]:
    if bbox is None:
        return None
    return {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h}


def build_record(
    meta: ImageMeta,
    det: Optional[DetectionInfo],
    pose: PoseResult,
    decision: FrontalDecision,
    blur: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "file": meta.path,
        "width": meta.width,
        "height": meta.height,
        "bbox": _bbox_to_dict(det.bbox) if det else None,
        "angles": {
            "yaw": pose.yaw,
            "pitch": pose.pitch,
            "roll": pose.roll,
        },
        "reproj_error": pose.reproj_error,
        "blur": blur,
        "accepted": decision.is_frontal,
        "maybe_frontal": decision.maybe_frontal,
        "reasons": decision.reasons,
    }
    if extra:
        rec.update(extra)
    return rec


class ResultsWriter:
    def __init__(self, output_dir: str | Path, cfg: Optional[Dict[str, Any]] = None):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        self.cfg = cfg or {}
        self.accepted: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        if record.get("accepted"):
            self.accepted.append(record)
        else:
            self.rejected.append(record)

    def maybe_copy(self, src_path: str | Path) -> Optional[str]:
        paths = (self.cfg or {}).get("paths", {})
        if not paths or not paths.get("copy_accepted", False):
            return None
        base_dir = paths.get("input_dir")
        accepted_dir = ensure_dir(self.output_dir / "accepted")
        src_path = Path(src_path)
        try:
            rel = Path(os.path.relpath(src_path, base_dir)) if base_dir else Path(src_path.name)
        except Exception:
            rel = Path(src_path.name)
        dest = accepted_dir / rel
        ensure_dir(dest.parent)
        if dest.exists():
            return str(dest)
        try:
            shutil.copy2(src_path, dest)
            return str(dest)
        except Exception as e:
            logger.warning("Failed to copy %s -> %s (%s)", src_path, dest, e)
            return None

    def _write_json(self, path: Path, data: List[Dict[str, Any]]):
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # CSV output removed per user request; JSON only.

    def finalize(self) -> Dict[str, Any]:
        out_dir = self.output_dir
        ensure_dir(out_dir)

        acc_path_json = out_dir / "frontal_index.json"
        rej_path_json = out_dir / "rejected_index.json"
        summary_path = out_dir / "summary.yaml"

        self._write_json(acc_path_json, self.accepted)
        self._write_json(rej_path_json, self.rejected)
        # CSV output removed; only JSON indices are written.

        summary = {
            "counts": {
                "accepted": len(self.accepted),
                "rejected": len(self.rejected),
                "total": len(self.accepted) + len(self.rejected),
            },
            "thresholds": (self.cfg or {}).get("thresholds", {}),
            "paths": (self.cfg or {}).get("paths", {}),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

        return summary


__all__ = [
    "build_record",
    "ResultsWriter",
]
