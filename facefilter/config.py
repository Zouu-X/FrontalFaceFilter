from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml

from .utils import seed_everything

logger = logging.getLogger(__name__)


DEFAULTS: Dict[str, Any] = {
    "paths": {
        "input_dir": "images",
        "output_dir": "outputs",
        "copy_accepted": False,
    },
    "camera": {
        # Focal length in pixels; default uses max(h, w) at runtime if None
        "focal": None,
        # Principal point; default uses image center (w/2, h/2)
        "principal": None,
        # Radial/tangential distortion coefficients (OpenCV order)
        # k1, k2, p1, p2, k3, k4, k5, k6
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "pnp_template": {
        # 3D template points in mm-like units: order must match keypoint selection
        # [nose, chin, left_eye, right_eye, left_mouth, right_mouth]
        "points": [
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1],
        ],
    },
    "thresholds": {
        "yaw": 15.0,
        "pitch": 10.0,
        "roll": 10.0,
        # Soft band handling for maybe_frontal classification
        # If exactly one angle exceeds by ≤ soft_band and the others are within near_zero,
        # mark as maybe_frontal
        "soft_band": 5.0,
        "near_zero": 3.0,
        "min_face": 128,
        "min_blur": 100.0,
    },
    "mediapipe": {
        "static_image_mode": True,
        "refine_landmarks": False,
        "max_faces": 2,
    },
    "runtime": {
        "seed": 1337,
        "workers": 0,  # 0 => single-thread; >0 => process pool size
        "max_files": None,
        "log_level": "INFO",
    },
}


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], MutableMapping) and isinstance(v, Mapping):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("YAML config not found: %s", p)
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level of YAML must be a mapping/dict")
    return data


def merge_config(yaml_cfg: Mapping[str, Any] | None = None, cli_overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    _deep_merge(cfg, DEFAULTS.copy())
    if yaml_cfg:
        _deep_merge(cfg, dict(yaml_cfg))
    if cli_overrides:
        _deep_merge(cfg, dict(cli_overrides))

    # Normalize and seed deterministically
    runtime = cfg.get("runtime", {})
    seed = int(runtime.get("seed", 1337))
    seed_everything(seed)
    return cfg


def load_and_merge(yaml_path: str | Path | None, cli_overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    yaml_cfg = load_yaml(yaml_path)
    return merge_config(yaml_cfg, cli_overrides)
