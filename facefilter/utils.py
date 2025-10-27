from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging with a simple, consistent format.

    Accepts either a logging level name (str) or numeric level.
    """
    if isinstance(level, str):
        lvl = logging.getLevelName(level.upper())
        if not isinstance(lvl, int):
            lvl = logging.INFO
    else:
        lvl = int(level)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: str | os.PathLike, exist_ok: bool = True) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=exist_ok)
    return p


def is_image_file(path: str | os.PathLike, exts: Iterable[str] | None = None) -> bool:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    suffix = Path(path).suffix.lower()
    return suffix in set(e.lower() for e in exts)


def seed_everything(seed: int) -> None:
    """Deterministic seeding across common libs and hash seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

