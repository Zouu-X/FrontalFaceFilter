"""Image loader implementation (Task 2).

- Recursive image enumeration with extension whitelist and optional `max_files`.
- Unicode-safe image reading via OpenCV (imdecode) with fallback.
- Emits image size metadata required by downstream components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import numpy as np
import cv2

from .types import ImageMeta

logger = logging.getLogger(__name__)


class ImageLoader:
    def __init__(
        self,
        input_dir: str | Path,
        exts: Optional[Iterable[str]] = None,
        max_files: Optional[int] = None,
    ):
        self.input_dir = Path(input_dir)
        self.exts = set(e.lower() for e in (exts or {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}))
        self.max_files = max_files

    def enumerate(self) -> Generator[Path, None, None]:
        count = 0
        if not self.input_dir.exists():
            logger.warning("Input directory does not exist: %s", self.input_dir)
            return
        for p in self.input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.exts:
                yield p
                count += 1
                if self.max_files is not None and count >= self.max_files:
                    return

    @staticmethod
    def _channels_of(img: np.ndarray) -> int:
        if img.ndim == 2:
            return 1
        if img.ndim == 3:
            return img.shape[2]
        return 0

    @staticmethod
    def _imread_unicode(path: Path) -> Optional[np.ndarray]:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Fallback to standard imread
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return img
        except Exception:
            return None

    def read_image(self, path: str | Path) -> Tuple[Optional[np.ndarray], Optional[ImageMeta], Optional[str]]:
        p = Path(path)
        img = self._imread_unicode(p)
        if img is None:
            return None, None, "unreadable"
        h, w = img.shape[:2]
        ch = self._channels_of(img)
        meta = ImageMeta(path=str(p), width=w, height=h, channels=ch, ext=p.suffix.lower())
        return img, meta, None

    def iter_images(self) -> Generator[Tuple[Path, Optional[np.ndarray], Optional[ImageMeta]], None, None]:
        for p in self.enumerate():
            img, meta, err = self.read_image(p)
            if err is not None or img is None or meta is None:
                logger.warning("Failed to read image: %s (%s)", p, err)
                continue
            yield p, img, meta

__all__ = ["ImageLoader"]

