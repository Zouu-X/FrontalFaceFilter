"""Image loader scaffolding.

Implements recursive enumeration and image reading in later tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, Iterable, Optional


class ImageLoader:
    def __init__(self, input_dir: str | Path, exts: Optional[Iterable[str]] = None, max_files: Optional[int] = None):
        self.input_dir = Path(input_dir)
        self.exts = set(e.lower() for e in (exts or {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}))
        self.max_files = max_files

    def enumerate(self) -> Generator[Path, None, None]:
        count = 0
        for p in self.input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.exts:
                yield p
                count += 1
                if self.max_files is not None and count >= self.max_files:
                    return

    def read_image(self, path: str | Path) -> Dict:
        """Stub for image reading; implemented in Task 2."""
        raise NotImplementedError

