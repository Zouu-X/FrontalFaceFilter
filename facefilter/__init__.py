"""Face Filter package scaffolding.

Contains the core modules for the frontal face filtering pipeline
as outlined in doc/design.md. This initial version provides shared
types, configuration loading/merging, deterministic seeding, and
basic utilities. Functional modules are stubbed for later tasks.
"""

from . import config as config
from . import types as types
from . import utils as utils
from . import keypoints as keypoints
from . import camera as camera

__all__ = [
    "config",
    "types",
    "utils",
    "keypoints",
    "camera",
]
