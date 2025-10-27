"""Convenience wrapper for facefilter.facemesh.

Re-exports the FaceMeshDetector and FaceMeshConfig from the package
implementation so user code can `import FaceMesh` directly.
"""

from facefilter.facemesh import FaceMeshDetector, FaceMeshConfig

__all__ = ["FaceMeshDetector", "FaceMeshConfig"]
