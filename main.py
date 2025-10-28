import argparse
import os
from pathlib import Path

import cv2

from facefilter.config import load_and_merge
from facefilter.utils import setup_logging
from facefilter.loader import ImageLoader
from facefilter.facemesh import FaceMeshDetector, FaceMeshConfig
from facefilter.keypoints import select_pnp_image_points, get_pnp_object_points
from facefilter.camera import build_camera_matrix, get_dist_coeffs
from facefilter.pose import estimate_pose
from facefilter.filters import is_frontal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frontal Face Filter demo: single image pose estimation")
    p.add_argument("--image", required=True, help="Path to an image (PNG/JPG)")
    p.add_argument("--config", default=None, help="Optional YAML config path")
    p.add_argument("--save-debug", default=None, help="Optional path to save landmark debug overlay")
    p.add_argument("--log-level", default=None, help="Override log level (e.g., INFO, WARNING)")
    return p.parse_args()


def draw_debug(image_bgr, landmarks_pixel, bbox, out_path: str):
    vis = image_bgr.copy()
    # Draw a subset of landmarks for visibility
    for x, y in landmarks_pixel[::10]:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(out_path, vis)


def main():
    # Reduce TF/MediaPipe verbosity if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()
    cfg = load_and_merge(args.config)

    setup_logging(args.log_level or cfg.get("runtime", {}).get("log_level", "INFO"))

    # Read image and metadata using loader utilities
    loader = ImageLoader(input_dir=Path(args.image).parent)
    image, meta, err = loader.read_image(args.image)
    if err or image is None or meta is None:
        raise SystemExit(f"Failed to read image: {args.image} ({err})")

    # FaceMesh detection
    mp_cfg = cfg.get("mediapipe", {})
    fm_cfg = FaceMeshConfig(
        static_image_mode=bool(mp_cfg.get("static_image_mode", True)),
        refine_landmarks=bool(mp_cfg.get("refine_landmarks", False)),
        max_faces=int(mp_cfg.get("max_faces", 2)),
    )

    with FaceMeshDetector(fm_cfg) as det:
        fl, det_info = det.detect(image, meta)
    if fl is None or det_info is None:
        print("No face detected")
        return

    # Keypoints and camera
    pts2d = select_pnp_image_points(fl)
    pts3d = get_pnp_object_points(cfg)
    cam_cfg = cfg.get("camera", {})
    focal = cam_cfg.get("focal")
    principal = cam_cfg.get("principal")
    K = build_camera_matrix(meta, focal=focal, principal=tuple(principal) if principal else None)
    dist = get_dist_coeffs(cfg)

    # Pose
    pose = estimate_pose(pts3d, pts2d, K, dist)
    if not pose.success:
        print("Pose estimation failed:", pose.reason)
        return

    print("Landmarks:", fl.pixel.shape, det_info.bbox)
    print("Angles (deg): yaw=%.2f pitch=%.2f roll=%.2f" % (pose.yaw, pose.pitch, pose.roll))
    print("Reproj error:", pose.reproj_error)

    decision = is_frontal(pose, cfg)
    print(
        "Frontal:", decision.is_frontal,
        "Maybe:", decision.maybe_frontal,
        "Reasons:", ",".join(decision.reasons) if decision.reasons else "-",
    )

    if args.save_debug:
        out_path = str(args.save_debug)
        draw_debug(image, fl.pixel, det_info.bbox, out_path)
        print("Saved debug overlay:", out_path)


if __name__ == "__main__":
    main()


