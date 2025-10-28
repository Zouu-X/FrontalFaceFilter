import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm

from facefilter.config import load_and_merge
from facefilter.utils import setup_logging
from facefilter.loader import ImageLoader
from facefilter.facemesh import FaceMeshDetector, FaceMeshConfig
from facefilter.keypoints import select_pnp_image_points, get_pnp_object_points
from facefilter.camera import build_camera_matrix, get_dist_coeffs
from facefilter.pose import estimate_pose
from facefilter.filters import is_frontal
from facefilter.writers import ResultsWriter, build_record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frontal Face Filter pipeline")
    # Single-image mode
    p.add_argument("--image", help="Path to a single image (PNG/JPG)")
    p.add_argument("--save-debug", default=None, help="Optional path to save landmark debug overlay (single-image mode)")
    # Batch mode
    p.add_argument("--input-dir", help="Directory of images to process (batch mode)")
    p.add_argument("--output-dir", help="Directory to write outputs (JSON + summary)")
    p.add_argument("--max-files", type=int, default=None, help="Optional max files to process (for testing)")
    p.add_argument("--workers", type=int, default=None, help="Number of worker processes (0=single-thread)")
    p.add_argument("--copy-accepted", action="store_true", help="Copy accepted images to output/accepted")
    # Config
    p.add_argument("--config", default=None, help="Optional YAML config path")
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


def process_one_path(path_str: str, cfg: dict) -> dict:
    # Local imports to ensure picklability in multiprocessing environments
    from facefilter.loader import ImageLoader
    from facefilter.facemesh import FaceMeshDetector, FaceMeshConfig
    from facefilter.keypoints import select_pnp_image_points, get_pnp_object_points
    from facefilter.camera import build_camera_matrix, get_dist_coeffs
    from facefilter.pose import estimate_pose
    from facefilter.filters import is_frontal
    from facefilter.types import ImageMeta as IMeta, PoseResult, FrontalDecision

    # Build FaceMesh config
    mp_cfg = cfg.get("mediapipe", {})
    fm_cfg = FaceMeshConfig(
        static_image_mode=bool(mp_cfg.get("static_image_mode", True)),
        refine_landmarks=bool(mp_cfg.get("refine_landmarks", False)),
        max_faces=int(mp_cfg.get("max_faces", 2)),
    )

    # Read image and meta
    loader = ImageLoader(input_dir=Path(path_str).parent)
    img, meta, err = loader.read_image(path_str)
    if err or img is None or meta is None:
        meta_fallback = meta if meta is not None else IMeta(path=str(path_str), width=0, height=0)
        pose = PoseResult(success=False, reason=err or "unreadable")
        decision = FrontalDecision(False, False, [err or "unreadable"])
        return build_record(meta_fallback, None, pose, decision)

    # Detect
    with FaceMeshDetector(fm_cfg) as det:
        fl, det_info = det.detect(img, meta)
    if fl is None or det_info is None:
        pose = PoseResult(success=False, reason="no_face")
        decision = FrontalDecision(False, False, ["no_face"])
        return build_record(meta, None, pose, decision)

    # Pose pipeline
    pts2d = select_pnp_image_points(fl)
    pts3d = get_pnp_object_points(cfg)
    cam_cfg = cfg.get("camera", {})
    focal = cam_cfg.get("focal")
    principal = cam_cfg.get("principal")
    K = build_camera_matrix(meta, focal=focal, principal=tuple(principal) if principal else None)
    dist = get_dist_coeffs(cfg)
    pose = estimate_pose(pts3d, pts2d, K, dist)

    decision = is_frontal(pose, cfg)
    rec = build_record(meta, det_info, pose, decision)
    return rec


def main():
    # Reduce TF/MediaPipe verbosity if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()

    # Build CLI overrides for config merging
    cli_overrides = {"paths": {}, "runtime": {}, "mediapipe": {}}
    if args.input_dir:
        cli_overrides["paths"]["input_dir"] = args.input_dir
    if args.output_dir:
        cli_overrides["paths"]["output_dir"] = args.output_dir
    if args.copy_accepted:
        cli_overrides["paths"]["copy_accepted"] = True
    if args.max_files is not None:
        cli_overrides["runtime"]["max_files"] = args.max_files
    if args.workers is not None:
        cli_overrides["runtime"]["workers"] = args.workers
    if args.log_level:
        cli_overrides["runtime"]["log_level"] = args.log_level

    cfg = load_and_merge(args.config, cli_overrides)

    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    # Single-image mode
    if args.image and not args.input_dir:
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
        return

    # Batch mode
    input_dir = cfg.get("paths", {}).get("input_dir")
    output_dir = cfg.get("paths", {}).get("output_dir")
    if not input_dir or not output_dir:
        raise SystemExit("Batch mode requires --input-dir and --output-dir (or set in config)")

    loader = ImageLoader(input_dir=input_dir, max_files=cfg.get("runtime", {}).get("max_files"))
    paths = list(loader.enumerate())
    if not paths:
        print("No images found in", input_dir)
        return

    writer = ResultsWriter(output_dir, cfg)

    workers = int(cfg.get("runtime", {}).get("workers", 0) or 0)
    if workers <= 0:
        for p in tqdm(paths, desc="Processing", unit="img"):
            rec = process_one_path(str(p), cfg)
            writer.add(rec)
            if rec.get("accepted"):
                writer.maybe_copy(rec["file"])  # type: ignore[index]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one_path, str(p), cfg): p for p in paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="img"):
                try:
                    rec = fut.result()
                    writer.add(rec)
                    if rec.get("accepted"):
                        writer.maybe_copy(rec["file"])  # type: ignore[index]
                except Exception as e:
                    print("Worker failed:", e)

    summary = writer.finalize()
    print("Summary:", summary)


if __name__ == "__main__":
    main()
