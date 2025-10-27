# Task Breakdown — Frontal Face Filtering Pipeline

## 1. Project Scaffolding & Utilities
- [✅] Create module layout (`loader.py`, `pose.py`, `filters.py`, `writers.py`, etc.) and shared `config.py` loader.
- [✅] Define dataclasses/typed dicts for intermediate artifacts (image metadata, landmarks, pose result).
- [✅] Implement YAML + CLI config merging helper (defaults → YAML → CLI overrides) with deterministic seeding.

## 2. Image Loader
- [✅] Implement recursive image enumeration with extension whitelist and optional `max_files` cap.
- [✅] Add OpenCV-based image reader with logging for unreadable files and Unicode-safe paths.
- [✅] Emit image size metadata needed by downstream components.

## 3. Face Mesh Integration
- [ ] Wrap MediaPipe FaceMesh with configurable options (static image mode, refinement toggle).
- [ ] Select primary face via largest bounding box among detections; expose fallback when no face found.
- [ ] Convert 468 normalized landmarks into pixel coordinates bundled with detection metadata.

## 4. Keypoint Selection & Camera Model
- [ ] Map required landmark indices (nose, chin, eyes, mouth) into ordered 2D arrays.
- [ ] Store 3D template coordinates as constants with optional YAML override.
- [ ] Build camera intrinsic matrix from image dims (focal=max(h,w), principal center) with optional distortion params.

## 5. Pose Estimation & Euler Conversion
- [ ] Integrate `cv2.solvePnP` (EPnP) with error handling for failure cases.
- [ ] Convert rotation vectors to matrices (`cv2.Rodrigues`) and compute yaw/pitch/roll using documented convention.
- [ ] Unit test pose pipeline against synthetic data to verify angle correctness and sign conventions.

## 6. Quality Filters
- [ ] Implement face size filter using bounding box; threshold configurable (default 128px).
- [ ] Add blur detection via Laplacian variance with configurable minimum (default 100).
- [ ] Validate landmark visibility ratio within bounding box and reject if below threshold.

## 7. Frontal Filter & Soft-Band Handling
- [ ] Apply absolute angle thresholds (yaw 15°, pitch 10°, roll 10°) to mark frontal candidates.
- [ ] Support `maybe_frontal` classification when exactly one angle slightly exceeds threshold and others near zero.
- [ ] Track rejection reasons (`pose_yaw`, `pose_pitch`, `pose_roll`, etc.) for reporting.

## 8. Output Writers
- [ ] Implement writers for `frontal_index.json/csv`, `rejected_index.json`, and aggregated `summary.yaml`.
- [ ] Include per-image metrics: file path, bbox, angles, blur score, flags, rejection reason.
- [ ] Optionally copy accepted files to output directory when flag enabled; ensure idempotency.

## 9. CLI & Execution Flow
- [ ] Build CLI entrypoint (`detect_pose_mediapipe.py` or `main.py`) orchestrating pipeline components.
- [ ] Support batch processing with ProcessPool executor and queue-based writer thread.
- [ ] Add progress logging (INFO) and structured warnings/errors per design.

## 10. Performance & Resumability
- [ ] Benchmark throughput; tune worker chunk sizes to meet ≥8 img/s CPU baseline.
- [ ] Ensure runs are resumable by skipping already processed items based on output index or checksum.
- [ ] Evaluate TurboJPEG / faster decoding path toggle.

## 11. Validation & QA
- [ ] Create audit script to sample accepted/rejected images and export thumbnails (`bad_cases/`).
- [ ] Implement evaluation notebook/script computing precision/recall from human labels.
- [ ] Document configuration, logging locations, and reproducibility steps in README.
