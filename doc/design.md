# design.md — Frontal-Face Filtering Pipeline (方案 A: MediaPipe + solvePnP)
          ┌────────────────────┐
          │   原始人脸图片集    │  （7万张）
          └────────┬───────────┘
                   │
          ┌────────▼────────┐
          │ Face Mesh检测    │ ← MediaPipe: 提取468个关键点
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ 姿态估计solvePnP │ ← 用3D模板 + 2D关键点解PnP求旋转向量
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ 欧拉角转换       │ ← 得到 yaw / pitch / roll
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ 角度阈值过滤     │ ← |yaw|≤15°, |pitch|≤10°, |roll|≤10°
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ 输出正脸列表     │ ← 保存 JSON / CSV / 移动文件
          └──────────────────┘


## 1) Objective & Success Criteria
**Goal:** From ~70k mixed frontal/side face images (anime/3D-styled), automatically keep frontal faces and filter out side faces.
- **Primary metric (manual audit on 400 sampled images):**
  - Precision (正脸准确率) ≥ 0.95
  - Recall (正脸召回) ≥ 0.90
- **Throughput:** ≥ 8 images/second on CPU-only (MacBook-class) OR ≥ 40 images/second on a 16-core server.
- **Reproducibility:** Deterministic outputs given the same inputs and config YAML.

## 2) Scope
- **In scope:** Single-image processing; face detection/landmarks via MediaPipe FaceMesh; pose estimation via solvePnP; frontal filtering by angle thresholds; basic quality filtering (blur/size); JSON/CSV export; optional file copy of accepted images.
- **Out of scope:** Face re-identification、表情/年龄/性别检测、多脸优先级策略训练、模型微调、UI 前端。

## 3) Assumptions & Constraints
- Images contain one prominent face; backgrounds simple; anime/virtual style.
- Resolution ≥ 256 px on the smallest dimension is typical.
- Run on a cluster with 16GB memory, 4 cores, A10 GPU
- Files may be nested in subdirectories; Unicode filenames allowed.
- Batch runs must be resumable (idempotent based on output index).

## 4) Definitions
- **yaw**: 左右转头角度（右为正/左为负）
- **pitch**: 上下点头角度（上为负/下为正，或按实现统一约定）
- **roll**: 头部旋转角度（顺时针为正）
- **Frontal**: |yaw| ≤ YAW_TH, |pitch| ≤ PITCH_TH, |roll| ≤ ROLL_TH

## 5) System Architecture (Batch CLI Tool)
```
images/ (70k)
   └── *.png|*.jpg|...  ──┐
                          │
[Loader]  →  [Face Mesh @ MediaPipe]  →  [Keypoint Selector]  →  [solvePnP Pose]
                                ↓                                    ↓
                           [Quality Filters]                    [Euler Angles]
                                   \                              /
                                    \→ [Frontal Filter] → [Writer: JSON/CSV + Copy]
```

## 6) Components & Responsibilities
### 6.1 Loader
- Recursively enumerate images by extensions.
- Robust read via OpenCV; log unreadable files.
- Optional max_files for dry-run/testing.

### 6.2 Face Mesh (MediaPipe)
- `static_image_mode=True`, refine landmarks disabled by default (toggleable).
- Return 468 landmarks (x, y, z in normalized coordinates).
- For multi-face images: choose the face with the **largest bounding box** (configurable).

### 6.3 Keypoint Selector
- Map 2D image coordinates for six stable points (indices): Nose(1), Chin(152), Left Eye (33), Right Eye (263), Left Mouth (61), Right Mouth (291).
- Convert normalized coords to pixel coords.

### 6.4 Camera Model
- Pinhole model with focal length `f ~ max(h, w)`; principal point at image center.
- Distortion coefficients assumed zero (configurable if known).

### 6.5 3D Template (Head Model)
- Fixed 3D reference points (mm-like units):
  ```
  [ (0.0,   0.0,   0.0  ),   # nose tip
    (0.0,  -63.6, -12.5 ),   # chin
    (-43.3, 32.7, -26.0),    # left eye corner
    (43.3,  32.7, -26.0),    # right eye corner
    (-28.9,-28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1) ]   # right mouth corner
  ```
- Store in code as constants; allow YAML override.

### 6.6 Pose Estimation (solvePnP)
- Use `cv2.solvePnP` (EPnP by default); check return flag.
- Convert rotation vector to rotation matrix via `cv2.Rodrigues`.
- Convert to Euler angles (yaw/pitch/roll) using a single, tested convention (documented).

### 6.7 Quality Filters
- **Min face size**: min(face_w, face_h) ≥ `min_face_size` (default 128 px).
- **Blur filter**: Laplacian variance ≥ `blur_threshold` (default 100).
- **Landmark visibility**: MediaPipe returns landmarks even with occlusion; require ≥ `min_visible_ratio` (default 0.9) based on bbox containment.

### 6.8 Frontal Filter
- Thresholds (default):
  - `abs(yaw)   ≤ 15°`
  - `abs(pitch) ≤ 10°`
  - `abs(roll)  ≤ 10°`
- Support **soft-band** admission: if only 1 angle slightly exceeds threshold but the others are near 0 and quality high, mark as `maybe_frontal=true` (for human review).

### 6.9 Writer / Outputs
- `frontal_index.json` (list of dicts) and `frontal_index.csv` with columns:
  - `file, bbox_x1, bbox_y1, bbox_x2, bbox_y2, yaw, pitch, roll, blur, is_frontal, reason`
- `rejected_index.json` with reason tags: `no_face`, `small_face`, `blurry`, `pose_yaw`, `pose_pitch`, `pose_roll`.
- Write a `summary.yaml` with counts, thresholds, runtime, version, hostname, and timing.

## 7) CLI & Config
### 7.1 Command Line
```
python detect_pose_mediapipe.py   --input /path/to/images   --out-dir /path/to/out   --yaw-th 15 --pitch-th 10 --roll-th 10   --min-face 128 --blur-th 100   --num-workers 8   --copy-accepted
```
### 7.2 YAML Config (optional)
- `config.yaml` overrides defaults; CLI flags override YAML.

## 8) Performance & Parallelism
- Use `concurrent.futures.ProcessPoolExecutor` with `num_workers = min(physical_cores, 16)`.
- I/O bound mitigations: prefetch list, avoid re-reading; optionally use TurboJPEG for decoding.
- Chunked writing to reduce lock contention (writer thread + queue).

## 9) Logging & Error Handling
- `logging.INFO` progress every N images.
- `logging.WARNING` for recoverable issues (bad image, no face).
- `logging.ERROR` when solvePnP fails or landmarks missing; continue.
- Emit `bad_cases/` with small thumbnails for quick inspection (optional).

## 10) Validation Plan
- Randomly sample 200 accepted + 200 rejected; human label “frontal/非正脸/不确定”。
- Compute precision/recall; adjust thresholds based on ROC-like sweep.
- Record angle histograms; verify distributions are sensible (no strong bias).

## 11) Directory Layout
```
project/
  detect_pose_mediapipe.py
  configs/
    default.yaml
  out/
    frontal_index.json
    frontal_index.csv
    rejected_index.json
    summary.yaml
  logs/
    run-2025-10-24.txt
```

## 12) Dependencies
- Python ≥ 3.9
- mediapipe ≥ 0.10
- opencv-python ≥ 4.8
- numpy, tqdm, pyyaml
- (optional) pillow, turbojpeg, rich

`requirements.txt`:
```
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.22
tqdm>=4.66
PyYAML>=6.0
Pillow>=10.0
```

## 13) Non-Functional Requirements
- **Determinism:** Same inputs + config ⇒ Same outputs.
- **Portability:** Works on macOS/Linux/Windows.
- **Robustness:** No crash on unreadable images; continue and log.
- **Traceability:** Each decision has a reason tag and recorded thresholds.

## 14) Risks & Mitigations
- **Anime风格偏差导致角度偏移** → 定制3D模板/焦距；以人工标注校准阈值。
- **多脸图像选择错误** → 策略从“最大脸”改为“中心最近”或“score最高”。
- **性能瓶颈** → 多进程；跳过写大图（只写索引）；仅在 `--copy-accepted` 时复制。

## 15) Roadmap & Deliverables
- **v0.1**: 单线程原型，输出角度与JSON。
- **v0.2**: 并行化 + 质量过滤 + CSV 输出。
- **v1.0**: 可配置阈值、复制合格图片、summary、日志与验证脚本。

## 16) Test Cases (Samples)
1. **空目录** → 处理结束，summary 显示 0；无异常。
2. **损坏图片** → 跳过并 WARNING；不阻塞。
3. **侧脸 30°** → `is_frontal=false`, reason 包含 `pose_yaw`。
4. **正脸轻微滚转 8°** → 通过。
5. **小人脸 (min_side < 128)** → 拒绝，`small_face`。

## 17) Extensions (Future)
- 使用 FaceMesh 的可见性/置信度筛选遮挡。
- 加入对称性特征作为次级判据（左右眼距/鼻唇比）。
- 以轻量 CNN 回归角度做纠偏（可训练）。
- GUI 标注工具，用于快速标注/校准阈值。
