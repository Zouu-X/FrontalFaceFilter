Frontal Face Filter (MediaPipe + solvePnP)

A lightweight, scriptable pipeline that detects face landmarks with MediaPipe Face Mesh, estimates head pose via OpenCV `solvePnP`, and filters images by frontal pose thresholds. Suitable for large image folders; outputs JSON indices for accepted and rejected files plus a summary.

**Features**
- MediaPipe Face Mesh (468 landmarks) on static images
- Pose estimation (EPnP + fallback) and Euler angle extraction
- Angle-based filtering with configurable thresholds and soft-band "maybe frontal" handling
- Batch processing with optional multiprocessing and progress bar
- JSON outputs only: `frontal_index.json`, `rejected_index.json`, `summary.yaml`

**Requirements**
- Python 3.9–3.11 recommended
- Install dependencies:
  - `pip install -r requirements.txt`
  - If running headless servers, consider `opencv-python-headless` instead of `opencv-python`

**Installation**
1. Create/activate a virtual environment (optional but recommended)
2. Install packages: `pip install -r requirements.txt`

**Quick Start**

- Single Image Demo
  - Run pose on one image and print bbox/angles:
  - `python main.py --image /path/to/image.png --save-debug debug.png`
  - Outputs debug overlay if `--save-debug` is provided.

- Batch Processing
  - Process a folder and write outputs to an output directory:
  - `python main.py --input-dir /data/images --output-dir /data/outputs --workers 4 --max-files 1000 --copy-accepted`
  - Notes:
    - `--workers 0` (default) runs single-process; increase to match CPU cores (e.g., 4).
    - `--copy-accepted` mirrors passing images under `output_dir/accepted/`.
    - CLI flags override YAML configuration (see below).

**Configuration**
The pipeline merges defaults → YAML → CLI overrides. Provide a YAML file and pass `--config config.yaml`.

Example `config.yaml`:

```
paths:
  input_dir: /data/images
  output_dir: /data/outputs
  copy_accepted: false

thresholds:
  yaw: 15.0
  pitch: 10.0
  roll: 10.0
  soft_band: 5.0
  near_zero: 3.0

camera:
  focal: null           # defaults to max(h, w)
  principal: null       # defaults to (w/2, h/2)
  distortion: [0, 0, 0, 0, 0, 0, 0, 0]

mediapipe:
  static_image_mode: true
  refine_landmarks: false
  max_faces: 2

runtime:
  seed: 1337
  workers: 0            # set >0 for multiprocessing
  max_files: null
  log_level: INFO
```

**Outputs**
- `frontal_index.json` – array of accepted records
- `rejected_index.json` – array of rejected records
- `summary.yaml` – counts and configuration snapshot

Record schema (JSON):

```
{
  "file": "/data/images/0001.png",
  "width": 256,
  "height": 256,
  "bbox": { "x": 43, "y": 42, "w": 172, "h": 180 },
  "angles": { "yaw": -2.5, "pitch": 1.2, "roll": -0.3 },
  "reproj_error": 5.0,
  "blur": null,
  "accepted": true,
  "maybe_frontal": false,
  "reasons": []
}
```

**Angle Conventions**
- Internals use OpenCV coordinates; angles returned are degrees.
- The code applies normalization to avoid ±180° flips for forward-facing heads.
- Thresholds are absolute on yaw/pitch/roll; tune them to your data.

**Tips for Your Cluster (16GB RAM, 4 cores, A10 GPU)**
- Start with `--workers 4` to utilize CPU cores; MediaPipe Face Mesh uses TFLite (CPU) by default.
- If GPU drivers are present, MediaPipe may log GL initialization—these are normal info messages.
- To reduce TensorFlow/MediaPipe logs, set `TF_CPP_MIN_LOG_LEVEL=2`.

**Troubleshooting**
- Many logs from TensorFlow/GL: normal; suppress with `TF_CPP_MIN_LOG_LEVEL=2`.
- No face detected: verify image is readable and face is prominent; try `mediapipe.refine_landmarks: true`.
- Angles look flipped on some images: rely on pass/fail thresholds; we normalize yaw/pitch to avoid 180° ambiguity.

**Project Structure**
- `main.py` – CLI for single-image and batch processing
- `facefilter/` – core modules
  - `loader.py` – image enumeration and reading
  - `facemesh.py` – MediaPipe Face Mesh wrapper
  - `keypoints.py` – landmark index selection + 3D template
  - `camera.py` – camera intrinsics and distortion helpers
  - `pose.py` – solvePnP + Euler
  - `filters.py` – frontal classification
  - `writers.py` – JSON writers and summary

**Acknowledgements**
- MediaPipe Face Mesh by Google
- OpenCV solvePnP for pose estimation

