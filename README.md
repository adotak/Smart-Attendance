# FaceSentinel AI

FaceSentinel AI is a local, privacy-friendly attendance tool that recognizes registered faces from a webcam and records one check-in per person per day. It uses OpenCV and `face_recognition`; optional DeepFace emotion analysis is available when explicitly enabled.

## What it does

- Loads supported images from `dataset/` and ignores unrelated files.
- Rejects low-confidence matches as `UNKNOWN` using a configurable distance threshold.
- Handles multiple faces in one frame and shows live FPS and face count.
- Writes UTF-8 CSV records to `Attendance.csv` in the format `NAME,TIMESTAMP`.
- Prevents duplicate attendance entries for the current date.
- Supports optional, throttled emotion analysis so it does not run on every frame.
- Exits cleanly with **Q**, **Enter**, or **Esc**.

## Installation

Python 3.9+ is recommended. `face-recognition` may require a C++ build toolchain because it depends on dlib.

```bash
git clone https://github.com/adotak/Smart-Attendance.git
cd Smart-Attendance
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install opencv-python face-recognition numpy
```

For optional emotion analysis:

```bash
python -m pip install deepface tf-keras
```

## Add registered faces

Create `dataset/` and add one clear image per person. The filename becomes the displayed name:

```text
dataset/
├── Ada_Lovelace.jpg
└── Alan_Turing.png
```

The script uses the first face in an image and warns when an image is unreadable, contains no face, or contains multiple faces.

## Run

```bash
python FaceSentinel_AI.py
```

Useful options:

```bash
python FaceSentinel_AI.py --camera 1 --tolerance 0.45 --scale 0.5
python FaceSentinel_AI.py --emotions --emotion-interval 3
python FaceSentinel_AI.py --dataset people --attendance logs/Attendance.csv
```

Lower `--tolerance` is stricter and reduces false matches. Increase `--scale` when recognition needs more detail, at the cost of performance. The default emotion mode is off because DeepFace is resource-intensive.

## Data and privacy

Face images and attendance records remain on the local machine. Do not commit personal images or generated attendance logs. The included `Attendance.csv` is only a placeholder; use a `.gitignore` entry for real data in deployments.

## Troubleshooting

- **Camera unavailable:** try another `--camera` index and check OS camera permissions.
- **No faces loaded:** use a well-lit, front-facing image in `dataset/`.
- **Too many unknown matches:** increase `--tolerance` slightly.
- **False matches:** lower `--tolerance` and use clearer registration images.
- **DeepFace errors:** run without `--emotions`; recognition and attendance do not depend on it.
