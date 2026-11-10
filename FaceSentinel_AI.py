"""FaceSentinel AI: webcam face recognition with daily attendance tracking."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None


LOGGER = logging.getLogger("facesentinel")
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Settings:
    dataset: Path
    attendance_file: Path
    camera: int = 0
    scale: float = 0.25
    tolerance: float = 0.50
    emotions: bool = False
    emotion_interval: float = 2.0


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description="Recognize registered faces and record one attendance entry per day."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--attendance", type=Path, default=Path("Attendance.csv"))
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Face-recognition frame scale (0.25 is fast; use 0.5 for accuracy).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.50,
        help="Maximum face distance to accept (lower is stricter).",
    )
    parser.add_argument(
        "--emotions",
        action="store_true",
        help="Enable optional DeepFace emotion analysis.",
    )
    parser.add_argument("--emotion-interval", type=float, default=2.0)
    args = parser.parse_args()

    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    if not 0.1 <= args.scale <= 1.0:
        parser.error("--scale must be between 0.1 and 1.0")
    if not 0.2 <= args.tolerance <= 0.8:
        parser.error("--tolerance must be between 0.2 and 0.8")
    if args.emotion_interval <= 0:
        parser.error("--emotion-interval must be greater than zero")
    return Settings(
        dataset=args.dataset,
        attendance_file=args.attendance,
        camera=args.camera,
        scale=args.scale,
        tolerance=args.tolerance,
        emotions=args.emotions,
        emotion_interval=args.emotion_interval,
    )


def load_known_faces(dataset: Path) -> tuple[list[np.ndarray], list[str]]:
    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset directory '{dataset}' does not exist. Create it and add face images."
        )

    encodings: list[np.ndarray] = []
    names: list[str] = []
    for image_path in sorted(dataset.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path.name)
            continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_image, model="hog")
        image_encodings = face_recognition.face_encodings(rgb_image, locations)
        if not image_encodings:
            LOGGER.warning("No face found in %s; skipping it.", image_path.name)
            continue
        if len(image_encodings) > 1:
            LOGGER.warning("Multiple faces found in %s; using the first one.", image_path.name)
        encodings.append(image_encodings[0])
        names.append(image_path.stem.replace("_", " ").strip().upper())

    if not encodings:
        raise ValueError(f"No usable face images found in '{dataset}'.")
    LOGGER.info("Loaded %d registered face(s): %s", len(names), ", ".join(names))
    return encodings, names


def load_todays_attendance(attendance_file: Path, today: str) -> set[str]:
    marked: set[str] = set()
    if not attendance_file.exists():
        return marked
    with attendance_file.open("r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) >= 2 and row[0].strip() and row[1].startswith(today):
                marked.add(row[0].strip().upper())
    return marked


def mark_attendance(
    name: str, attendance_file: Path, marked_today: set[str]
) -> bool:
    normalized_name = name.strip().upper()
    if normalized_name in marked_today:
        return False
    attendance_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with attendance_file.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([normalized_name, timestamp])
    marked_today.add(normalized_name)
    LOGGER.info("Attendance marked for %s at %s", normalized_name, timestamp)
    return True


def analyze_emotion(face_crop: np.ndarray) -> Optional[str]:
    if DeepFace is None or face_crop.size == 0:
        return None
    try:
        result = DeepFace.analyze(
            face_crop, actions=["emotion"], enforce_detection=False, silent=True
        )
    except Exception as error:
        LOGGER.debug("Emotion analysis unavailable for this frame: %s", error)
        return None
    result = result[0] if isinstance(result, list) and result else result
    emotion = result.get("dominant_emotion") if isinstance(result, dict) else None
    return str(emotion) if emotion else None


def draw_face(
    frame: np.ndarray,
    location: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1
    )
    text_top = max(0, bottom - text_height - 12)
    cv2.rectangle(
        frame, (left, text_top), (left + text_width + 12, bottom), color, cv2.FILLED
    )
    cv2.putText(
        frame,
        label,
        (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run(settings: Settings) -> int:
    emotions_enabled = settings.emotions and DeepFace is not None
    if settings.emotions and not emotions_enabled:
        LOGGER.warning("DeepFace is not installed; continuing without emotions.")

    known_encodings, known_names = load_known_faces(settings.dataset)
    known_encoding_matrix = np.asarray(known_encodings)
    today = datetime.now().strftime("%Y-%m-%d")
    marked_today = load_todays_attendance(settings.attendance_file, today)
    LOGGER.info("%d attendance record(s) already exist for today.", len(marked_today))

    camera = cv2.VideoCapture(settings.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {settings.camera}.")

    last_time = time.monotonic()
    emotion_cache: dict[int, tuple[float, str]] = {}
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera stopped returning frames.")

            small_frame = cv2.resize(frame, None, fx=settings.scale, fy=settings.scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            encodings = face_recognition.face_encodings(rgb_small_frame, locations)

            for index, (encoding, small_location) in enumerate(zip(encodings, locations)):
                distances = np.linalg.norm(known_encoding_matrix - encoding, axis=1)
                best_index = int(np.argmin(distances))
                distance = float(distances[best_index])
                top, right, bottom, left = (
                    int(value / settings.scale) for value in small_location
                )
                crop = frame[max(0, top):bottom, max(0, left):right]

                if distance <= settings.tolerance:
                    name = known_names[best_index]
                    just_marked = mark_attendance(
                        name, settings.attendance_file, marked_today
                    )
                    label = f"{name} {'JUST MARKED' if just_marked else 'LOGGED'}"
                    color = (0, 200, 0) if just_marked else (0, 165, 255)
                    if emotions_enabled:
                        cached_at, emotion = emotion_cache.get(index, (0.0, ""))
                        if time.monotonic() - cached_at >= settings.emotion_interval:
                            emotion = analyze_emotion(crop) or ""
                            emotion_cache[index] = (time.monotonic(), emotion)
                        if emotion:
                            label += f" ({emotion})"
                else:
                    label = f"UNKNOWN ({distance:.2f})"
                    color = (0, 0, 255)
                draw_face(frame, (top, right, bottom, left), label, color)

            now = time.monotonic()
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now
            cv2.putText(
                frame, f"FPS: {fps:.1f} | Faces: {len(locations)}",
                (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.imshow("FaceSentinel AI - Press Q or Enter to exit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 13, 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        return run(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
