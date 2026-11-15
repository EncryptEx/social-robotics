"""Conservative visual observations from Furhat camera frames.

The detector deliberately supports one reviewed cue: red headwear. It uses a
face detector to limit colour analysis to the area around the participant's
head, then requires the cue to be present in a majority of several frames.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


RED_SCORE_THRESHOLD = 0.06
MIN_CONSENSUS_RATIO = 0.6


@dataclass(frozen=True)
class FrameDetection:
    """Result of inspecting one frame."""

    face_found: bool
    red_headwear_score: float = 0.0

    @property
    def is_red_headwear(self) -> bool:
        return self.face_found and self.red_headwear_score >= RED_SCORE_THRESHOLD


@dataclass(frozen=True)
class VisualObservation:
    """A stable, dialogue-safe observation assembled from several frames."""

    kind: str
    confidence: float
    matching_frames: int
    usable_frames: int

    @property
    def utterance(self) -> str:
        if self.kind == "red_headwear":
            # Colour near a face does not prove that the object is a hat.
            return (
                "It looks like you might be wearing a red hat. "
                "That is a nice splash of colour."
            )
        raise ValueError(f"Unsupported visual observation: {self.kind}")


def decode_camera_frame(event: dict) -> np.ndarray:
    """Decode a Realtime API ``response.camera.data`` event into a BGR image."""

    encoded = event.get("image")
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError("Camera response did not contain an image")

    encoded = encoded.strip()
    if encoded.startswith("data:"):
        try:
            encoded = encoded.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("Malformed camera data URI") from exc

    try:
        jpeg = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Camera image was not valid base64") from exc

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Camera image was not a valid JPEG")
    return frame


class RedHeadwearDetector:
    """Find a face and estimate how much connected red appears around its top."""

    def __init__(self, cascade_path: str | Path | None = None):
        path = Path(cascade_path) if cascade_path else Path(
            cv2.data.haarcascades
        ) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load face detector from {path}")

    def detect_event(self, event: dict) -> FrameDetection:
        return self.detect(decode_camera_frame(event))

    def detect(
        self,
        frame: np.ndarray,
        *,
        face_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> FrameDetection:
        """Inspect a BGR frame.

        ``face_boxes`` is exposed for deterministic tests and calibration.
        Each box is ``(x, y, width, height)`` in image pixels.
        """

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")

        if face_boxes is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            face_boxes = [tuple(int(value) for value in box) for box in detected]

        if len(face_boxes) == 0:
            return FrameDetection(face_found=False)

        # Furhat attends to the closest participant. In a 2-D image, the
        # largest face is the safest available proxy for that same person.
        target_face = max(face_boxes, key=lambda box: box[2] * box[3])
        score = self._red_score_near_face(frame, target_face)
        return FrameDetection(face_found=True, red_headwear_score=score)

    @staticmethod
    def _red_score_near_face(
        frame: np.ndarray, face_box: tuple[int, int, int, int]
    ) -> float:
        height, width = frame.shape[:2]
        x, y, face_width, face_height = face_box
        if face_width <= 0 or face_height <= 0:
            return 0.0

        # Include the crown/brim and a small part of the forehead, but exclude
        # the torso so that a red shirt does not become a red-hat claim.
        left = max(0, int(x - 0.35 * face_width))
        right = min(width, int(x + 1.35 * face_width))
        top = max(0, int(y - 1.05 * face_height))
        bottom = min(height, int(y + 0.25 * face_height))
        region = frame[top:bottom, left:right]
        if region.size == 0:
            return 0.0

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(hsv, (0, 100, 55), (12, 255, 255))
        high_red = cv2.inRange(hsv, (168, 100, 55), (179, 255, 255))
        mask = cv2.bitwise_or(low_red, high_red)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        components, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not components:
            return 0.0

        largest_red_area = max(cv2.contourArea(component) for component in components)
        return float(largest_red_area / region.shape[0] / region.shape[1])


def summarize_detections(
    detections: Iterable[FrameDetection],
) -> VisualObservation | None:
    """Return an observation only when most usable frames support it."""

    usable = [detection for detection in detections if detection.face_found]
    if not usable:
        return None

    matching = [detection for detection in usable if detection.is_red_headwear]
    required = max(2, ceil(len(usable) * MIN_CONSENSUS_RATIO))
    if len(matching) < required:
        return None

    return VisualObservation(
        kind="red_headwear",
        confidence=len(matching) / len(usable),
        matching_frames=len(matching),
        usable_frames=len(usable),
    )


def inspect_image(path: str | Path) -> FrameDetection:
    """Developer helper for checking a still image before a robot rehearsal."""

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    return RedHeadwearDetector().detect(frame)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a still image for red headwear")
    parser.add_argument("image", help="Path to a JPEG or PNG containing one frontal face")
    args = parser.parse_args()
    result = inspect_image(args.image)
    print(
        f"face_found={result.face_found} "
        f"red_headwear={result.is_red_headwear} "
        f"score={result.red_headwear_score:.3f} "
        f"threshold={RED_SCORE_THRESHOLD:.3f}"
    )
