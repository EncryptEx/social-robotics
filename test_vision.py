"""Tests for camera decoding, spatial filtering, and multi-frame consensus."""

import base64
import unittest

import cv2
import numpy as np

from vision import (
    FrameDetection,
    RED_SCORE_THRESHOLD,
    RedHeadwearDetector,
    decode_camera_frame,
    summarize_detections,
)


class CameraDecodingTests(unittest.TestCase):
    def test_decodes_base64_jpeg(self):
        image = np.full((20, 30, 3), 255, dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(encoded_ok)

        decoded = decode_camera_frame(
            {"image": base64.b64encode(encoded.tobytes()).decode("ascii")}
        )

        self.assertEqual(decoded.shape, image.shape)

    def test_rejects_missing_or_invalid_data(self):
        with self.assertRaises(ValueError):
            decode_camera_frame({})
        with self.assertRaises(ValueError):
            decode_camera_frame({"image": "not base64!"})


class RedHeadwearTests(unittest.TestCase):
    def setUp(self):
        self.detector = RedHeadwearDetector()
        self.face = (80, 90, 50, 50)

    def test_red_above_face_is_detected(self):
        image = np.full((220, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (70, 42), (140, 86), (0, 0, 255), thickness=-1)

        result = self.detector.detect(image, face_boxes=[self.face])

        self.assertTrue(result.face_found)
        self.assertTrue(result.is_red_headwear)
        self.assertGreater(result.red_headwear_score, RED_SCORE_THRESHOLD)

    def test_red_shirt_below_face_is_ignored(self):
        image = np.full((220, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (55, 150), (155, 215), (0, 0, 255), thickness=-1)

        result = self.detector.detect(image, face_boxes=[self.face])

        self.assertTrue(result.face_found)
        self.assertFalse(result.is_red_headwear)
        self.assertEqual(result.red_headwear_score, 0.0)

    def test_no_face_produces_no_claim(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = self.detector.detect(image, face_boxes=[])
        self.assertFalse(result.face_found)
        self.assertFalse(result.is_red_headwear)

    def test_smaller_bystander_does_not_override_closest_face(self):
        image = np.full((260, 320, 3), 255, dtype=np.uint8)
        closest_face = (120, 100, 80, 80)
        smaller_bystander = (20, 80, 40, 40)
        cv2.rectangle(image, (10, 40), (70, 77), (0, 0, 255), thickness=-1)

        result = self.detector.detect(
            image, face_boxes=[smaller_bystander, closest_face]
        )

        self.assertTrue(result.face_found)
        self.assertFalse(result.is_red_headwear)


class ConsensusTests(unittest.TestCase):
    def test_majority_of_usable_frames_produces_cautious_comment(self):
        detections = [
            FrameDetection(True, 0.12),
            FrameDetection(True, 0.10),
            FrameDetection(True, 0.09),
            FrameDetection(True, 0.0),
            FrameDetection(False, 0.0),
        ]

        observation = summarize_detections(detections)

        self.assertIsNotNone(observation)
        self.assertEqual(observation.matching_frames, 3)
        self.assertEqual(observation.usable_frames, 4)
        self.assertEqual(observation.confidence, 0.75)
        self.assertIn("might", observation.utterance)
        self.assertIn("red hat", observation.utterance)

    def test_single_positive_frame_is_not_enough(self):
        detections = [
            FrameDetection(True, 0.20),
            FrameDetection(True, 0.0),
            FrameDetection(True, 0.0),
        ]
        self.assertIsNone(summarize_detections(detections))

    def test_frames_without_faces_do_not_create_an_observation(self):
        self.assertIsNone(
            summarize_detections([FrameDetection(False), FrameDetection(False)])
        )


if __name__ == "__main__":
    unittest.main()
