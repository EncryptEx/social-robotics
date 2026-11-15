"""Tests for Furhat camera decoding and safe Cohere visual observations."""

import base64
import json
import unittest
from types import SimpleNamespace

from vision import CohereVisionObserver, camera_image_data_url


JPEG = b"\xff\xd8\xff\xe0test-frame\xff\xd9"


def camera_event() -> dict:
    return {"image": base64.b64encode(JPEG).decode("ascii")}


def cohere_response(payload: dict):
    block = SimpleNamespace(type="text", text=json.dumps(payload))
    return SimpleNamespace(message=SimpleNamespace(content=[block]))


class FakeCohereClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class CameraEncodingTests(unittest.TestCase):
    def test_converts_furhat_jpeg_to_data_url(self):
        result = camera_image_data_url(camera_event())
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(result.split(",", 1)[1]), JPEG)

    def test_rejects_missing_invalid_and_non_jpeg_images(self):
        with self.assertRaises(ValueError):
            camera_image_data_url({})
        with self.assertRaises(ValueError):
            camera_image_data_url({"image": "not base64!"})
        with self.assertRaises(ValueError):
            camera_image_data_url(
                {"image": base64.b64encode(b"plain text").decode("ascii")}
            )


class VisionObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_general_description_is_spoken(self):
        client = FakeCohereClient(
            cohere_response(
                {
                    "should_comment": True,
                    "utterance": "I notice you're wearing a blue sweater and round glasses.",
                    "reason": "Two clear visible details",
                }
            )
        )
        observer = CohereVisionObserver("test-key", client=client)

        result = await observer.observe(camera_event())

        self.assertTrue(result.should_speak)
        self.assertEqual(result.source, "cohere")
        self.assertIn("blue sweater", result.utterance)
        call = client.calls[0]
        self.assertEqual(call["model"], "command-a-vision-07-2025")
        self.assertEqual(call["temperature"], 0)
        content = call["messages"][0]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["detail"], "low")

    async def test_no_clear_person_produces_no_comment(self):
        client = FakeCohereClient(
            cohere_response(
                {
                    "should_comment": False,
                    "utterance": "",
                    "reason": "No person clearly visible",
                }
            )
        )
        result = await CohereVisionObserver("test-key", client=client).observe(
            camera_event()
        )
        self.assertFalse(result.should_speak)
        self.assertEqual(result.source, "cohere")

    async def test_sensitive_description_is_suppressed(self):
        client = FakeCohereClient(
            cohere_response(
                {
                    "should_comment": True,
                    "utterance": "I notice you're a young woman wearing a green shirt.",
                    "reason": "Visible person",
                }
            )
        )
        result = await CohereVisionObserver("test-key", client=client).observe(
            camera_event()
        )
        self.assertFalse(result.should_speak)
        self.assertEqual(result.source, "fallback")
        self.assertIn("ValueError", result.reason)

    async def test_api_failure_makes_no_appearance_claim(self):
        client = FakeCohereClient(error=TimeoutError("offline"))
        result = await CohereVisionObserver("test-key", client=client).observe(
            camera_event()
        )
        self.assertFalse(result.should_speak)
        self.assertEqual(result.source, "fallback")
        self.assertIn("TimeoutError", result.reason)

    async def test_wrong_opening_or_long_output_is_suppressed(self):
        for utterance in (
            "You are wearing a blue shirt.",
            "I notice " + "very " * 21 + "bright clothes.",
        ):
            with self.subTest(utterance=utterance):
                client = FakeCohereClient(
                    cohere_response(
                        {
                            "should_comment": True,
                            "utterance": utterance,
                            "reason": "Visible clothes",
                        }
                    )
                )
                result = await CohereVisionObserver(
                    "test-key", client=client
                ).observe(camera_event())
                self.assertFalse(result.should_speak)


if __name__ == "__main__":
    unittest.main()
