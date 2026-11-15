"""Checks that the social cue changes Furhat's response at the boundary."""

import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


# The dialogue policy can be checked before the robot client is installed.
if "furhat_realtime_api" not in sys.modules:
    try:
        import furhat_realtime_api  # noqa: F401
    except ImportError:
        module = types.ModuleType("furhat_realtime_api")
        module.AsyncFurhatClient = object
        module.Events = object
        sys.modules["furhat_realtime_api"] = module

from lab1 import Demo, HESITATION_SECONDS, response_for
from transcript_validation import TranscriptValidation


class ResponsePolicyTests(unittest.TestCase):
    def test_demo_can_be_created_before_event_loop_starts(self):
        with patch("lab1.AsyncFurhatClient"):
            Demo("127.0.0.1", None)

    def test_prompt_answer_and_hesitation_use_different_branches(self):
        prompt = response_for(HESITATION_SECONDS - 0.01)
        hesitant = response_for(HESITATION_SECONDS)
        self.assertIn("Thanks for telling me", prompt)
        self.assertIn("too personal", hesitant)
        self.assertEqual(response_for(None), hesitant)


class FakeValidator:
    def __init__(self, validations):
        self.validations = iter(validations)
        self.transcripts = []

    async def validate(self, question, transcript):
        self.transcripts.append((question, transcript))
        return next(self.validations)


class ValidatedListeningTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_transcript_is_retried_once(self):
        validator = FakeValidator(
            [
                TranscriptValidation(False, "", "Gibberish", "cohere"),
                TranscriptValidation(True, "Yes, that is right", "Relevant", "cohere"),
            ]
        )
        with patch("lab1.AsyncFurhatClient"):
            demo = Demo("127.0.0.1", None)
        demo.listen = AsyncMock(
            side_effect=[(1.25, "Blue seven window"), (0.4, "Yes, that is right")]
        )
        demo.say = AsyncMock()

        latency, transcript = await demo.listen_for_valid_answer(
            "Do you prefer small gatherings?", validator
        )

        self.assertEqual(latency, 1.25)
        self.assertEqual(transcript, "Yes, that is right")
        demo.say.assert_awaited_once()
        self.assertEqual(len(validator.transcripts), 2)

    async def test_silence_is_not_sent_to_cohere_or_retried(self):
        validator = FakeValidator([])
        with patch("lab1.AsyncFurhatClient"):
            demo = Demo("127.0.0.1", None)
        demo.listen = AsyncMock(return_value=(None, ""))
        demo.say = AsyncMock()

        latency, transcript = await demo.listen_for_valid_answer(
            "What hobby do you enjoy?", validator
        )

        self.assertIsNone(latency)
        self.assertEqual(transcript, "")
        demo.say.assert_not_awaited()
        self.assertEqual(validator.transcripts, [])


if __name__ == "__main__":
    unittest.main()
