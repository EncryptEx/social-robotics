"""Checks that the social cue changes Furhat's response at the boundary."""

import sys
import types
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
