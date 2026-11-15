"""Unit tests for local and Cohere-backed transcript validation."""

import json
import unittest
from types import SimpleNamespace

from transcript_validation import CohereTranscriptValidator


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


class TranscriptValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_meaningful_answer_is_accepted_without_rewriting(self):
        client = FakeCohereClient(
            cohere_response({"valid": True, "reason": "Relevant answer"})
        )
        validator = CohereTranscriptValidator("test-key", client=client)

        result = await validator.validate(
            "What hobby do you enjoy?", "  Painting tiny figures.  "
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.transcript, "Painting tiny figures.")
        self.assertEqual(result.source, "cohere")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["temperature"], 0)
        self.assertEqual(client.calls[0]["response_format"]["type"], "json_object")
        self.assertIn("json_schema", client.calls[0]["response_format"])

    async def test_incoherent_answer_is_rejected(self):
        client = FakeCohereClient(
            cohere_response({"valid": False, "reason": "Unrelated fragments"})
        )
        validator = CohereTranscriptValidator("test-key", client=client)

        result = await validator.validate(
            "What do you do on weekends?", "Window seven because train blue"
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.transcript, "")
        self.assertEqual(result.reason, "Unrelated fragments")

    async def test_empty_and_obvious_artifact_do_not_spend_api_calls(self):
        client = FakeCohereClient()
        validator = CohereTranscriptValidator("test-key", client=client)

        empty = await validator.validate("Question", "   ")
        artifact = await validator.validate("Question", "[inaudible]")

        self.assertFalse(empty.accepted)
        self.assertFalse(artifact.accepted)
        self.assertEqual(empty.source, "local")
        self.assertEqual(client.calls, [])

    async def test_api_failure_preserves_interaction_and_marks_fallback(self):
        client = FakeCohereClient(error=TimeoutError("offline"))
        validator = CohereTranscriptValidator("test-key", client=client)

        result = await validator.validate("What hobby?", "I enjoy climbing")

        self.assertTrue(result.accepted)
        self.assertEqual(result.transcript, "I enjoy climbing")
        self.assertEqual(result.source, "fallback")
        self.assertIn("TimeoutError", result.reason)

    async def test_malformed_response_uses_visible_fallback(self):
        client = FakeCohereClient(cohere_response({"unexpected": True}))
        validator = CohereTranscriptValidator("test-key", client=client)

        result = await validator.validate("What hobby?", "I enjoy climbing")

        self.assertTrue(result.accepted)
        self.assertEqual(result.source, "fallback")
        self.assertIn("ValueError", result.reason)


if __name__ == "__main__":
    unittest.main()
