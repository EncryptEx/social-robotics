"""Tests for matched factual and inference study dialogue generation."""

import json
import unittest
from types import SimpleNamespace

from knowledge_graph import SemanticFact
from study_dialogue import CohereDialogueGenerator


def cohere_response(payload: dict):
    block = SimpleNamespace(type="text", text=json.dumps(payload))
    return SimpleNamespace(message=SimpleNamespace(content=[block]))


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return cohere_response(self.payload)


FACTS = [
    SemanticFact("LIKES", "cars", "interest", "I like cars."),
    SemanticFact("PREFERS", "quiet settings", "setting", "I prefer quiet settings."),
]


def valid_payload():
    return {
        "factual_utterance": (
            "You told us you like cars and prefer quiet settings. What do you "
            "enjoy most about relaxed free time with other people?"
        ),
        "inference_utterance": (
            "You told us you like cars and prefer quiet settings. I might be "
            "wrong, but I would guess you enjoy small-group outings. Is that right?"
        ),
        "inference_claim": "you enjoy small-group outings",
        "source_fact_ids": ["LIKES:cars", "PREFERS:quiet-settings"],
    }


class DialogueGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_both_conditions_from_the_same_facts(self):
        client = FakeClient(valid_payload())
        pair = await CohereDialogueGenerator("test-key", client=client).generate(FACTS)

        self.assertIn("What do you enjoy", pair.for_condition("factual"))
        self.assertIn("I might be wrong", pair.for_condition("inference"))
        self.assertEqual(
            pair.source_fact_ids,
            ("LIKES:cars", "PREFERS:quiet-settings"),
        )
        call = client.calls[0]
        self.assertEqual(call["temperature"], 0)
        prompt = call["messages"][0]["content"]
        self.assertIn("Hard-coded study topic", prompt)
        self.assertIn('"relation": "LIKES"', prompt)

    async def test_rejects_inference_wording_in_factual_condition(self):
        payload = valid_payload()
        payload["factual_utterance"] = (
            "You told us you like cars and prefer quiet settings. Maybe you "
            "enjoy relaxed free time with other people; what do you think?"
        )
        with self.assertRaises(ValueError):
            await CohereDialogueGenerator(
                "test-key", client=FakeClient(payload)
            ).generate(FACTS)

    async def test_rejects_sensitive_inference(self):
        payload = valid_payload()
        payload["inference_claim"] = "you are an introvert"
        with self.assertRaises(ValueError):
            await CohereDialogueGenerator(
                "test-key", client=FakeClient(payload)
            ).generate(FACTS)


if __name__ == "__main__":
    unittest.main()
