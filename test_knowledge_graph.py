"""Tests for semantic fact extraction and the direct-edge graph model."""

import json
import unittest
from types import SimpleNamespace

from knowledge_graph import (
    CohereFactExtractor,
    SemanticFact,
    UPSERT_FACT_QUERY,
)


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


class FactExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_direct_semantic_facts_with_original_provenance(self):
        client = FakeClient(
            {
                "facts": [
                    {
                        "question_key": "interest",
                        "relation": "LIKES",
                        "concept": "Cars",
                    },
                    {
                        "question_key": "setting",
                        "relation": "PREFERS",
                        "concept": "quiet settings",
                    },
                ]
            }
        )
        extractor = CohereFactExtractor("test-key", client=client)

        facts = await extractor.extract(
            {
                "interest": "I really like cars.",
                "setting": "I prefer quiet settings.",
            }
        )

        self.assertEqual(
            facts[0],
            SemanticFact("LIKES", "cars", "interest", "I really like cars."),
        )
        self.assertEqual(facts[0].fact_id, "LIKES:cars")
        self.assertEqual(facts[1].concept_key, "quiet-settings")
        self.assertEqual(client.calls[0]["temperature"], 0)
        self.assertNotIn("Answer", UPSERT_FACT_QUERY)
        self.assertIn("(p)-[r:$($relation)]->(c)", UPSERT_FACT_QUERY)

    async def test_rejects_a_relation_not_allowed_for_question(self):
        client = FakeClient(
            {
                "facts": [
                    {
                        "question_key": "interest",
                        "relation": "PREFERS",
                        "concept": "cars",
                    }
                ]
            }
        )
        with self.assertRaises(ValueError):
            await CohereFactExtractor("test-key", client=client).extract(
                {"interest": "I like cars"}
            )

    async def test_empty_answers_do_not_call_cohere(self):
        client = FakeClient({"facts": []})
        facts = await CohereFactExtractor("test-key", client=client).extract({})
        self.assertEqual(facts, [])
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
