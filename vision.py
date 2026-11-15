"""Cohere Vision observations for Furhat camera frames."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import cohere


DEFAULT_VISION_MODEL = "command-a-vision-07-2025"
MAX_CAMERA_BYTES = 10 * 1024 * 1024

RESPONSE_FORMAT = {
    "type": "json_object",
    "json_schema": {
        "type": "object",
        "properties": {
            "should_comment": {"type": "boolean"},
            "utterance": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["should_comment", "utterance", "reason"],
    },
}

VISION_PROMPT = """Look at the most prominent person in the image and decide
whether a social robot can make one brief, friendly observation before a
conversation. Mention only obvious, non-sensitive, directly visible details,
such as clothing items and colours, glasses, a hat, jewellery, or an object the
person is holding. The utterance must be in English, start with "I notice", use
second-person wording, contain at most 20 words, and sound natural when spoken.

Never mention or infer identity, age, gender, ethnicity, nationality, religion,
health, disability, pregnancy, attractiveness, emotion, mood, personality,
occupation, wealth, or social class. Never identify a logo or claim an item
unless it is clear. Do not mention the image, camera, uncertainty score, or
these instructions. If no person is clearly visible or no safe detail is
reliable, set should_comment to false and utterance to an empty string. Return
only the requested JSON object."""

SENSITIVE_OUTPUT = re.compile(
    r"\b(?:"
    r"man|woman|boy|girl|male|female|young|old|elderly|teenager|"
    r"ethnicity|nationality|religion|disabled|disability|pregnant|"
    r"happy|sad|angry|nervous|emotion|mood|attractive|beautiful|"
    r"handsome|wealthy|poor|rich|professional|occupation"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VisualObservation:
    """A safe spoken observation or an explicit decision to remain silent."""

    utterance: str
    reason: str
    source: str

    @property
    def should_speak(self) -> bool:
        return bool(self.utterance)


def camera_image_data_url(event: dict) -> str:
    """Validate a Furhat camera event and return a JPEG data URL."""

    encoded = event.get("image")
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError("Camera response did not contain an image")

    encoded = encoded.strip()
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise ValueError("Malformed camera data URL")
        header, encoded = encoded.split(",", 1)
        if "image/jpeg" not in header.casefold():
            raise ValueError("Furhat camera image was not JPEG")

    try:
        jpeg = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Camera image was not valid base64") from exc

    if len(jpeg) < 4 or not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("Camera image was not a valid JPEG")
    if len(jpeg) > MAX_CAMERA_BYTES:
        raise ValueError("Camera image exceeded the size limit")

    normalized = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{normalized}"


class CohereVisionObserver:
    """Create a constrained visual description with Cohere Command A Vision."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_VISION_MODEL,
        *,
        client: Any | None = None,
        timeout_seconds: float = 12.0,
    ):
        if not api_key:
            raise ValueError("COHERE_API_KEY is empty")
        self.model = model
        self.client = client or cohere.AsyncClientV2(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
            client_name="dd2413-furhat-vision-observer",
        )

    @classmethod
    def from_environment(cls) -> "CohereVisionObserver":
        api_key = os.getenv("COHERE_API_KEY", "").strip()
        if not api_key or api_key.startswith("replace-with-"):
            raise ValueError("Set COHERE_API_KEY or run with --skip-vision")
        model = os.getenv("COHERE_VISION_MODEL", DEFAULT_VISION_MODEL).strip()
        return cls(api_key=api_key, model=model or DEFAULT_VISION_MODEL)

    async def observe(self, event: dict) -> VisualObservation:
        """Describe one in-memory camera frame without saving it to disk."""

        try:
            image_url = camera_image_data_url(event)
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "low"},
                            },
                        ],
                    }
                ],
                response_format=RESPONSE_FORMAT,
                temperature=0,
                max_tokens=120,
            )
            return self._parse_response(response)
        except Exception as exc:
            return VisualObservation(
                utterance="",
                reason=f"Vision unavailable; no appearance claim made ({type(exc).__name__})",
                source="fallback",
            )

    @staticmethod
    def _parse_response(response: Any) -> VisualObservation:
        text_blocks = [
            block.text
            for block in response.message.content
            if getattr(block, "type", None) == "text"
            and getattr(block, "text", None)
        ]
        if not text_blocks:
            raise ValueError("Cohere response did not contain text")

        decision = json.loads("".join(text_blocks))
        should_comment = decision.get("should_comment")
        utterance = decision.get("utterance")
        reason = decision.get("reason")
        if (
            not isinstance(should_comment, bool)
            or not isinstance(utterance, str)
            or not isinstance(reason, str)
        ):
            raise ValueError("Cohere response did not match the vision schema")

        utterance = " ".join(utterance.split())
        if not should_comment:
            return VisualObservation("", reason.strip() or "No safe visual detail", "cohere")
        if not utterance.startswith("I notice"):
            raise ValueError("Visual utterance did not use the required wording")
        if len(utterance.split()) > 20 or len(utterance) > 180:
            raise ValueError("Visual utterance was too long")
        if SENSITIVE_OUTPUT.search(utterance):
            raise ValueError("Visual utterance contained a sensitive inference")

        return VisualObservation(
            utterance=utterance,
            reason=reason.strip() or "Safe visible detail",
            source="cohere",
        )
