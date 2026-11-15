"""Short, scripted Furhat interaction for DD2413 Lab 1."""

import argparse
import asyncio
import os
import time

from furhat_realtime_api import AsyncFurhatClient, Events

from transcript_validation import CohereTranscriptValidator
from vision import RedHeadwearDetector, VisualObservation, summarize_detections


PROFILES = {
    "games": (
        "You told us that you enjoy video games and quiet weekends.",
        "I might be wrong, but I would guess you prefer smaller gatherings. Is that right?",
    ),
    "reading": (
        "You told us that you enjoy reading and quiet weekends.",
        "I might be wrong, but I would guess you like having time to yourself. Is that right?",
    ),
}
HESITATION_SECONDS = 3.0
LISTEN_TIMEOUT_SECONDS = 8.0
VISION_FRAMES = 5
VISION_FRAME_INTERVAL_SECONDS = 0.35


def response_for(latency: float | None) -> str:
    """Treat a long pause as a cue to reduce the intensity of the interaction."""
    if latency is None or latency >= HESITATION_SECONDS:
        return "I may have made that too personal. We can leave the guess there. Thank you for talking with me."
    return "Thanks for telling me. A guess can sound different from a fact, and I appreciate your reaction."


class Demo:
    def __init__(self, host: str, auth_key: str | None):
        self.furhat = AsyncFurhatClient(host, auth_key=auth_key)
        self.speech_end: asyncio.Future | None = None
        self.hear_start: asyncio.Future | None = None
        self.hear_end: asyncio.Future | None = None

    async def on_speech_end(self, event):
        if self.speech_end is not None and not self.speech_end.done():
            self.speech_end.set_result(event)

    async def on_hear_start(self, event):
        if self.hear_start is not None and not self.hear_start.done():
            self.hear_start.set_result(time.monotonic())

    async def on_hear_end(self, event):
        if self.hear_end is not None and not self.hear_end.done():
            self.hear_end.set_result(event.get("text", ""))

    async def say(self, text: str):
        self.speech_end = asyncio.get_running_loop().create_future()
        print(f"Furhat: {text}")
        await self.furhat.request_speak_text(text)
        event = await asyncio.wait_for(self.speech_end, timeout=30)
        self.speech_end = None
        if event.get("failed") or event.get("aborted"):
            raise RuntimeError("Furhat speech failed or was interrupted")

    async def wait_for_person(self):
        print("Waiting for a person in Furhat's camera view...")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = await self.furhat.request_users_once()
            if result.get("users"):
                await self.furhat.request_attend_user("closest")
                return
            await asyncio.sleep(0.5)
        raise TimeoutError("No participant detected in 30 seconds")

    async def observe_user(
        self,
        detector: RedHeadwearDetector,
        frame_count: int = VISION_FRAMES,
        frame_interval: float = VISION_FRAME_INTERVAL_SECONDS,
    ) -> VisualObservation | None:
        """Use several in-memory camera frames to form one cautious observation."""

        detections = []
        print(f"Checking {frame_count} camera frames for a reviewed visual cue...")
        for frame_number in range(1, frame_count + 1):
            try:
                event = await self.furhat.request_camera_once()
                detection = detector.detect_event(event)
            except Exception as exc:
                # Visual personalization is optional at runtime. A bad frame or
                # unavailable camera must not prevent the consented dialogue.
                print(f"Camera frame {frame_number} skipped: {exc}")
            else:
                detections.append(detection)
                print(
                    f"Camera frame {frame_number}: face={detection.face_found}, "
                    f"red-headwear score={detection.red_headwear_score:.3f}"
                )
            if frame_number < frame_count:
                await asyncio.sleep(frame_interval)

        observation = summarize_detections(detections)
        if observation is None:
            print(
                "Visual observation: no stable red-headwear cue; "
                "saying nothing about appearance."
            )
        else:
            print(
                "Visual observation: "
                f"{observation.kind}, confidence={observation.confidence:.2f}, "
                f"frames={observation.matching_frames}/{observation.usable_frames}"
            )
        return observation

    async def listen(
        self,
        no_speech_timeout: float = LISTEN_TIMEOUT_SECONDS,
        end_speech_timeout: float = 1.0,
        answer_timeout: float = 15.0,
    ) -> tuple[float | None, str]:
        self.hear_start = asyncio.get_running_loop().create_future()
        self.hear_end = asyncio.get_running_loop().create_future()
        # Start the clock when listening is ready, not when speech synthesis ends.
        await self.furhat.request_listen_start(
            no_speech_timeout=no_speech_timeout,
            end_speech_timeout=end_speech_timeout,
            stop_no_speech=True,
            stop_user_end=True,
        )
        start = time.monotonic()
        try:
            heard_at = await asyncio.wait_for(self.hear_start, timeout=no_speech_timeout + 2)
            latency = max(0.0, heard_at - start)
            try:
                text = await asyncio.wait_for(self.hear_end, timeout=answer_timeout)
            except asyncio.TimeoutError:
                text = ""
            return latency, text
        except asyncio.TimeoutError:
            return None, ""
        finally:
            self.hear_start = None
            self.hear_end = None
            await self.furhat.request_listen_stop()

    async def listen_for_valid_answer(
        self,
        question: str,
        validator: CohereTranscriptValidator | None,
        *,
        no_speech_timeout: float = LISTEN_TIMEOUT_SECONDS,
        end_speech_timeout: float = 1.0,
        answer_timeout: float = 15.0,
        max_attempts: int = 2,
    ) -> tuple[float | None, str]:
        """Listen and ask once for repetition when Cohere rejects the transcript."""

        first_latency = None
        for attempt in range(max_attempts):
            latency, transcript = await self.listen(
                no_speech_timeout=no_speech_timeout,
                end_speech_timeout=end_speech_timeout,
                answer_timeout=answer_timeout,
            )
            if attempt == 0:
                first_latency = latency

            transcript = transcript.strip()
            if not transcript or validator is None:
                return first_latency, transcript

            validation = await validator.validate(question, transcript)
            print(
                "Transcript validation: "
                f"accepted={validation.accepted}, source={validation.source}, "
                f"reason={validation.reason!r}"
            )
            if validation.accepted:
                return first_latency, validation.transcript

            if attempt + 1 < max_attempts:
                await self.say(
                    "I may not have heard that correctly. Could you say that again?"
                )

        return first_latency, ""

    async def run(
        self,
        profile: str,
        vision_enabled: bool = True,
        transcript_validator: CohereTranscriptValidator | None = None,
    ):
        await self.furhat.connect()
        self.furhat.add_handler(Events.response_speak_end, self.on_speech_end)
        self.furhat.add_handler(Events.response_hear_start, self.on_hear_start)
        self.furhat.add_handler(Events.response_hear_end, self.on_hear_end)
        try:
            await self.wait_for_person()
            if vision_enabled:
                observation = await self.observe_user(RedHeadwearDetector())
                if observation is not None:
                    # The visual remark deliberately happens before the study
                    # dialogue, as required by the project demonstration.
                    await self.say(observation.utterance)
            await self.say("Hello. I am Furhat. May I try a small guess based on something you shared?")
            # The operator confirms consent before the scripted interaction proceeds.
            await asyncio.to_thread(input, "Operator: press Enter after the participant agrees, or Ctrl+C to stop: ")
            fact, guess = PROFILES[profile]
            await self.say(fact)
            await self.say(guess)
            latency, transcript = await self.listen_for_valid_answer(
                guess, transcript_validator
            )
            print(f"Response latency: {latency!r} seconds; transcript: {transcript!r}")
            if latency is not None and not transcript:
                await self.say(
                    "I am sorry I still could not understand clearly. "
                    "We can leave the guess there."
                )
            else:
                await self.say(response_for(latency))
        finally:
            try:
                await self.furhat.request_listen_stop()
            finally:
                await self.furhat.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.114", help="Robot IP address")
    parser.add_argument("--profile", choices=PROFILES, default="games")
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip the camera-based opening remark (for troubleshooting only)",
    )
    parser.add_argument(
        "--skip-cohere",
        action="store_true",
        help="Accept raw speech transcripts without Cohere validation",
    )
    args = parser.parse_args()
    try:
        validator = (
            None if args.skip_cohere else CohereTranscriptValidator.from_environment()
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        asyncio.run(
            Demo(args.host, os.getenv("FURHAT_AUTH_KEY")).run(
                args.profile,
                vision_enabled=not args.skip_vision,
                transcript_validator=validator,
            )
        )
    except (KeyboardInterrupt, EOFError):
        print("Demo cancelled.")
    except OSError as exc:
        parser.exit(1, f"Cannot connect to Furhat at {args.host}:9000 ({exc}). Check the robot IP, network, and Realtime API setting.\n")


if __name__ == "__main__":
    main()
