"""Run the controlled factual/inference Furhat interaction for DD2413."""

import argparse
import asyncio
import os
import time

from furhat_realtime_api import AsyncFurhatClient, Events

from knowledge_graph import load_facts
from study_dialogue import CONDITIONS, CohereDialogueGenerator, DialoguePair, STUDY_TOPIC
from vision import CohereVisionObserver, VisualObservation


HESITATION_SECONDS = 3.0
LISTEN_TIMEOUT_SECONDS = 8.0


def response_for(latency: float | None, condition: str = "inference") -> str:
    """Treat a long pause as a cue to reduce the intensity of the interaction."""
    if latency is None or latency >= HESITATION_SECONDS:
        if condition == "inference":
            return "I may have made that too personal. We can leave the guess there. Thank you for talking with me."
        return "We can leave that question there. Thank you for talking with me."
    if condition == "inference":
        return "Thanks for telling me. I appreciate your reaction to the guess."
    return "Thanks for telling me. I appreciate your answer."


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
        observer: CohereVisionObserver,
    ) -> VisualObservation | None:
        """Ask Cohere for one safe observation from an in-memory camera frame."""

        print("Capturing one camera frame for a visual opening...")
        try:
            event = await self.furhat.request_camera_once()
        except Exception as exc:
            print(
                "Visual observation: source=fallback, "
                f"reason='Camera unavailable ({type(exc).__name__})'"
            )
            return None
        observation = await observer.observe(event)
        print(
            "Visual observation: "
            f"source={observation.source}, reason={observation.reason!r}, "
            f"details={observation.visible_details!r}, "
            f"utterance={observation.utterance!r}"
        )
        return observation if observation.should_speak else None

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

    async def run(
        self,
        study_utterance: str,
        condition: str,
        vision_observer: CohereVisionObserver | None = None,
    ):
        await self.furhat.connect()
        self.furhat.add_handler(Events.response_speak_end, self.on_speech_end)
        self.furhat.add_handler(Events.response_hear_start, self.on_hear_start)
        self.furhat.add_handler(Events.response_hear_end, self.on_hear_end)
        try:
            await self.wait_for_person()
            if vision_observer is not None:
                observation = await self.observe_user(vision_observer)
                if observation is not None:
                    # The visual remark deliberately happens before the study
                    # dialogue, as required by the project demonstration.
                    await self.say(observation.utterance)
            permission = (
                "May I try a small guess based on something you shared?"
                if condition == "inference"
                else "May I ask a short question based on something you shared?"
            )
            await self.say(f"Hello. I am Furhat. {permission}")
            # The operator confirms consent before the scripted interaction proceeds.
            await asyncio.to_thread(input, "Operator: press Enter after the participant agrees, or Ctrl+C to stop: ")
            await self.say(study_utterance)
            latency, transcript = await self.listen()
            print(f"Response latency: {latency!r} seconds; transcript: {transcript!r}")
            await self.say(response_for(latency, condition))
        finally:
            try:
                await self.furhat.request_listen_stop()
            finally:
                await self.furhat.disconnect()


def print_review(
    participant_id: str,
    condition: str,
    pair: DialoguePair,
) -> None:
    """Show both matched variants so a researcher can check the manipulation."""

    print("\n--- REQUIRED RESEARCHER REVIEW ---")
    print(f"Participant: {participant_id}")
    print(f"Hard-coded topic: {STUDY_TOPIC}")
    print(f"Source facts: {', '.join(pair.source_fact_ids)}")
    print(f"Factual:   {pair.factual_utterance}")
    print(f"Inference: {pair.inference_utterance}")
    print(f"Assigned condition: {condition}")
    print("----------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.114", help="Robot IP address")
    parser.add_argument("--participant-id", required=True, help="Pseudonymous graph ID")
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip the camera-based opening remark (for troubleshooting only)",
    )
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    if not password or password.startswith("replace-with-"):
        parser.error("Set NEO4J_PASSWORD before loading participant facts")
    try:
        facts = load_facts(args.participant_id, uri, user, password)
    except Exception as exc:
        parser.exit(2, f"Cannot load facts from Neo4j at {uri}: {exc}\n")
    if not facts:
        parser.error(
            f"No explicit semantic facts found for participant {args.participant_id!r}"
        )

    try:
        dialogue_generator = CohereDialogueGenerator.from_environment()
        vision_observer = (
            None if args.skip_vision else CohereVisionObserver.from_environment()
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        pair = asyncio.run(dialogue_generator.generate(facts))
        print_review(args.participant_id, args.condition, pair)
        approval = input(
            "Researcher: type APPROVE after checking both variants, or Ctrl+C to stop: "
        )
        if approval.strip() != "APPROVE":
            parser.exit(2, "Dialogue was not approved; robot interaction cancelled.\n")
        asyncio.run(
            Demo(args.host, os.getenv("FURHAT_AUTH_KEY")).run(
                pair.for_condition(args.condition),
                args.condition,
                vision_observer=vision_observer,
            )
        )
    except (KeyboardInterrupt, EOFError):
        print("Demo cancelled.")
    except OSError as exc:
        parser.exit(1, f"Cannot connect to Furhat at {args.host}:9000 ({exc}). Check the robot IP, network, and Realtime API setting.\n")
    except Exception as exc:
        parser.exit(1, f"Study interaction failed safely: {exc}\n")


if __name__ == "__main__":
    main()
