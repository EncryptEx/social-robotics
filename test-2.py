"""Ask a few non-sensitive questions and save the answers as a Neo4j graph."""

import argparse
import asyncio
import os

from lab1 import Demo


QUESTIONS = (
    ("interest", "What is one hobby or interest you enjoy?"),
    ("weekend", "What do you usually like to do on a free weekend?"),
    ("setting", "Do you prefer a quiet setting or a lively one when relaxing?"),
)
QUESTION_START_TIMEOUT = 20.0
ANSWER_PAUSE_TIMEOUT = 3.0
ANSWER_TOTAL_TIMEOUT = 60.0


def save_graph(participant_id: str, answers: dict[str, str], uri: str, user: str, password: str):
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        for topic, answer in answers.items():
            driver.execute_query(
                """
                MERGE (p:Participant {id: $participant_id})
                MERGE (t:Topic {key: $topic})
                MERGE (a:Answer {participant_id: $participant_id, topic: $topic})
                SET a.text = $answer
                MERGE (p)-[:GAVE]->(a)
                MERGE (a)-[:ABOUT]->(t)
                """,
                participant_id=participant_id,
                topic=topic,
                answer=answer,
            )
        records, _, _ = driver.execute_query(
            """
            MATCH (p:Participant {id: $participant_id})-[:GAVE]->(a:Answer)-[:ABOUT]->(t:Topic)
            RETURN t.key AS topic, a.text AS answer
            ORDER BY topic
            """,
            participant_id=participant_id,
        )
        return [(record["topic"], record["answer"]) for record in records]


def check_graph(uri: str, user: str, password: str):
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()


async def ask_with_furhat(host: str, auth_key: str | None) -> dict[str, str]:
    demo = Demo(host, auth_key)
    await demo.furhat.connect()
    demo.furhat.add_handler("response.speak.end", demo.on_speech_end)
    demo.furhat.add_handler("response.hear.start", demo.on_hear_start)
    demo.furhat.add_handler("response.hear.end", demo.on_hear_end)
    answers = {}
    try:
        await demo.wait_for_person()
        await demo.say("Hello. I will ask three questions about everyday preferences and save your answers locally. You may skip any question by staying silent.")
        for topic, question in QUESTIONS:
            await demo.say(question)
            _, text = await demo.listen(
                no_speech_timeout=QUESTION_START_TIMEOUT,
                end_speech_timeout=ANSWER_PAUSE_TIMEOUT,
                answer_timeout=ANSWER_TOTAL_TIMEOUT,
            )
            text = text.strip()
            if text:
                print(f"{topic}: {text}")
                answers[topic] = text
                await demo.say("Thank you.")
            else:
                print(f"{topic}: skipped")
                await demo.say("That's okay. We can skip that one.")
        await demo.say("I have finished the questions.")
    finally:
        try:
            await demo.furhat.request_listen_stop()
        finally:
            await demo.furhat.disconnect()
    return answers


def ask_in_terminal() -> dict[str, str]:
    answers = {}
    for topic, question in QUESTIONS:
        text = input(f"{question} (Enter to skip)\n> ").strip()
        if text:
            answers[topic] = text
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-id", required=True, help="Pseudonymous ID, such as demo01")
    parser.add_argument("--mode", choices=("voice", "terminal"), default="voice")
    parser.add_argument("--host", default="192.168.1.114")
    parser.add_argument("--dry-run", action="store_true", help="Show answers without writing to Neo4j")
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not args.dry_run:
        if not password:
            parser.exit(2, "Set NEO4J_PASSWORD before starting. Use --dry-run to preview without Neo4j.\n")
        try:
            check_graph(uri, user, password)
        except ImportError:
            parser.exit(2, "Neo4j Python driver missing. Run: env/bin/python -m pip install -r requirements.txt\n")
        except Exception as exc:
            parser.exit(2, f"Cannot connect to Neo4j at {uri}: {exc}\n")

    if args.mode == "voice":
        answers = asyncio.run(ask_with_furhat(args.host, os.getenv("FURHAT_AUTH_KEY")))
    else:
        answers = ask_in_terminal()

    if not answers:
        print("No answers given; nothing was saved.")
        return
    if args.dry_run:
        print(f"Preview for {args.participant_id}: {answers}")
        return

    saved = save_graph(args.participant_id, answers, uri, user, password)
    print("Saved graph facts:")
    for topic, answer in saved:
        print(f"  {topic}: {answer}")


if __name__ == "__main__":
    main()
