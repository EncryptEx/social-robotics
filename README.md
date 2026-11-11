# DD2413 Lab 1: personal guesses with Furhat

This is a short prototype for the project *When Does Personalization Become Creepy?* Furhat refers to a disclosed fact, makes one explicitly uncertain inference, and adjusts its response to the participant's pause before replying. It is a demonstration of an observed social cue, not the full two-condition study.

## Run

Use a laptop on the same network as Furhat. Enable the Realtime API for local network clients in Furhat's web interface first; see [setup](docs/setup.md).
Python 3.10 or newer is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python lab1.py --host 192.168.1.114 --profile games
```

If you already created the environment as `env/`, use `source env/bin/activate` instead of `.venv/bin/activate`, or run `env/bin/python lab1.py` directly.

If the Realtime API requires a key, set `FURHAT_AUTH_KEY` in your shell. This is a Realtime API key, not the web interface password. Do not commit it.

Two reviewed profiles are available: `games` and `reading`. Only use a profile whose facts the participant actually supplied. The operator presses Enter only after the participant agrees to the guess; Ctrl+C ends the interaction.

See [the documentation index](docs/README.md) for the assignment, rehearsal, and submission details.

## Neo4j questionnaire prototype

[`test-2.py`](test-2.py) asks three non-sensitive questions and stores each answer as a `Participant → Answer → Topic` path in Neo4j. See [Neo4j setup and commands](docs/neo4j.md).
