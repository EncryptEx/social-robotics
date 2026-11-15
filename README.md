# Furhat personalisation study

Prototype for the DD2413 project **“When Does Personalization Become Creepy?”**.
Before the scripted conversation starts, Furhat captures one camera frame and
asks Cohere Vision for a brief, non-sensitive observation about the participant.
It may mention visible clothing, colours, accessories, or a held object—for
example, a red hat—but it is not limited to any single feature.

The robot then contrasts a fact disclosed by the participant with a cautious
personal inference and adapts its closing response to the participant's pause
before replying.

## Interaction flow

1. Furhat's built-in user detector waits for a participant and attends to the
   closest person.
2. Furhat captures one JPEG frame with its camera.
3. The frame is sent in memory to Cohere `command-a-vision-07-2025`.
4. Cohere returns a structured decision and, when safe, a short line beginning
   with “I notice…”. Furhat says it before the main dialogue.
5. If the image is unclear, Cohere is unavailable, or the output violates the
   local safety checks, Furhat makes no appearance claim and continues.
6. Furhat asks permission to make the scripted personal guess. The operator
   advances only after the participant agrees.
7. A prompt reply receives a neutral acknowledgement. A pause of three seconds
   or more, or silence, makes Furhat back off and end the topic.

Images are never written to disk by the interaction. They are sent to Cohere
only for the opening observation.

## Visual safety boundary

Cohere is instructed to describe only directly visible, non-sensitive details:

- clothing items and colours;
- glasses, hats, jewellery, and similar accessories;
- an object the participant is clearly holding.

It must not mention or infer identity, age, gender, ethnicity, nationality,
religion, health, disability, pregnancy, attractiveness, emotion, personality,
occupation, wealth, or social class. The code also rejects long responses,
unexpected wording, and a set of sensitive terms before anything is spoken.
This is a safety layer, not a guarantee; the team must rehearse and review the
actual output before collecting study data.

## Repository layout

| Path | Purpose |
| --- | --- |
| `lab1.py` | Furhat connection, visual opening, scripted dialogue, and timing policy |
| `vision.py` | Furhat JPEG validation, Cohere Vision request, and output safety checks |
| `test_lab1.py` | Dialogue timing-policy tests |
| `test_vision.py` | Mocked Cohere, camera-data, safety, and fallback tests |
| `test-2.py` | Optional questionnaire that stores disclosed answers in Neo4j |
| `compose.yaml` | Local Neo4j service for the optional questionnaire |

## Requirements

- Python 3.10 or newer
- A physical Furhat with the Realtime API enabled
- The laptop and Furhat on the same network
- A Cohere API key with access to Command A Vision
- Participant consent for sending one camera image to Cohere

The virtual Furhat SDK does not expose a camera, so the complete visual opening
must be rehearsed on the physical robot.

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a private environment file:

```sh
cp .env.example .env
# Edit .env and replace the placeholders.
set -a
source .env
set +a
```

The relevant values are:

```dotenv
COHERE_API_KEY=your-real-key
COHERE_VISION_MODEL=command-a-vision-07-2025
```

`.env` is ignored by Git. Never put the real key in Python, screenshots, chat,
or commits.

## Run the complete interaction

Choose only a profile whose facts the participant actually disclosed:

```sh
python lab1.py --host 192.168.1.114 --profile games
```

Available profiles are `games` and `reading`. Replace the example address with
the IP shown by the robot. The operator terminal prints whether Cohere produced
a spoken observation or whether the safe fallback suppressed it.

For robot or network troubleshooting, the study dialogue can run without the
visual opening:

```sh
python lab1.py --host 192.168.1.114 --profile games --skip-vision
```

This fallback does not demonstrate the camera requirement.

## API usage

The opening uses one Cohere request per interaction. It sends one low-detail
JPEG and limits the model response to 120 tokens. The request has a 12-second
timeout and no automatic retries. If it fails, the robot stays silent about
appearance rather than inventing a description.

## Tests

```sh
python -m unittest -v
```

The tests use a fake Cohere client, spend no API credits, and require neither a
camera nor example photographs. They cover valid general descriptions, missing
people, malformed camera data, API failure, overly long output, and sensitive
descriptions. Passing tests do not replace a real-robot rehearsal.

## Live rehearsal checklist

- Test several obvious details: different clothing colours, glasses, a hat, and
  a held object. Confirm the output is not hard-coded to a red hat.
- Test an empty scene and a partially visible participant; Furhat should make no
  claim when the image is unreliable.
- Test more than one person and keep bystanders outside the final study frame.
- Confirm no sensitive characteristic or psychological interpretation is said.
- Disconnect the internet and confirm the dialogue continues without a visual
  remark and the terminal reports `source=fallback`.
- Rehearse a prompt answer, a pause longer than three seconds, and silence.
- Confirm Ctrl+C disconnects cleanly.

## Optional Neo4j questionnaire

`test-2.py` asks three non-sensitive questions and stores the answers as:

```text
(Participant)-[:GAVE]->(Answer)-[:ABOUT]->(Topic)
```

Start the local database and run the voice questionnaire:

```sh
docker compose up -d
set -a
source .env
set +a
python test-2.py --participant-id demo01 --mode voice --host 192.168.1.114
```

Use pseudonymous IDs. `--dry-run` avoids database writes, and `--mode terminal`
checks the questionnaire without Furhat. These answers are disclosed facts and
must not be labelled as visual observations.

## Privacy and study use

Obtain consent before activating the visual opening or recording a demo. Tell
participants that one camera frame is sent to Cohere, explain the permitted
description categories, and allow them to opt out by using `--skip-vision`.
Speech may also be processed by Furhat's configured speech recognizer. Allow
participants to skip questions, correct the robot, stop the interaction, and
withdraw their data. Any future expansion of visual inference requires a new
review of the consent text and study protocol.
