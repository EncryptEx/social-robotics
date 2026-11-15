# Furhat personalisation study

Prototype for the DD2413 project **“When Does Personalization Become Creepy?”**.
Furhat first observes a participant with its camera and, when the evidence is
stable, makes a cautious comment about a red hat. It then runs a short scripted
interaction contrasting a disclosed fact with a personal inference and adapts
its closing response to the participant's hesitation. Cohere checks whether
recognized speech is understandable and relevant before the robot accepts it.

The camera remark happens before the study dialogue. It is intentionally
limited to one reviewed, non-sensitive visual cue; the robot does not send
images to an LLM or attempt to infer identity, age, gender, ethnicity, emotion,
health, or other personal attributes.

## Interaction flow

1. Furhat waits until its built-in user detector sees someone and attends to the
   closest participant.
2. It captures five camera frames and looks for a frontal face.
3. Within the area around the face, it measures connected red pixels. A red-hat
   comment is allowed only if at least 60% of usable frames pass the threshold.
4. If the evidence is stable, Furhat says: “It looks like you might be wearing a
   red hat. That is a nice splash of colour.” Otherwise it makes no claim about
   the participant's appearance.
5. Furhat asks permission to make the scripted personal guess. The operator
   presses Enter only after the participant agrees.
6. Cohere validates each non-empty speech transcript. If it rejects the text,
   Furhat asks once for repetition instead of treating it as a real answer.
7. A reply that starts within three seconds receives a neutral acknowledgement.
   A longer pause or silence makes Furhat back off and end the topic.

All camera frames are processed in memory and discarded immediately. Nothing
is written to disk by the interaction.

## Repository layout

| Path | Purpose |
| --- | --- |
| `lab1.py` | Furhat connection, visual opening, scripted dialogue, and response-timing policy |
| `vision.py` | Camera decoding, face-localised red detection, and multi-frame consensus |
| `transcript_validation.py` | Cohere structured-output validation for speech-to-text |
| `test_lab1.py` | Dialogue-policy tests |
| `test_vision.py` | Synthetic tests for image decoding, spatial filtering, and consensus |
| `test_transcript_validation.py` | Mocked Cohere and fallback-policy tests |
| `test-2.py` | Optional questionnaire that stores disclosed answers in Neo4j |
| `compose.yaml` | Local Neo4j service for the optional questionnaire |

## Requirements

- Python 3.10 or newer
- A physical Furhat robot running a version with the Realtime API
- The laptop and Furhat on the same network
- Realtime API local-network access enabled in Furhat's web interface
- A participant facing the camera in reasonable, even lighting
- A Cohere production API key for transcript validation

The virtual Furhat SDK does not expose a camera, so the visual opening must be
tested on a physical robot. If the Realtime API is protected, create a Realtime
API key in the robot interface and export it as `FURHAT_AUTH_KEY`. This is not
the robot web-interface password and must never be committed.

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the environment template, add your Cohere key, and load it into the shell:

```sh
cp .env.example .env
# Edit .env and replace the placeholder values.
set -a
source .env
set +a
```

The code reads `COHERE_API_KEY` and defaults to `command-a-03-2025`. Set
`COHERE_MODEL` only if the team deliberately validates another compatible
model. `.env` is ignored by Git; never add an API key to Python or commit it.

## Run the complete interaction

Choose only a profile whose facts the participant actually disclosed:

```sh
python lab1.py --host 192.168.1.114 --profile games
```

Available profiles are `games` and `reading`. The IP above is an example; use
the address shown by the robot. The program prints every per-frame score and
the final consensus so the team can explain and tune the perception step.

For camera troubleshooting, run the dialogue without the visual opening:

```sh
python lab1.py --host 192.168.1.114 --profile games --skip-vision
```

`--skip-vision` is a fallback, not the intended assessed demonstration.

To rehearse without spending Cohere credits or when offline, add
`--skip-cohere`. This accepts Furhat's raw speech transcript and should not be
used when demonstrating the LLM validation feature.

## Cohere transcript validation

Each non-empty speech transcript is sent together with the question to Cohere's
Chat V2 API. Structured Output restricts the response to two fields: a boolean
decision and a short reason. The model never rewrites the participant's answer:
accepted text is stored exactly as Furhat recognized it, while rejected text
causes one request to repeat.

The integration limits usage and latency:

- obvious empty or `[inaudible]` results are rejected locally without an API call;
- at most two recognized attempts are validated for any question;
- input is truncated to 1,200 characters and output to 100 tokens;
- API calls time out after eight seconds;
- if Cohere is unavailable, the original transcript is accepted but clearly
  logged as an unvalidated `fallback`, so the live interaction can continue.

The operator terminal prints the decision source and reason, but never the API
key. Review fallback entries before using collected answers in the study.

Verify the key and model before connecting Furhat (this makes one API call):

```sh
python transcript_validation.py \
  --question "What hobby do you enjoy?" \
  --transcript "I enjoy indoor climbing."
```

A working cloud call reports `source=cohere`; `source=fallback` means the
interaction would continue, but the LLM did not validate that transcript.

## Check the detector with a still image

Before the live rehearsal, a local JPEG or PNG can be inspected without saving
or transmitting anything:

```sh
python vision.py path/to/rehearsal-photo.jpg
```

The command reports whether it found a face, whether the red-headwear cue
passed, the measured score, and the configured threshold. Use a frontal pose,
keep the complete hat in frame, and avoid a large red background immediately
behind the participant's head.

## Tests

Run the complete offline suite with:

```sh
python -m unittest -v
```

The image tests generate their own arrays and do not need a camera or example
photos. Cohere tests use a fake client, so they spend no credits and need no API
key. A passing offline suite does not replace a rehearsal on the real robot.

## Live rehearsal checklist

- Put one participant in view and keep bystanders outside the frame.
- Test a red hat from the real interaction distance and inspect all five scores.
- Test without a red hat and confirm that Furhat makes no appearance claim.
- Test a red shirt and confirm that the head-only region prevents a hat claim.
- Rehearse one prompt answer, one pause longer than three seconds, and silence.
- Say one deliberately incoherent answer and confirm Furhat asks once to repeat.
- Disconnect the internet and confirm the operator sees a `fallback` decision.
- Confirm that Ctrl+C stops the interaction and disconnects cleanly.
- If lighting or camera placement changes, repeat both positive and negative
  tests before collecting data.

The constants `RED_SCORE_THRESHOLD` and `MIN_CONSENSUS_RATIO` in `vision.py`
control sensitivity. Change them only after checking multiple positive and
negative examples, and record the final values used for every participant.

## Known limitations

This is a transparent colour-and-location heuristic, not a general clothing
classifier. A red scarf, hair accessory, or red background near the face may
look like headwear; dark or partially hidden red hats may be missed. The spoken
line therefore says “might,” and unstable detections are suppressed. The
system should not be presented as recognizing arbitrary clothing or as making
psychological judgments from appearance.

The response delay is likewise an interaction cue, not proof that a participant
felt uncomfortable. The research outcome should come from the planned CRoSS,
Godspeed, trust, and willingness-to-interact measures, not from this heuristic.

## Optional Neo4j questionnaire

`test-2.py` asks three non-sensitive questions and stores answers as:

```text
(Participant)-[:GAVE]->(Answer)-[:ABOUT]->(Topic)
```

Create `.env` from `.env.example`, choose a local password, and start Neo4j:

```sh
cp .env.example .env
docker compose up -d
set -a
source .env
set +a
python test-2.py --participant-id demo01 --mode voice --host 192.168.1.114
```

Use pseudonymous participant IDs. `--dry-run` previews answers without writing,
and `--mode terminal` tests the questionnaire without Furhat. Questionnaire
answers are disclosed facts; they must not be labelled as visual observations.

## Privacy and study use

Obtain consent before collecting questionnaire answers or recording a demo.
Tell participants that the camera is used for the opening observation and that
speech is processed by Furhat's configured recognizer. With Cohere enabled, the
question and recognized transcript are also sent to Cohere for validation; the
consent form and data-processing description must say so. Allow participants
to skip questions, correct the robot, stop the interaction, and withdraw their
data. Do not silently add face recognition, image storage, cloud vision, or new
appearance-based inferences without updating the consent and study protocol.
