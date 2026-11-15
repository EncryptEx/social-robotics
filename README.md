# Furhat personalisation study

Prototype for the DD2413 project **“When Does Personalization Become
Creepy?”**. It implements a controlled between-subjects interaction with two
conditions:

- **Factual:** Furhat refers only to preferences explicitly disclosed by the
  participant and asks a related question.
- **Inference:** Furhat uses the same disclosed facts and topic, adds one
  plausible non-sensitive prediction, and clearly frames it as a guess.

Before either condition, Furhat can make one brief camera-based observation.
This is a general description of obvious visible clothing, accessories, or a
held object—not a detector for any particular colour or item.

## What is implemented

### 1. General visual opening

Furhat waits for a participant, attends to the closest detected user, captures
one JPEG frame, and sends it in memory to Cohere Command A Vision. The model
returns structured output containing up to three visible details and an
optional utterance beginning with “I notice…”.

Permitted observations include clothing items and colours, glasses, jewellery,
other accessories, and clearly held objects. The prompt and local checks reject
identity and sensitive or speculative attributes such as age, gender,
ethnicity, nationality, religion, health, emotion, personality, occupation,
wealth, or attractiveness. If the frame or response is unreliable, Furhat says
nothing about the participant and continues safely.

The prompt targets the most prominent person. For the actual study, keep
bystanders out of the frame; automatic selection is not a substitute for
consent.

### 2. RDF-style semantic knowledge graph

The questionnaire answers are converted into direct participant-to-concept
relations. For example:

```text
(:Participant {id: "demo01"})-[:LIKES]->(:Concept {label: "cars"})
(:Participant {id: "demo01"})-[:PREFERS]->(:Concept {label: "quiet settings"})
```

The controlled relationship vocabulary is `LIKES`, `DISLIKES`, `PREFERS`, and
`DOES_ON_WEEKENDS`. There is no intermediate `Answer` node. The original answer
is retained on the relationship as provenance together with
`source="questionnaire"`, `explicit=true`, and the question key. Cohere is
instructed to extract only literal disclosures; invalid relation/question
combinations are rejected locally before Neo4j is written.

### 3. Matched LLM-generated study dialogue

The study topic is hard-coded in `study_dialogue.py` as:

> how the participant prefers to spend relaxing free time with other people

For each interaction, Cohere receives only the participant’s explicit semantic
facts and generates both experimental variants in one request. The two variants
must use the same source facts, similar wording length and tone, and the same
topic. Only the inference variant may add an unstated prediction, using the
wording “I might be wrong, but I would guess … Is that right?”.

Local validation rejects unsupported fact IDs, inference language in the
factual condition, sensitive predictions, excessive length differences, and
malformed questions. Both variants are printed before the robot connects, and
the researcher must type `APPROVE`. This is a mandatory protocol check, not a
claim that arbitrary model output is automatically study-ready.

The randomized assignment remains external to the program: use the study’s
allocation schedule and pass exactly one of `--condition factual` or
`--condition inference`. Do not choose the condition after seeing the generated
text.

## Repository layout

| Path | Purpose |
| --- | --- |
| `vision.py` | Camera-frame validation, Cohere Vision, and visual safety checks |
| `knowledge_graph.py` | Explicit fact extraction and direct Neo4j relations |
| `study_dialogue.py` | Matched factual/inference dialogue generation and validation |
| `lab1.py` | Furhat interaction, consent gate, listening, and timing policy |
| `test-2.py` | Voice or terminal questionnaire |
| `compose.yaml` | Local Neo4j service |
| `test_*.py` | Offline tests using fake Cohere responses |

## Requirements

- Python 3.10 or newer
- Docker for the local Neo4j instance
- A physical Furhat with the Realtime API enabled
- The laptop and Furhat on the same network
- A Cohere API key with Command A and Command A Vision access
- Participant consent for the questionnaire and external camera processing

The virtual Furhat SDK does not expose a camera, so the complete visual opening
must be rehearsed on the physical robot.

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, replacing the placeholders, then load it:

```sh
set -a
source .env
set +a
```

Never commit the real Cohere key. The configured defaults are
`command-a-03-2025` for text and `command-a-vision-07-2025` for images.

## 1. Start Neo4j

```sh
docker compose up -d
```

Neo4j Browser is available at <http://localhost:7474>; Bolt listens only on
`localhost:7687`.

## 2. Collect and normalize disclosed facts

With Furhat:

```sh
python test-2.py --participant-id demo01 --mode voice --host 192.168.1.114
```

Without Furhat, for a quick check:

```sh
python test-2.py --participant-id demo01 --mode terminal --dry-run
```

`--dry-run` still calls Cohere to show the proposed semantic triples but does
not write to Neo4j. Use pseudonymous IDs and let participants skip questions.

Inspect the resulting graph with:

```cypher
MATCH (p:Participant)-[r]->(c:Concept)
RETURN p.id, type(r), c.label, r.question_key, r.source_text;
```

## 3. Run the assigned study condition

Inference condition example:

```sh
python lab1.py \
  --host 192.168.1.114 \
  --participant-id demo01 \
  --condition inference
```

For the factual condition, change only `--condition inference` to
`--condition factual`. The program:

1. loads explicit facts from Neo4j;
2. asks Cohere for the matched pair;
3. prints both variants and requires researcher approval;
4. connects to Furhat and waits for a participant;
5. optionally makes the safe visual observation;
6. asks condition-appropriate permission;
7. says only the assigned variant and listens for a response.

Use `--skip-vision` only when the participant opts out or for troubleshooting.
It does not satisfy the camera demonstration requirement.

## Timing response

A reply beginning within three seconds receives a neutral acknowledgement. A
longer pause or silence makes Furhat back off. The closing line is specific to
the assigned condition, so the factual branch never calls its question a
“guess”. This adaptive ending is separate from the experimental manipulation;
record and report it consistently if it is used during data collection.

## API usage and failure behaviour

A full new-participant flow normally makes:

- one text request to extract semantic facts from all questionnaire answers;
- one text request to generate both matched dialogue variants;
- one vision request for a single low-detail JPEG.

Requests use structured JSON, temperature zero, bounded output, finite
timeouts, and no automatic retries. Vision failure suppresses the visual line.
Fact or dialogue failure stops before the study utterance so the robot cannot
silently substitute invented participant information.

## Tests

```sh
python -m unittest -v
```

Tests use fake clients, spend no Cohere credits, and need neither Neo4j nor a
robot. They cover camera validation, general visible details, sensitive-output
suppression, semantic triple validation, matched-condition validation, and the
pause policy.

## Rehearsal checklist

- Try varied clothing, accessories, glasses, and held objects; verify that the
  observation describes whichever safe details are actually prominent.
- Test an empty or unclear frame; no appearance claim should be spoken.
- Keep only the consenting participant in frame and verify the robot attends to
  the intended person.
- Confirm no sensitive characteristic or psychological interpretation is said.
- Inspect Neo4j and verify direct semantic edges, correct concepts, and source
  provenance.
- Compare the generated factual and inference variants: same facts/topic/tone,
  with exactly one tentative extrapolation only in the inference condition.
- Reject and regenerate any pair that is awkward, unsafe, unsupported, or not
  sufficiently matched; document the review procedure before recruitment.
- Disconnect the internet and verify that vision fails silently while text
  generation fails before the study interaction.
- Rehearse a prompt answer, a pause over three seconds, silence, cancellation,
  and clean disconnection.

## Privacy and protocol notes

Obtain consent before activating the camera or collecting answers. Explain that
one image and questionnaire text are sent to Cohere, and offer a camera opt-out.
Speech may also be processed by Furhat’s configured recognizer. Store only
pseudonymous participant IDs, protect the Neo4j volume because relationship
provenance contains raw answers, and define a deletion/withdrawal procedure.

The implementation follows the proposal’s intended manipulation: the factual
condition does not extrapolate, while the inference condition makes a plausible
unstated assumption and labels it as a guess. Keep the number and timing of
personalized remarks matched, use the same disclosed information in both
conditions, review generated material before sessions, exclude sensitive
inferences, and debrief participants about what the robot knew versus inferred.
