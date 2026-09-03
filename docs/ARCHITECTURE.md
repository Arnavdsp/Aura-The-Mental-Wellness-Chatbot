# Architecture

## One turn, end to end

Everything the system does happens inside `Coach.prepare()` and one of the two
generation methods. The order matters, and it is the same for the buffered and
streaming paths — only step 5 differs, so the two cannot drift apart.

```
1. Resolve attachments      audio → transcript (asr.py)
                            image → validated, EXIF-stripped PNG (vision.py)

2. Screen for risk          safety.py, on the combined text
                            ── CRISIS short-circuits here; the model never runs

3. Estimate affect          affect.py: lexicon over text
                            ⊕ optional wav2vec2 over prosody, confidence-weighted

4. Build the prompt         prompts.py: stance + affect note + modality notes
                            + memory.context_note() (topic graph, mood direction)

5. Generate                 engine/gemma.py (or echo.py), buffered or streamed

6. Synthesise speech        tts.py — best effort; a failure never costs the reply

7. Commit                   both turns appended to ConversationMemory
```

## Why the safety screen runs before generation

The screen is lexical, deterministic and free. Running it *first* means a crisis
message never reaches the model at all, so there is no prompt, jailbreak or
sampling temperature that can route around it. A screen that ran on the *output*
would be an optimisation problem; one that runs on the *input* is a guarantee.

The cost is false positives, which is the right way to be wrong here. The
mitigator rules (past tense, third-party) exist to keep that cost bearable for
people recounting recovery rather than living it.

## Why there are two engines

`GemmaEngine` is the product. `EchoEngine` exists so that:

- the API contract, the SSE protocol, the UI and CI are exercised for real on any
  machine, with no GPU and no weights to download;
- a GPU outage degrades the service instead of ending it;
- contributors can work on the interface without a 4 GB download.

It is a rule-based reflective-listening coach, not a stub returning fixed text.
It has structure but no world knowledge, and its output is deterministic per
input (the RNG is seeded from the message text, not from instance state — an
earlier version seeded per instance, which made `generate()` and `stream()`
disagree on the same input).

## Why the memory is a graph

A transcript answers "what was said". A weighted co-occurrence graph over topics
answers "what keeps coming up, and with what" — which is the question a coach
actually needs. It costs one regex pass per turn and no model calls, and it feeds
both the system prompt (`context_note()`) and the UI's insights panel.

Mood is tracked as a valence series rather than a single current value, because
direction ("this got worse over the last three turns") is more useful than
position.

## Threading model

`transformers` is synchronous and its generation is not re-entrant for a single
model instance. So:

- weights load in `asyncio.to_thread` during a background warmup task, which
  never blocks startup or the health check;
- `generate()` runs in a thread;
- `stream()` runs generation in a dedicated thread and bridges
  `TextIteratorStreamer` onto the event loop through an executor, with a queue
  carrying exceptions so a generation failure surfaces to the client instead of
  hanging;
- a `threading.Lock` serialises generation calls against the shared model.

## State

`SessionStore` is in-memory, TTL'd and LRU-bounded. For a wellness product that
is the right default: conversations are sensitive, so nothing touches disk and
everything expires. The interface is deliberately narrow (`get_or_create`, `get`,
`delete`, plus attachment/audio blobs) so a Redis or Postgres implementation can
replace it without touching a single caller.

The consequence is that the service is single-node as written. That is a
deliberate trade, not an oversight.

## The client

Vanilla ES modules, no build step, no framework. One `state` object, small pure
render functions, one transport function that speaks SSE. The entire client is
readable in a sitting — which is worth more here than the ergonomics a framework
would buy, given how little state there is.

Model output and crisis resources both flow through `renderMarkdown()`, which
HTML-escapes before introducing any tag, so untrusted text cannot inject markup.
