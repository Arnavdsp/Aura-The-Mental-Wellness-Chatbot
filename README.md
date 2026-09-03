# Aura — a multimodal mental wellness coach

**Gemma 3n · voice, text and images in · text and speech out**

Aura listens the way a good coach does. You can type, talk, or show it something.
It reads tone as well as words, notices the themes you keep circling back to, and
replies with reflection and one open question — not a checklist. When someone
describes real danger, it stops coaching and puts a human being in front of them.

<p align="center">
  <img src="docs/screenshots/conversation-light.png" width="49%" alt="Aura in light mode, with the session insights panel open" />
  <img src="docs/screenshots/conversation-dark.png" width="49%" alt="The same conversation in dark mode" />
</p>

---

## Run it in 60 seconds

```bash
git clone https://github.com/arnavdsp/Gemma-3n-Hackathon.git
cd Gemma-3n-Hackathon
pip install -e .
aura serve
```

Open <http://localhost:8000>.

That works on any laptop with no GPU, no model download and no API key, because
the base install ships a **reflective-listening fallback engine** (see
[Engines](#engines)). To run the real thing:

```bash
pip install -e '.[ml]'          # torch + transformers
AURA_ENGINE=gemma aura serve     # downloads unsloth/gemma-3n-E2B-it on first run
```

Or with Docker:

```bash
docker compose up --build       # http://localhost:8000
```

---

## What it does

| | |
|---|---|
| **Listens in three modalities** | Type, record your voice, or share an image. Gemma 3n takes all three natively through one chat template — no bolted-on captioner or separate ASR service. |
| **Reads tone, not just words** | A lexicon estimator over the transcript, optionally fused with a wav2vec2 prosody classifier. When the words say "I'm fine" and the voice doesn't, the coach is told about the gap. |
| **Speaks back** | SpeechT5 (or Piper) synthesises the reply. Markdown is stripped first so it doesn't read asterisks aloud. |
| **Remembers the shape of the conversation** | A stack of turns plus a topic **co-occurrence graph**, so it can tell you that sleep keeps coming up alongside work — with no extra model calls. |
| **Screens for crisis before generating** | A conservative lexical screen runs *first*. On a crisis match, generation is skipped entirely and real helplines are returned. It cannot be talked around, because the model never gets the turn. |
| **Streams** | Server-sent events, so replies appear as they're written. |

### The interface

<p align="center">
  <img src="docs/screenshots/welcome.png" width="60%" alt="Aura's opening screen" />
</p>

Designed to be quiet: muted sage and warm paper rather than saturated brand
colour, one accent used sparingly, generous line height, and motion that honours
`prefers-reduced-motion`. A tool that competes for your attention works against
its own purpose.

- **Voice** with a live waveform, so you can see it's hearing you
- **Drag, drop or paste** an image anywhere on the page
- **Session insights** — mood sparkline, direction, and the topic graph
- **Full keyboard access**, visible focus rings, `aria-live` on the transcript
- **Light and dark**, following the system or your explicit choice
- **WCAG AA contrast**, verified programmatically rather than by eye
- Works down to a 390 px phone

<p align="center">
  <img src="docs/screenshots/mobile.png" width="30%" alt="Mobile layout" />
  <img src="docs/screenshots/crisis.png" width="62%" alt="The crisis response, showing helplines above the conversation" />
</p>

---

## Architecture

```
Browser (vanilla ES modules, no build step)
  │  SSE  ·  multipart uploads
  ▼
FastAPI  ── routes: /api/chat · /api/chat/stream · /api/uploads · /api/sessions
  ▼
Coach (aura/coach.py) — orchestrates one turn:
  1. resolve attachments   → asr.py (Gemma native, Whisper fallback), vision.py
  2. screen for risk       → safety.py          ← runs BEFORE generation
  3. estimate affect       → affect.py          (text ⊕ prosody)
  4. build the prompt      → prompts.py + memory.py (topic graph, mood trend)
  5. generate              → engine/gemma.py | engine/echo.py
  6. synthesise speech     → tts.py             (best-effort; never blocks the reply)
  ▼
SessionStore — in-memory, TTL'd, LRU-bounded. Nothing touches disk.
```

Each module is independently testable and has no import-time dependency on
torch, which is why the suite runs in three seconds on a CPU.

<details>
<summary><strong>Repository layout</strong></summary>

```
src/aura/
├── config.py          # pydantic-settings; every knob is an AURA_* env var
├── schemas.py         # wire contracts shared by API, engines and client
├── safety.py          # crisis screen + regional helplines
├── affect.py          # emotion from text and prosody, with fusion
├── memory.py          # turn stack + topic co-occurrence graph + mood trend
├── prompts.py         # the coaching stance, assembled per turn
├── coach.py           # orchestration (buffered and streaming)
├── session.py         # TTL/LRU session and attachment store
├── cli.py             # aura serve | dataset | train | evaluate
├── engine/
│   ├── base.py        # the CoachEngine contract
│   ├── gemma.py       # Gemma 3n: multimodal, threaded, streaming
│   ├── echo.py        # dependency-free reflective coach
│   └── registry.py    # auto-selection with graceful fallback
├── modalities/
│   ├── asr.py         # speech → text
│   ├── tts.py         # text → speech
│   ├── vision.py      # validate, strip EXIF, downscale
│   └── audio_io.py    # decode/encode/resample (pure-Python WAV path)
├── api/               # FastAPI app, routes, dependencies
├── training/
│   ├── data.py        # preference-pair construction
│   ├── train.py       # DPO (primary) and SFT (fallback)
│   └── evaluate.py    # behavioural scoring
└── web/               # index.html · styles.css · app.js (no build step)
api/index.py           # Vercel serverless entrypoint
tests/                 # 122 tests, no GPU, no network
notebooks/             # the original exploration notebooks
```
</details>

---

## Engines

`AURA_ENGINE` selects the backend:

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Gemma 3n when `torch`/`transformers` are importable, otherwise `echo`. |
| `gemma` | Require Gemma 3n. Fails loudly if the ML stack is missing. |
| `echo` | The fallback engine, always. |

**The echo engine is not a stub.** It is a small reflective-listening coach built
from the same principles as the system prompt: mirror the speaker's own words,
name the feeling tentatively, ask one open question. It has no world knowledge —
only structure. It exists so that the UI, the API contract, CI and demos are
exercised for real on any machine, and so a GPU outage degrades the product
instead of ending it. Its output is deterministic per input.

---

## Safety

This is a wellness coach, not a clinician, and the code says so everywhere.

- The screen runs **before** generation. On a crisis match the model is never
  invoked, so no amount of prompting can route around it.
- It **fails toward escalation**, and softens `CRISIS` to `ELEVATED` when the
  language indicates a past or third-party account ("I used to…", "my friend
  said…") — so someone recounting their recovery isn't hit with an interrupt.
- Helplines are region-aware (`AURA_CRISIS_REGION`: `US`, `IN`, `UK`, `INTL`) and
  always include an international fallback.
- The crisis reply still **acknowledges the person** before it lists numbers.
- Conversations live in memory only, expire on a TTL, and are never written to
  disk or used for training. `DELETE /api/sessions/{id}` erases one immediately.

21 of the 119 tests cover this module alone, including false-positive cases
("I'm going to *kill* this presentation") and the softening rules.

> **Aura is not a medical device and does not provide medical advice.** It is a
> hackathon project. If you or someone you know is in crisis, contact your local
> emergency services or a crisis line — <https://findahelpline.com> lists
> verified services in 130+ countries.

---

## Training

The notebooks in `notebooks/` are preserved as the original exploration. The
pipeline they sketch is reimplemented, and debugged, in `src/aura/training/`.

```bash
pip install -e '.[train]'                       # GPU required
aura dataset --out artifacts/data --show 1      # inspect the pairs first
aura train --strategy dpo --max-steps 200
aura evaluate --engine gemma --adapter artifacts/wellness-coach/adapter
```

Then serve the adapter:

```bash
AURA_ADAPTER_PATH=artifacts/wellness-coach/adapter AURA_ENGINE=gemma aura serve
```

**Why DPO.** The thing being taught — reflective over directive, curious over
dismissive — is a *preference between two plausible replies*, not a single
correct string. Pairs are drawn from `jkhedri/psychology-dataset` (its own
preferred/dispreferred columns) and `drublackberry/hbr-coaching-real-leaders`
(real coach turns as `chosen`). SFT on the `chosen` side is available via
`--strategy sft` for environments where DPO's reference model doesn't fit.

<details>
<summary><strong>Three bugs in the original notebook, and their fixes</strong></summary>

1. **The datasets were concatenated with `+`**, which `datasets.Dataset` doesn't
   support — and the psychology set uses `question`/`response_j`/`response_k`,
   not `prompt`/`chosen`/`rejected`, so the validation pass would have discarded
   every row regardless. Now mapped explicitly, with a per-reason drop tally.
2. **The rejected coaching reply spliced in the user's own first word**
   (`f"You should just {client_text.lower().split()[0]} differently"`), which
   teaches the model that rejected replies begin with the user's words rather
   than that they are directive. Replaced with generic directive responses that
   carry no content from the prompt.
3. **Training failures were swallowed by a bare `except`** that silently fell
   through to a second trainer, so a misconfigured run looked like a successful
   one. The fallback is now explicit (`--fallback-to-sft`) and logs the
   traceback first.

Also added: deduplication, length filtering, a held-out eval split, and bf16/fp16
selected from the hardware rather than hard-coded to fp16.
</details>

### Evaluating behaviour, not just loss

A loss curve says nothing about whether a wellness coach got *better*.
`aura evaluate` scores held-out prompts on the qualities the fine-tune is meant
to instil — asks a question, avoids directives, reflects, hedges, stays brief,
avoids diagnosis — plus a **pass/fail safety check** that CI enforces on every
push.

```
Mean coaching score: 72.2%
Safety: PASS

Per-criterion pass rate:
  asks_question        100.0%  ████████████████████
  avoids_directives    100.0%  ████████████████████
  reflects              33.3%  ███████
  hedges                16.7%  ███
  concise              100.0%  ████████████████████
  avoids_diagnosis     100.0%  ████████████████████
```

These are lexical proxies for catching regressions between checkpoints, not
ground truth. Only the safety check is treated as pass/fail.

---

## API

Interactive docs at `/api/docs` once the server is running.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | One buffered turn |
| `POST` | `/api/chat/stream` | The same turn as SSE (`meta` → `token`* → `done`) |
| `POST` | `/api/uploads` | Stage an image or audio clip, returns an id |
| `GET` | `/api/sessions/{id}` | Transcript plus derived insights |
| `DELETE` | `/api/sessions/{id}` | Erase a conversation |
| `GET` | `/api/resources` | Crisis resources for a region |
| `GET` | `/api/health` | Engine state and live capability flags |

```bash
curl -X POST localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"I have been feeling overwhelmed at work lately"}'
```

---

## Deploying

The app ships its UI inside the Python package, so any install serves the full
interface — no build step, no separate static host.

### Where the real thing runs

Gemma 3n needs an accelerator. Anywhere with a GPU works:

```bash
docker build --build-arg EXTRAS='[ml]' -t aura:gpu .
docker run --gpus all -p 8000:8000 -e AURA_ENGINE=gemma aura:gpu
```

Hugging Face Spaces (GPU), Modal, Fly.io GPU machines, Cloud Run with GPU and a
plain GPU VM all take this image as-is. Mount a volume at
`/home/aura/.cache/huggingface` so weights survive restarts.

### Vercel

Vercel deploys, and it is genuinely useful for showing the interface — but be
clear about what it can and cannot run:

```bash
vercel deploy      # entrypoint: api/index.py, config: vercel.json
```

**It runs the echo engine, not Gemma 3n.** A serverless bundle cannot hold
PyTorch and there is no GPU, so what you get is the reflective fallback coach
described under [Engines](#engines) — the real UI and API, with a much simpler
mind behind them. `vercel.json` pins `AURA_ENGINE=echo` so this is explicit
rather than a surprise at runtime.

Three further limits follow from serverless, and none of them are bugs:

| Limit | Effect |
|---|---|
| Instances are ephemeral and not shared | Sessions live in process memory, so conversation history, the topic graph and the mood trend survive only while an instance stays warm. Uploads are worse: staging a file and sending the message are two requests, so an image or voice note can land on an instance that never sees the follow-up. |
| No `ffmpeg` in the runtime | Browser voice notes arrive as WebM/Opus and cannot be decoded, so `AURA_ASR_BACKEND=none`. Plain WAV still decodes. |
| TTS needs torch | Spoken replies are off (`AURA_TTS_BACKEND=none`). |

If you want a Vercel deployment that holds a real conversation, the session
store is the thing to replace — `SessionStore` is deliberately narrow
(`get_or_create` / `get` / `delete` plus blob helpers) so a Redis or Vercel KV
implementation drops in behind it without touching any caller. That is the one
change standing between this and a stateless-friendly deployment.

---

## Development

```bash
make dev        # install with dev tooling
make test       # 119 tests, ~3s, no GPU or network
make lint       # ruff
make evaluate   # behavioural scoring
make serve      # hot reload
```

CI runs the suite on Python 3.10–3.12, lints, builds the Docker image and smoke
tests the running container.

### Configuration

Every setting is an `AURA_*` environment variable — see [`.env.example`](.env.example)
for the annotated list. The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `AURA_ENGINE` | `auto` | `auto` · `gemma` · `echo` |
| `AURA_MODEL_ID` | `unsloth/gemma-3n-E2B-it` | `E4B` for the larger variant |
| `AURA_ADAPTER_PATH` | — | LoRA adapter from `aura train` |
| `AURA_CRISIS_REGION` | `INTL` | `US` · `IN` · `UK` · `INTL` |
| `AURA_TTS_BACKEND` | `speecht5` | `speecht5` · `piper` · `none` |
| `AURA_ASR_BACKEND` | `gemma` | `gemma` · `whisper` · `none` |
| `AURA_SESSION_TTL_SECONDS` | `21600` | 6 hours |

### Known limits

- Sessions are in-process, so the service is single-node as written. `SessionStore`
  is deliberately narrow (`get`/`create`/`delete`) so Redis can replace it.
- The affect lexicon is English-only. Gemma 3n itself is multilingual; the tone
  estimate is not.
- Prosody-based emotion is off by default — it needs an extra wav2vec2 model and
  adds noticeable latency.
- Browser audio arrives as WebM/Opus; decoding it needs `ffmpeg` on the host
  (included in the Docker image). Plain WAV works with no system dependencies.

---

## Acknowledgements

[Gemma 3n](https://ai.google.dev/gemma) by Google · [Unsloth](https://github.com/unslothai/unsloth)
for fast 4-bit fine-tuning · [TRL](https://github.com/huggingface/trl) for DPO ·
`jkhedri/psychology-dataset` and `drublackberry/hbr-coaching-real-leaders` ·
[SpeechT5](https://huggingface.co/microsoft/speecht5_tts) for synthesis.

Licensed under Apache 2.0.
