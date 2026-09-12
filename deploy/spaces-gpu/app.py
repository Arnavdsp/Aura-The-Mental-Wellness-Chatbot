"""Aura on ZeroGPU — a multimodal wellness coach running real Gemma 3n.

The interesting property of this app is *where the work happens*. Only token
generation runs on the GPU; the coaching judgement runs in the main process:

    main process              GPU worker (@spaces.GPU fork)
    ─────────────────────     ─────────────────────────────
    crisis screen             model.generate(...)
    affect estimate
    memory / topic graph
    system-prompt assembly
    ↑ record the reply

That split is not an optimisation. The crisis screen runs **before** anything
reaches the model, and on a crisis match the model is never invoked at all — so
no prompt can talk around it, because there is no prompt. Keeping the screen out
of the GPU function is what makes that guarantee structural.

Everything under `aura/` is the same code the FastAPI service runs:
https://github.com/Arnavdsp/Gemma-3n-Hackathon
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from threading import Thread

import spaces  # noqa: F401  — MUST precede torch; patches torch.cuda for ZeroGPU
import torch
import gradio as gr
from transformers import AutoModelForImageTextToText, AutoProcessor, BatchFeature
from transformers.generation.streamers import TextIteratorStreamer

from aura.affect import analyse_text, describe
from aura.memory import ConversationMemory, alternating_history
from aura.prompts import build_system_prompt, suggestions_for
from aura.safety import crisis_message, resources_for, screen
from aura.schemas import AffectSignal, RiskLevel, Role, SafetyAssessment, Turn
from aura.session import new_id

# The Unsloth mirror is ungated, so the Space needs no token and no licence
# approval. It is the same weights as google/gemma-3n-E4B-it.
MODEL_ID = os.getenv("AURA_MODEL_ID", "unsloth/gemma-3n-E4B-it")
REGION = os.getenv("AURA_CRISIS_REGION", "INTL")
MAX_NEW_TOKENS = int(os.getenv("AURA_MAX_NEW_TOKENS", "320"))
MAX_INPUT_TOKENS = int(os.getenv("AURA_MAX_INPUT_TOKENS", "8192"))
HISTORY_TURNS = int(os.getenv("AURA_MAX_TURNS_IN_CONTEXT", "12"))

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
AUDIO_TYPES = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16
)


# ── Generation: the only part that needs a GPU ────────────────────────────────


# Measured, not guessed: the slowest observed turn on this Space was 14.8s (an
# image on a cold worker); warm text turns run 5-7s. The skill's rule of thumb is
# `measured_max x 1.4`, rounded up here for cold-start variance.
#
# The number matters twice over. ZeroGPU compares *requested* duration against
# remaining quota, not actual usage, so an over-declared 90s blocked every call
# once fewer than 90 seconds were left — while each call was really costing six.
# Shorter requests also rank higher in the queue.
@spaces.GPU(duration=30)
@torch.inference_mode()
def _generate_on_gpu(inputs: BatchFeature, max_new_tokens: int) -> Iterator[str]:
    """Stream a reply. Runs in the forked GPU worker; returns plain strings."""
    inputs = inputs.to(device=model.device, dtype=torch.bfloat16)
    streamer = TextIteratorStreamer(
        processor, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    failures: list[Exception] = []

    def run() -> None:
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=64,
                repetition_penalty=1.05,
                disable_compile=True,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the user below
            failures.append(exc)

    thread = Thread(target=run)
    thread.start()

    chunks: list[str] = []
    for piece in streamer:
        chunks.append(piece)
        yield "".join(chunks)
    thread.join()

    if failures:
        raise gr.Error(f"Generation failed: {failures[0]}")


# ── Turn assembly ─────────────────────────────────────────────────────────────


def _kind(path: str) -> str | None:
    lowered = path.lower()
    if lowered.endswith(IMAGE_TYPES):
        return "image"
    if lowered.endswith(AUDIO_TYPES):
        return "audio"
    return None


def _user_content(text: str, files: list[str]) -> tuple[list[dict], set[str]]:
    """Gemma 3n takes text, images and audio as typed parts of one message."""
    parts: list[dict] = []
    modalities: set[str] = set()
    for path in files:
        kind = _kind(path)
        if kind is None:
            gr.Warning(f"I can't read {os.path.basename(path)} — images and audio only.")
            continue
        parts.append({"type": kind, kind: path})
        modalities.add(kind)
    if text:
        parts.append({"type": "text", "text": text})
        modalities.add("text")
    if not parts:
        parts.append({"type": "text", "text": "(no message)"})
    return parts, modalities


def _history_messages(memory: ConversationMemory) -> list[dict]:
    """Past turns in the typed-part shape Gemma's template expects.

    The ordering rules live in `aura.memory.alternating_history` so they are
    covered by the repo's test suite rather than only by this Space.
    """
    return [
        {"role": role, "content": [{"type": "text", "text": text}]}
        for role, text in alternating_history(memory.turns, HISTORY_TURNS)
    ]


def _insights(memory: ConversationMemory, affect: AffectSignal, safety: SafetyAssessment) -> str:
    lines = [f"**Reading right now** — {describe(affect)}"]
    if affect.source != "none":
        lines.append(
            f"valence `{affect.valence:+.2f}` · arousal `{affect.arousal:.2f}` · "
            f"confidence `{affect.confidence:.0%}`"
        )

    recurring = memory.graph.recurring()
    dominant = memory.graph.dominant()
    if dominant:
        lines.append("")
        lines.append("**Themes** — " + ", ".join(f"`{t}`" for t in dominant))
        for topic in dominant[:1]:
            linked = memory.graph.linked_to(topic)
            if linked:
                pairs = ", ".join(f"`{other}` ×{w}" for other, w in linked)
                lines.append(f"`{topic}` keeps coming up alongside {pairs}")
    if recurring:
        lines.append(f"Returned to: {', '.join(f'`{t}`' for t in recurring)}")

    if len(memory.mood_trend) > 1:
        spark = "".join("▁▂▃▅▆▇"[min(5, int((v + 1) / 2 * 6))] for v in memory.mood_trend[-16:])
        lines.append("")
        lines.append(f"**Mood** `{spark}` — {memory.mood_direction()}")

    if safety.risk is not RiskLevel.NONE:
        lines.append("")
        lines.append(f"**Safety** — risk `{safety.risk.value}`")

    suggestions = suggestions_for(affect, safety.risk)
    if suggestions:
        lines.append("")
        lines.append("**You might say** — " + " · ".join(f"*{s}*" for s in suggestions))
    return "\n".join(lines)


def _crisis_panel(safety: SafetyAssessment) -> str:
    if safety.risk is RiskLevel.NONE:
        return ""
    rows = [
        f"- **{r.name}** — {r.contact}" + (f" · [{r.url}]({r.url})" if r.url else "")
        for r in (safety.resources or resources_for(REGION))
    ]
    return "### If you need someone now\n" + "\n".join(rows)


# ── The turn ──────────────────────────────────────────────────────────────────


def respond(
    message: dict, history: list[dict], memory: ConversationMemory | None
) -> Iterator[tuple]:
    """One turn, orchestrated in the main process."""
    memory = memory or ConversationMemory(session_id=new_id("ses"))
    text = (message.get("text") or "").strip()
    files = list(message.get("files") or [])

    if not text and not files:
        gr.Warning("Tell me something, or share an image or a voice note.")
        return

    parts, modalities = _user_content(text, files)

    # 1. Screen BEFORE generating. This is the whole safety design.
    safety = screen(text, region=REGION)
    affect = analyse_text(text)

    user_turn = Turn(id=new_id("turn"), role=Role.USER, text=text, affect=affect, safety=safety)
    memory.add(user_turn)

    shown = history + [{"role": "user", "content": text or "(shared a file)"}]

    if safety.should_short_circuit:
        reply = crisis_message(safety)
        memory.add(Turn(id=new_id("turn"), role=Role.ASSISTANT, text=reply))
        yield (
            shown + [{"role": "assistant", "content": reply}],
            memory,
            _insights(memory, affect, safety),
            gr.update(value=_crisis_panel(safety), visible=True),
        )
        return

    # 2. Otherwise: assemble the prompt from stance + affect + memory, and generate.
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": build_system_prompt(
                        memory, affect, modalities=modalities, risk=safety.risk
                    ),
                }
            ],
        },
        *_history_messages(memory),
        {"role": "user", "content": parts},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise gr.Error("This conversation has grown past the context window — start a new one.")

    insights = _insights(memory, affect, safety)
    crisis = gr.update(value=_crisis_panel(safety), visible=safety.risk is not RiskLevel.NONE)

    reply = ""
    for reply in _generate_on_gpu(inputs=inputs, max_new_tokens=MAX_NEW_TOKENS):
        yield shown + [{"role": "assistant", "content": reply}], memory, insights, crisis

    # 3. Record the reply so the next turn has it.
    memory.add(Turn(id=new_id("turn"), role=Role.ASSISTANT, text=reply.strip()))
    yield (
        shown + [{"role": "assistant", "content": reply}],
        memory,
        _insights(memory, affect, safety),
        crisis,
    )


def reset() -> tuple:
    return [], ConversationMemory(session_id=new_id("ses")), _BLANK_INSIGHTS, gr.update(visible=False)


_BLANK_INSIGHTS = (
    "Aura reads tone as well as words, and tracks the themes you return to.\n\n"
    "What it notices will appear here as you talk."
)

CSS = """
.aura-title { text-align: center; }
.aura-title h1 { margin-bottom: 4px; font-weight: 600; }
.aura-sub { text-align: center; opacity: .72; margin-top: 0; font-size: 15px; }
#crisis { border-left: 4px solid #b3261e; padding-left: 14px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Aura — Multimodal Wellness Coach", fill_height=True) as demo:
    memory_state = gr.State(None)

    gr.Markdown("# Aura", elem_classes="aura-title")
    gr.Markdown(
        "A multimodal wellness coach on **Gemma 3n**. Type, speak, or show it "
        "something. Not a medical device — and not a substitute for a person.",
        elem_classes="aura-sub",
    )

    with gr.Row():
        with gr.Column(scale=3):
            crisis_panel = gr.Markdown(visible=False, elem_id="crisis")
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                avatar_images=(None, None),
                placeholder="<p style='text-align:center;opacity:.6'>"
                "What's been on your mind?</p>",
            )
            composer = gr.MultimodalTextbox(
                file_types=list(IMAGE_TYPES + AUDIO_TYPES),
                file_count="multiple",
                sources=["upload", "microphone"],
                placeholder="Type, attach an image, or record a voice note…",
                show_label=False,
                autofocus=True,
            )
            with gr.Row():
                clear = gr.Button("New conversation", variant="secondary", size="sm")

        with gr.Column(scale=2):
            insights = gr.Markdown(_BLANK_INSIGHTS, label="Session insights")

    gr.Examples(
        examples=[
            {"text": "I've been feeling really anxious about my job performance lately.", "files": []},
            {"text": "I had a good week for once and I'm scared it won't last.", "files": []},
            {"text": "My partner says I've been distant and I don't know how to explain it.", "files": []},
            {"text": "I can't seem to stick to any routine I set for myself.", "files": []},
        ],
        inputs=composer,
        label="Try one",
    )

    outputs = [chatbot, memory_state, insights, crisis_panel]
    composer.submit(respond, [composer, chatbot, memory_state], outputs).then(
        lambda: gr.update(value=None), None, composer
    )
    clear.click(reset, None, outputs)

if __name__ == "__main__":
    demo.launch(css=CSS, show_error=True)
