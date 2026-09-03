"""Prompt construction.

The coaching stance encoded here is the product: non-directive, reflective, and
honest about being a machine. It mirrors what the DPO dataset in
``aura.training.data`` rewards — empathy over dismissal, questions over orders —
so the prompt and the fine-tune pull in the same direction.
"""

from __future__ import annotations

from aura.memory import ConversationMemory
from aura.schemas import AffectSignal, RiskLevel

SYSTEM_PROMPT = """\
You are Aura, a warm and steady wellness coach. You are not a therapist, a \
doctor, or a crisis service, and you never pretend otherwise.

How you talk:
- Reflect back what you heard before you respond to it, in your own words, so \
the person knows they landed.
- Ask one open question at a time. Never stack questions.
- Offer perspective, not instructions. "Some people find..." beats "You should...".
- Keep replies to 3-6 sentences unless they ask for more. Silence and space are \
part of coaching.
- Match their energy. Someone flat does not want exclamation marks; someone \
panicking needs short, concrete sentences.
- Name feelings tentatively ("it sounds like...", "I might be off, but..."), \
never as fact.

What you do not do:
- Do not diagnose, do not name conditions, do not discuss medication.
- Do not promise outcomes or say you understand exactly how they feel.
- Do not perform empathy with stock phrases. If you have nothing to add, say so \
plainly and ask what would help.
- Do not moralise, and do not rush anyone toward a positive reframe.

If someone describes danger to themselves or others, drop the coaching frame \
immediately: say clearly that this needs a real person, give the crisis \
resources, and stay present with them.
"""

MODALITY_NOTES = {
    "audio": (
        "This message arrived as voice. The transcript may contain recognition "
        "errors — read past them, and take tone cues from the affect note rather "
        "than from punctuation."
    ),
    "image": (
        "The person shared an image. Respond to what it appears to mean to them, "
        "not just to its contents, and check your reading of it rather than "
        "asserting it."
    ),
}


def _affect_note(affect: AffectSignal) -> str:
    from aura.affect import describe

    gloss = describe(affect)
    if gloss == "unclear":
        return (
            "Emotional read: unclear. Do not guess at a feeling — ask instead."
        )
    source = {
        "text": "from their words",
        "speech": "from their tone of voice",
        "fused": "from both their words and their tone",
        "none": "",
    }.get(affect.source, "")
    note = f"Emotional read ({source}): {gloss}."
    if affect.source in {"speech", "fused"} and affect.confidence > 0.5:
        note += (
            " If their words and their voice disagree, gently name the gap rather "
            "than taking the words at face value."
        )
    return note


def build_system_prompt(
    memory: ConversationMemory | None,
    affect: AffectSignal,
    *,
    modalities: set[str] | None = None,
    risk: RiskLevel = RiskLevel.NONE,
) -> str:
    """Assemble the system prompt for one turn."""
    sections = [SYSTEM_PROMPT.strip(), _affect_note(affect)]

    for modality in sorted(modalities or set()):
        note = MODALITY_NOTES.get(modality)
        if note:
            sections.append(note)

    if memory:
        context = memory.context_note()
        if context:
            sections.append("Conversation so far — " + context)

    if risk is RiskLevel.ELEVATED:
        sections.append(
            "This message contains signals of significant distress. Slow down, "
            "stay concrete, and make sure they know professional support exists "
            "without lecturing them about it."
        )

    return "\n\n".join(section for section in sections if section)


# Follow-up chips offered in the UI. Chosen by affect so they never feel canned.
_SUGGESTIONS_BY_LABEL: dict[str, list[str]] = {
    "overwhelmed": [
        "Help me pick just one thing to focus on",
        "What would 10 minutes of relief look like?",
        "I want to say all of it out loud",
    ],
    "anxious": [
        "Walk me through a grounding exercise",
        "Help me separate what's real from what I'm imagining",
        "What's the smallest next step?",
    ],
    "sad": [
        "I'd rather just be heard right now",
        "Help me understand where this is coming from",
        "What has helped before?",
    ],
    "lonely": [
        "Who could I reach out to this week?",
        "I want to talk about why this feels hard",
        "Help me be kinder to myself about it",
    ],
    "angry": [
        "Help me figure out what's underneath this",
        "I need to vent first",
        "What would I want to say if I were calm?",
    ],
    "tired": [
        "Help me protect my rest this week",
        "What's draining me the most?",
        "I want a smaller plan, not a better one",
    ],
    "joy": [
        "Help me hold onto this",
        "What made today different?",
        "I want to build on this",
    ],
    "calm": [
        "What would I like to work on next?",
        "Help me reflect on this week",
        "I want to set an intention",
    ],
}

_DEFAULT_SUGGESTIONS = [
    "Tell me more about what's on your mind",
    "Help me make sense of this",
    "What should I be asking myself?",
]


def suggestions_for(affect: AffectSignal, risk: RiskLevel) -> list[str]:
    """Follow-up prompts surfaced as chips under the reply."""
    if risk is RiskLevel.CRISIS:
        return [
            "I'd like to talk to someone right now",
            "Stay with me for a bit",
            "Help me tell someone what's happening",
        ]
    return _SUGGESTIONS_BY_LABEL.get(affect.label, _DEFAULT_SUGGESTIONS)
