"""Dependency-free fallback engine.

This is not a mock that returns "hello world". It is a small reflective-listening
coach built from the same principles the system prompt describes: mirror what
was said, name the feeling tentatively, ask one open question. That makes the
app genuinely usable for demos, UI work, CI and machines without a GPU, and it
keeps the API contract honest — every code path around it is exercised for real.

It is deliberately not a substitute for Gemma 3n: it has no world knowledge and
no memory of meaning, only structure.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections.abc import AsyncIterator

from aura.affect import analyse_text
from aura.config import Settings
from aura.engine.base import CoachEngine, GenerationRequest

_OPENERS = {
    "overwhelmed": [
        "That sounds like a lot to be carrying at once.",
        "It sounds like everything is arriving at the same time.",
    ],
    "anxious": [
        "There's a lot of tension in what you're describing.",
        "It sounds like your mind hasn't been able to put this down.",
    ],
    "sad": [
        "That sounds genuinely heavy.",
        "I hear how much this is weighing on you.",
    ],
    "lonely": [
        "That sounds isolating.",
        "It sounds like you've been holding this by yourself.",
    ],
    "angry": [
        "There's real frustration in that.",
        "That sounds like it's been building for a while.",
    ],
    "tired": [
        "You sound worn down.",
        "It sounds like there hasn't been much room to rest.",
    ],
    "joy": [
        "That's genuinely good to hear.",
        "There's real lightness in how you're describing this.",
    ],
    "calm": [
        "That sounds steadier than where you've been.",
        "There's a settledness in that.",
    ],
    "neutral": [
        "Thank you for putting that into words.",
        "I'm listening.",
    ],
}

_QUESTIONS = {
    "overwhelmed": [
        "If you could set one of those things down for a week, which would it be?",
        "What's the piece that feels heaviest right now?",
    ],
    "anxious": [
        "What does the worry tell you is going to happen?",
        "When was the last time this felt even slightly quieter?",
    ],
    "sad": [
        "What would you want someone to understand about this?",
        "How long has it felt this way?",
    ],
    "lonely": [
        "Who used to feel close, before this?",
        "What would connection look like this week, even in a small way?",
    ],
    "angry": [
        "What feels unfair about it?",
        "What would you want to be different?",
    ],
    "tired": [
        "What's been taking the most out of you?",
        "What would actual rest look like for you?",
    ],
    "joy": [
        "What made the difference this time?",
        "How do you want to hold onto this?",
    ],
    "calm": [
        "What would you like to use this steadiness for?",
        "What's been helping?",
    ],
    "neutral": [
        "What feels most important to say about it?",
        "Where would you like to start?",
    ],
}

_TENTATIVE = [
    "I might be reading this wrong, but it sounds {mood}.",
    "If I'm hearing you right, there's something {mood} in this.",
    "It sounds — tell me if this is off — a little {mood}.",
]

_MOOD_WORDS = {
    "overwhelmed": "overwhelming", "anxious": "anxious", "sad": "painful",
    "lonely": "lonely", "angry": "frustrating", "tired": "exhausting",
    "joy": "hopeful", "calm": "settled", "neutral": "unresolved",
}

# Written as prose rather than a 60-element list literal, for readability.
_STOPWORD_TEXT = """
i me my myself we our you your he she it they them the a an and or but if then
than so because as of at by for with about into over after is am are was were be
been being do does did doing have has had having will would can could should
just really very much more most that this these those to in on not no yes get
got go going feel feeling felt like know think want
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


class EchoEngine(CoachEngine):
    name = "echo"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def ready(self) -> bool:
        return True

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"text": True, "vision": False, "audio_in": False, "streaming": True}

    async def warmup(self) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> str:
        return self._compose(request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Emit word by word so the client's streaming path is exercised."""
        text = self._compose(request)
        for token in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0.012)
            yield token

    async def transcribe(self, audio: bytes, media_type: str) -> str:
        raise NotImplementedError("the echo engine cannot transcribe audio")

    # -- composition -----------------------------------------------------

    def _compose(self, request: GenerationRequest) -> str:
        text = request.user_text.strip()
        # Seed per request, not per instance: the same input must always give
        # the same reply, whether it arrives through generate() or stream().
        rng = random.Random(hashlib.sha256(text.encode()).hexdigest()[:16])
        if not text and request.images:
            return (
                "Thanks for sharing that image with me. I can't see it clearly enough "
                "to say anything useful about it — would you tell me what it means to "
                "you?"
            )
        if not text:
            return "I'm here whenever you're ready. What's on your mind?"

        affect = analyse_text(text)
        mood = affect.label if affect.confidence > 0.3 else "neutral"
        bucket = mood if mood in _OPENERS else "neutral"

        lines = [rng.choice(_OPENERS[bucket])]

        theme = self._salient_phrase(text)
        if theme and " " in theme:  # a single word mirrors back as parroting
            lines.append(f"What stands out is what you said about {theme}.")

        if affect.confidence > 0.4:
            lines.append(
                rng.choice(_TENTATIVE).format(mood=_MOOD_WORDS.get(bucket, "difficult"))
            )

        if len(request.history) >= 4:
            lines.append("We've covered some ground already — I don't want to rush past it.")

        lines.append(rng.choice(_QUESTIONS[bucket]))
        return " ".join(lines)

    def _salient_phrase(self, text: str) -> str | None:
        """The longest run of content words, used to mirror the speaker's own words.

        Mirroring only works if the phrase is theirs verbatim, so this takes a
        contiguous span from the original text rather than assembling one.
        """
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        best: list[str] = []
        run: list[str] = []
        for token in tokens:
            if token.lower() in _STOPWORDS or len(token) <= 3 or "'" in token:
                if len(run) > len(best):
                    best = run
                run = []
            else:
                run.append(token)
        if len(run) > len(best):
            best = run
        if not best:
            return None
        return " ".join(best[:3]).lower()
