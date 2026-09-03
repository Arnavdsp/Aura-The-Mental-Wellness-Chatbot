"""Conversation memory.

The README's original design called for a stack of turns plus a graph that maps
how topics connect. That is what this is:

* ``turns``      — the ordered transcript (the stack), trimmed for context.
* ``TopicGraph`` — nodes are themes the person keeps returning to, edges are
  co-occurrences within a turn. It gives us "you've mentioned sleep alongside
  work three times this week" without any extra model calls.
* ``mood_trend`` — valence over time, so the coach can notice direction, not
  just the current point.

Everything here is pure data manipulation: no I/O, no model, trivially testable.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aura.schemas import AffectSignal, RiskLevel, Turn

# Themes worth tracking in a wellness context, each with its trigger words.
TOPIC_LEXICON: dict[str, tuple[str, ...]] = {
    "work": ("work", "job", "boss", "manager", "career", "office", "deadline",
             "promotion", "colleague", "meeting", "project", "workload"),
    "study": ("exam", "study", "studying", "college", "university", "school",
              "assignment", "grades", "thesis", "semester"),
    "sleep": ("sleep", "sleeping", "insomnia", "tired", "exhausted", "rest",
              "awake", "nightmare", "bed"),
    "relationships": ("partner", "wife", "husband", "girlfriend", "boyfriend",
                      "friend", "friends", "family", "mother", "father", "mom",
                      "dad", "parents", "relationship", "breakup", "divorce"),
    "health": ("health", "sick", "illness", "pain", "doctor", "medication",
               "therapy", "therapist", "diagnosis", "body"),
    "money": ("money", "rent", "debt", "bills", "salary", "afford", "financial",
              "savings", "loan"),
    "self_worth": ("failure", "worthless", "not good enough", "useless",
                   "hate myself", "confidence", "self-esteem", "imposter"),
    "loneliness": ("lonely", "alone", "isolated", "no one", "nobody",
                   "disconnected"),
    "habits": ("exercise", "gym", "routine", "habit", "diet", "eating",
               "drinking", "smoking", "screen time", "motivation"),
    "future": ("future", "purpose", "meaning", "direction", "goals", "plan",
               "decision", "choice", "stuck"),
}

_TOPIC_PATTERNS = {
    topic: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    for topic, words in TOPIC_LEXICON.items()
}


def extract_topics(text: str) -> list[str]:
    """Topics mentioned in a piece of text, in lexicon order."""
    if not text:
        return []
    return [topic for topic, pattern in _TOPIC_PATTERNS.items() if pattern.search(text)]


@dataclass
class TopicGraph:
    """Weighted undirected co-occurrence graph over conversation themes."""

    counts: Counter[str] = field(default_factory=Counter)
    edges: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    last_seen: dict[str, datetime] = field(default_factory=dict)

    def observe(self, topics: list[str], *, at: datetime | None = None) -> None:
        stamp = at or datetime.now(timezone.utc)
        for topic in topics:
            self.counts[topic] += 1
            self.last_seen[topic] = stamp
        for a, b in itertools.combinations(sorted(set(topics)), 2):
            self.edges[(a, b)] += 1

    def dominant(self, limit: int = 3) -> list[str]:
        return [topic for topic, _ in self.counts.most_common(limit)]

    def linked_to(self, topic: str, limit: int = 2) -> list[tuple[str, int]]:
        """Topics that keep showing up alongside ``topic``, strongest first."""
        neighbours = [
            (b if a == topic else a, weight)
            for (a, b), weight in self.edges.items()
            if topic in (a, b)
        ]
        return sorted(neighbours, key=lambda item: -item[1])[:limit]

    def recurring(self, min_count: int = 2) -> list[str]:
        return [topic for topic, count in self.counts.most_common() if count >= min_count]

    def as_dict(self) -> dict[str, object]:
        return {
            "topics": dict(self.counts),
            "links": [
                {"source": a, "target": b, "weight": w}
                for (a, b), w in sorted(self.edges.items(), key=lambda kv: -kv[1])
            ],
        }


@dataclass
class ConversationMemory:
    """Everything we remember about one conversation."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    graph: TopicGraph = field(default_factory=TopicGraph)
    mood_trend: list[float] = field(default_factory=list)
    highest_risk: RiskLevel = RiskLevel.NONE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.updated_at = datetime.now(timezone.utc)

        if turn.role.value == "user":
            searchable = " ".join(
                [turn.text]
                + [a.transcript or "" for a in turn.attachments]
                + [a.caption or "" for a in turn.attachments]
            )
            self.graph.observe(extract_topics(searchable), at=turn.created_at)
            if turn.affect and turn.affect.source != "none":
                self.mood_trend.append(turn.affect.valence)
            if turn.safety and turn.safety.risk.rank > self.highest_risk.rank:
                self.highest_risk = turn.safety.risk

    def recent(self, limit: int) -> list[Turn]:
        return self.turns[-limit:] if limit > 0 else list(self.turns)

    def mood_direction(self) -> str:
        """"improving" / "declining" / "steady" / "unknown" over recent turns."""
        if len(self.mood_trend) < 3:
            return "unknown"
        window = self.mood_trend[-6:]
        midpoint = len(window) // 2
        earlier = sum(window[:midpoint]) / midpoint
        later = sum(window[midpoint:]) / (len(window) - midpoint)
        delta = later - earlier
        if delta > 0.15:
            return "improving"
        if delta < -0.15:
            return "declining"
        return "steady"

    def context_note(self) -> str:
        """A compact briefing injected into the system prompt each turn.

        Cheap continuity: it lets the coach reference patterns without us
        re-sending the whole transcript on every call.
        """
        parts: list[str] = []
        recurring = self.graph.recurring()
        if recurring:
            parts.append("Recurring themes: " + ", ".join(recurring[:4]) + ".")
            top = recurring[0]
            linked = self.graph.linked_to(top)
            if linked:
                partners = ", ".join(name for name, _ in linked)
                parts.append(f"'{top}' keeps coming up alongside {partners}.")
        direction = self.mood_direction()
        if direction != "unknown":
            parts.append(f"Mood across this conversation appears {direction}.")
        if self.highest_risk.rank >= RiskLevel.ELEVATED.rank:
            parts.append(
                "Earlier in this conversation they raised something serious — stay "
                "attentive and don't be afraid to check in on it gently."
            )
        return " ".join(parts)

    def latest_affect(self) -> AffectSignal:
        for turn in reversed(self.turns):
            if turn.role.value == "user" and turn.affect:
                return turn.affect
        return AffectSignal()
