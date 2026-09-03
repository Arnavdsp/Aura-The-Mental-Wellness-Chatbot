from __future__ import annotations

from aura.affect import analyse_text
from aura.memory import ConversationMemory, TopicGraph, extract_topics
from aura.schemas import RiskLevel, Role, SafetyAssessment, Turn


def _user(text: str) -> Turn:
    return Turn(id="t", role=Role.USER, text=text, affect=analyse_text(text))


def test_topic_extraction() -> None:
    topics = extract_topics("My boss set another deadline and I can't sleep")
    assert "work" in topics
    assert "sleep" in topics
    assert extract_topics("") == []


def test_graph_records_cooccurrence() -> None:
    graph = TopicGraph()
    graph.observe(["work", "sleep"])
    graph.observe(["work", "sleep"])
    graph.observe(["work", "money"])
    assert graph.dominant(1) == ["work"]
    assert graph.linked_to("work")[0] == ("sleep", 2)
    assert graph.recurring(min_count=2) == ["work", "sleep"]


def test_graph_serialises_for_the_ui() -> None:
    graph = TopicGraph()
    graph.observe(["work", "sleep"])
    payload = graph.as_dict()
    assert payload["topics"]["work"] == 1
    assert payload["links"][0]["weight"] == 1


def test_memory_tracks_mood_direction() -> None:
    memory = ConversationMemory(session_id="s")
    for text in [
        "I feel hopeless and exhausted",
        "Everything is overwhelming",
        "Today was a little calmer",
        "I actually felt grateful and happy today",
    ]:
        memory.add(_user(text))
    assert memory.mood_direction() == "improving"


def test_mood_direction_is_unknown_before_enough_evidence() -> None:
    memory = ConversationMemory(session_id="s")
    memory.add(_user("I'm anxious"))
    assert memory.mood_direction() == "unknown"


def test_context_note_summarises_recurring_themes() -> None:
    memory = ConversationMemory(session_id="s")
    for text in ["Work is crushing me and I can't sleep",
                 "My manager again, and I'm exhausted",
                 "The deadline plus no rest is wrecking me"]:
        memory.add(_user(text))
    note = memory.context_note()
    assert "work" in note
    assert "sleep" in note


def test_highest_risk_is_sticky() -> None:
    memory = ConversationMemory(session_id="s")
    turn = _user("I feel hopeless")
    turn.safety = SafetyAssessment(risk=RiskLevel.ELEVATED)
    memory.add(turn)
    memory.add(_user("Anyway, work is fine"))
    assert memory.highest_risk is RiskLevel.ELEVATED
    assert "serious" in memory.context_note()


def test_recent_limits_context_window() -> None:
    memory = ConversationMemory(session_id="s")
    for index in range(10):
        memory.add(_user(f"message {index}"))
    assert len(memory.recent(4)) == 4
    assert memory.recent(4)[-1].text == "message 9"
