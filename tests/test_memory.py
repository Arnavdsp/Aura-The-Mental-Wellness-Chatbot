from __future__ import annotations

from aura.affect import analyse_text
from aura.memory import (
    ConversationMemory,
    TopicGraph,
    alternating_history,
    extract_topics,
)
from aura.schemas import (
    Attachment,
    Modality,
    RiskLevel,
    Role,
    SafetyAssessment,
    Turn,
)


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


def _exchange(memory: ConversationMemory, index: int) -> None:
    memory.add(Turn(id=f"u{index}", role=Role.USER, text=f"user {index}"))
    memory.add(Turn(id=f"a{index}", role=Role.ASSISTANT, text=f"reply {index}"))


def _alternates(history: list[tuple[str, str]]) -> bool:
    roles = [role for role, _ in history]
    if not roles:
        return True
    return (
        roles[0] == "user"
        and roles[-1] == "assistant"
        and all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    )


def test_history_alternates_once_the_window_starts_sliding() -> None:
    """The regression this exists for.

    With a plain slice the 12-turn window begins on an assistant turn from the
    seventh exchange onward, and Gemma's chat template raises rather than
    repairing it — so the deployed app crashed only in conversations long enough
    to reach that point, and passed every short manual test.
    """
    memory = ConversationMemory(session_id="s")
    for index in range(1, 16):
        memory.add(Turn(id=f"u{index}", role=Role.USER, text=f"user {index}"))
        assert _alternates(alternating_history(memory.turns, 12)), f"exchange {index}"
        memory.add(Turn(id=f"a{index}", role=Role.ASSISTANT, text=f"reply {index}"))


def test_history_excludes_the_turn_being_sent() -> None:
    memory = ConversationMemory(session_id="s")
    _exchange(memory, 1)
    memory.add(Turn(id="u2", role=Role.USER, text="user 2"))
    assert alternating_history(memory.turns, 12) == [("user", "user 1"), ("assistant", "reply 1")]


def test_history_skips_a_turn_with_no_text_without_breaking_alternation() -> None:
    """An image with no caption yet has empty text. Dropping it alone would put
    two assistant turns together."""
    memory = ConversationMemory(session_id="s")
    memory.add(Turn(id="u1", role=Role.USER, text="",
                    attachments=[Attachment(kind=Modality.IMAGE, media_type="image/png")]))
    memory.add(Turn(id="a1", role=Role.ASSISTANT, text="I can see it."))
    memory.add(Turn(id="u2", role=Role.USER, text="what do you make of it?"))
    assert _alternates(alternating_history(memory.turns, 12))


def test_history_folds_an_attachment_transcript_into_the_text() -> None:
    memory = ConversationMemory(session_id="s")
    memory.add(Turn(id="u1", role=Role.USER, text="listen to this",
                    attachments=[Attachment(kind=Modality.AUDIO, media_type="audio/wav",
                                            transcript="I can't sleep")]))
    memory.add(Turn(id="a1", role=Role.ASSISTANT, text="That sounds hard."))
    memory.add(Turn(id="u2", role=Role.USER, text="yeah"))
    history = alternating_history(memory.turns, 12)
    assert history[0] == ("user", "listen to this I can't sleep")
