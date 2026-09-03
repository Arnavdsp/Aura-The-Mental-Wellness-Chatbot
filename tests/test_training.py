"""Dataset construction and the behavioural evaluator (no GPU, no downloads)."""

from __future__ import annotations

from aura.training.data import (
    BuildStats,
    PreferencePair,
    build_coaching_pairs,
    build_psychology_pairs,
    deduplicate,
    normalise,
    validate,
)
from aura.training.evaluate import evaluate, score_response

GOOD_PSYCH_ROW = {
    "question": "I keep second-guessing every decision I make at work and it's exhausting.",
    "response_j": (
        "That sounds draining. Second-guessing often grows out of caring a great "
        "deal about getting things right, and that care has a real cost."
    ),
    "response_k": (
        "Just decide faster. Overthinking wastes everyone's time and you need to "
        "get over it before people notice."
    ),
}

GOOD_COACHING_ROW = {
    "messages": [
        {
            "role": "user",
            "content": (
                "I've been passed over for promotion twice now and I'm starting to "
                "wonder whether I belong at this company at all."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "That's a painful pattern to sit with. Before we get to the "
                "promotions themselves, what does belonging here mean to you?"
            ),
        },
    ]
}


def test_psychology_pairs_map_the_source_columns() -> None:
    stats = BuildStats()
    pairs = list(build_psychology_pairs([GOOD_PSYCH_ROW], stats))
    assert len(pairs) == 1
    assert pairs[0].prompt == GOOD_PSYCH_ROW["question"]
    assert pairs[0].rejected == GOOD_PSYCH_ROW["response_k"]
    assert pairs[0].chosen.endswith("?")  # an invitation is appended


def test_psychology_invitations_vary_across_rows() -> None:
    stats = BuildStats()
    pairs = list(build_psychology_pairs([GOOD_PSYCH_ROW] * 4, stats))
    endings = {pair.chosen.rsplit("\n\n", 1)[-1] for pair in pairs}
    assert len(endings) == 4


def test_rows_missing_columns_are_counted_not_crashed() -> None:
    stats = BuildStats()
    assert list(build_psychology_pairs([{"question": "hi"}], stats)) == []
    assert stats.dropped["missing_field"] == 1


def test_coaching_pairs_require_a_question_from_the_coach() -> None:
    stats = BuildStats()
    assert len(list(build_coaching_pairs([GOOD_COACHING_ROW], stats))) == 1

    flat = {
        "messages": [
            GOOD_COACHING_ROW["messages"][0],
            {"role": "assistant", "content": "You should just talk to your manager about it."},
        ]
    }
    stats = BuildStats()
    assert list(build_coaching_pairs([flat], stats)) == []
    assert stats.dropped["no_question"] == 1


def test_coaching_rejections_do_not_leak_the_prompt() -> None:
    """The original notebook spliced the user's own words into the rejection."""
    stats = BuildStats()
    pair = next(iter(build_coaching_pairs([GOOD_COACHING_ROW], stats)))
    assert "promotion" not in pair.rejected.lower()


def test_validation_rejects_short_and_identical_pairs() -> None:
    stats = BuildStats()
    assert not validate(PreferencePair("hi", "a" * 50, "b" * 50), stats)
    assert stats.dropped["prompt_too_short"] == 1
    assert not validate(PreferencePair("x" * 40, "same" * 20, "same" * 20), stats)
    assert stats.dropped["identical_pair"] == 1


def test_deduplicate_collapses_repeats() -> None:
    pair = PreferencePair("p" * 40, "c" * 60, "r" * 60)
    assert len(deduplicate([pair, pair, pair])) == 1


def test_normalise_collapses_whitespace() -> None:
    assert normalise("  a\n\n b\t c ") == "a b c"


def test_scoring_distinguishes_coaching_from_lecturing() -> None:
    coaching = score_response(
        "I'm stuck",
        "It sounds like you're carrying a lot. I might be wrong, but what feels heaviest?",
    )
    lecturing = score_response(
        "I'm stuck",
        "You should just make a list and stop overthinking it. You have depression.",
    )
    assert coaching.total > lecturing.total
    assert lecturing.avoids_directives is False
    assert lecturing.avoids_diagnosis is False


def test_evaluator_fails_a_model_that_ignores_crisis() -> None:
    report = evaluate(lambda prompt: "Have you tried going for a walk?")
    assert not report.passed_safety
    assert report.safety_failures


def test_evaluator_passes_a_model_that_surfaces_resources() -> None:
    def generate(prompt: str) -> str:
        from aura.safety import screen
        from aura.schemas import RiskLevel

        if screen(prompt).risk is RiskLevel.CRISIS:
            return "Please call 988, the crisis lifeline, right now."
        return "It sounds heavy. I might be off, but what feels hardest?"

    report = evaluate(generate)
    assert report.passed_safety
    assert report.mean_score > 0.5
