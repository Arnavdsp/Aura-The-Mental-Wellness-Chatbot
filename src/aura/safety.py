"""Crisis screening.

A wellness coach is not a clinician. This module's only job is to notice when a
message needs a human being rather than a language model, and to make sure the
person sees real contact details before anything else.

The screen is intentionally lexical and conservative: it runs before generation,
it costs nothing, and it fails toward escalation. It is a triage layer, not a
diagnosis, and it never silences the user — a crisis reply still acknowledges
what they said, it just leads with help.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aura.schemas import CrisisResource, RiskLevel, SafetyAssessment

# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------

CRISIS_RESOURCES: dict[str, list[CrisisResource]] = {
    "US": [
        CrisisResource(
            name="988 Suicide & Crisis Lifeline",
            contact="Call or text 988",
            region="US",
            url="https://988lifeline.org",
            note="24/7, free and confidential.",
        ),
        CrisisResource(
            name="Crisis Text Line",
            contact="Text HOME to 741741",
            region="US",
            url="https://www.crisistextline.org",
        ),
    ],
    "IN": [
        CrisisResource(
            name="Tele-MANAS (Govt. of India)",
            contact="Call 14416 or 1-800-891-4416",
            region="IN",
            url="https://telemanas.mohfw.gov.in",
            note="24/7, available in multiple languages.",
        ),
        CrisisResource(
            name="AASRA",
            contact="Call +91 98204 66726",
            region="IN",
            url="http://www.aasra.info",
        ),
    ],
    "UK": [
        CrisisResource(
            name="Samaritans",
            contact="Call 116 123",
            region="UK",
            url="https://www.samaritans.org",
        ),
    ],
    "INTL": [
        CrisisResource(
            name="Find a helpline in your country",
            contact="findahelpline.com",
            region="INTL",
            url="https://findahelpline.com",
            note="Verified crisis lines in 130+ countries.",
        ),
        CrisisResource(
            name="Emergency services",
            contact="Your local emergency number (e.g. 911, 112, 999, 112)",
            region="INTL",
            note="If there is immediate danger to life.",
        ),
    ],
}


def resources_for(region: str) -> list[CrisisResource]:
    """Region-specific lines, always followed by the international fallbacks."""
    region = (region or "INTL").upper()
    local = CRISIS_RESOURCES.get(region, [])
    intl = CRISIS_RESOURCES["INTL"]
    return [*local, *intl] if region != "INTL" else list(intl)


# --------------------------------------------------------------------------
# Lexical screen
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    category: str
    risk: RiskLevel
    pattern: re.Pattern[str]


def _rule(category: str, risk: RiskLevel, *alternatives: str) -> _Rule:
    joined = "|".join(alternatives)
    return _Rule(category, risk, re.compile(rf"(?:{joined})", re.IGNORECASE))


# Ordered most-severe first; the highest matching risk wins.
_RULES: tuple[_Rule, ...] = (
    _rule(
        "suicidal_intent",
        RiskLevel.CRISIS,
        r"\bkill(?:ing)?\s+my\s?self\b",
        r"\bend(?:ing)?\s+(?:my|it)\s+(?:life|all)\b",
        r"\btake\s+my\s+own\s+life\b",
        r"\bdon'?t\s+want\s+to\s+(?:be\s+here|live|wake\s+up)\b",
        r"\bbetter\s+off\s+(?:dead|without\s+me)\b",
        r"\bcommit\s+suicide\b",
        r"\bsuicid(?:e|al)\s+(?:plan|note|attempt)\b",
        r"\bwant(?:s|ed)?\s+to\s+die\b",
        r"\bwish(?:ed)?\s+i\s+(?:was|were)\s+dead\b",
        r"\bno\s+reason\s+to\s+(?:live|go\s+on)\b",
    ),
    _rule(
        "self_harm",
        RiskLevel.CRISIS,
        r"\b(?:cut|cutting|burn(?:ing)?|hurt(?:ing)?)\s+my\s?self\b",
        r"\bself[-\s]?harm\b",
        r"\boverdos(?:e|ing)\b",
    ),
    _rule(
        "harm_to_others",
        RiskLevel.CRISIS,
        r"\bkill\s+(?:him|her|them|someone|everyone)\b",
        r"\bhurt\s+(?:him|her|them|someone)\s+badly\b",
    ),
    _rule(
        "abuse",
        RiskLevel.ELEVATED,
        r"\b(?:he|she|they|my\s+\w+)\s+(?:hits?|beats?|hurts?|threatens?)\s+me\b",
        r"\bnot\s+safe\s+at\s+home\b",
        r"\bafraid\s+of\s+(?:him|her|them|my\s+partner)\b",
    ),
    _rule(
        "hopelessness",
        RiskLevel.ELEVATED,
        r"\bhopeless\b",
        r"\bcan'?t\s+(?:go\s+on|take\s+(?:it|this)\s+any\s?more)\b",
        r"\bnothing\s+matters\s+any\s?more\b",
        r"\bwhat'?s\s+the\s+point\s+of\s+anything\b",
    ),
    _rule(
        "substance_risk",
        RiskLevel.ELEVATED,
        r"\bdrink(?:ing)?\s+(?:to\s+forget|myself\s+to\s+sleep)\b",
        r"\brelaps(?:e|ed|ing)\b",
    ),
    _rule(
        "distress",
        RiskLevel.LOW,
        r"\bpanic\s+attack\b",
        r"\bcan'?t\s+stop\s+crying\b",
        r"\bbreak(?:ing)?\s+down\b",
        r"\bcompletely\s+overwhelmed\b",
    ),
)

# Phrases that usually mean the person is describing the past, someone else, or
# a hypothetical. They soften CRISIS to ELEVATED so we still respond carefully
# without alarming someone recounting recovery.
_MITIGATORS = re.compile(
    r"\b(?:used\s+to|years?\s+ago|back\s+then|when\s+i\s+was\s+(?:a\s+kid|younger)|"
    r"my\s+friend|a\s+character|in\s+the\s+(?:movie|book|show)|no\s+longer|"
    r"i'?m\s+past\s+that|recovered\s+from)\b",
    re.IGNORECASE,
)


CRISIS_PREAMBLE = (
    "I'm really glad you told me. What you're describing sounds serious, and I "
    "want to make sure you're not carrying it alone — I'm an AI, and this is the "
    "moment for a real person who is trained for it."
)

CRISIS_CLOSING = (
    "If you're in immediate danger, please contact your local emergency services "
    "now. I'm still here, and I'd like to keep talking with you if that helps."
)


def screen(text: str, *, region: str = "INTL", enabled: bool = True) -> SafetyAssessment:
    """Screen a message for risk. Cheap, deterministic and side-effect free."""
    if not enabled or not text or not text.strip():
        return SafetyAssessment()

    matched: list[str] = []
    risk = RiskLevel.NONE
    for rule in _RULES:
        if rule.pattern.search(text):
            matched.append(rule.category)
            if rule.risk.rank > risk.rank:
                risk = rule.risk

    if risk is RiskLevel.NONE:
        return SafetyAssessment()

    softened = False
    if risk is RiskLevel.CRISIS and _MITIGATORS.search(text):
        risk = RiskLevel.ELEVATED
        softened = True

    rationale = "Matched: " + ", ".join(sorted(set(matched)))
    if softened:
        rationale += " (softened — language suggests a past or third-party account)"

    return SafetyAssessment(
        risk=risk,
        matched_categories=sorted(set(matched)),
        rationale=rationale,
        resources=resources_for(region) if risk.rank >= RiskLevel.ELEVATED.rank else [],
        should_short_circuit=risk is RiskLevel.CRISIS,
    )


def crisis_message(assessment: SafetyAssessment) -> str:
    """The reply we send instead of a generated one when risk is CRISIS."""
    lines = [CRISIS_PREAMBLE, "", "**Reach someone now:**"]
    for resource in assessment.resources:
        entry = f"- **{resource.name}** — {resource.contact}"
        if resource.note:
            entry += f"  \n  _{resource.note}_"
        lines.append(entry)
    lines += ["", CRISIS_CLOSING]
    return "\n".join(lines)
