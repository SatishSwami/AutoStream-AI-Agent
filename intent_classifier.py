"""
Intent Classification for AutoStream Agent.
Uses keyword heuristics as a fast pre-filter, then delegates to LLM for
ambiguous cases. Returns one of: GREETING, PRODUCT_INQUIRY, HIGH_INTENT_LEAD.
"""

from enum import Enum
from typing import Optional
import re


class Intent(str, Enum):
    GREETING = "GREETING"
    PRODUCT_INQUIRY = "PRODUCT_INQUIRY"
    HIGH_INTENT_LEAD = "HIGH_INTENT_LEAD"
    UNKNOWN = "UNKNOWN"


# --- Keyword signal banks ---

GREETING_SIGNALS = [
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b",
    r"\bgood afternoon\b", r"\bgood evening\b", r"\bwhat's up\b",
    r"\bhowdy\b", r"\bgreetings\b",
]

PRODUCT_SIGNALS = [
    r"\bprice\b", r"\bpricing\b", r"\bplan\b", r"\bplans\b",
    r"\bcost\b", r"\bfee\b", r"\bfeature\b", r"\bfeatures\b",
    r"\bbasic\b", r"\bpro\b", r"\bsubscription\b", r"\brefund\b",
    r"\bsupport\b", r"\bpolicy\b", r"\bhow does\b", r"\bwhat is\b",
    r"\btell me about\b", r"\bexplain\b", r"\binclude\b",
    r"\bresolution\b", r"\b720p\b", r"\b4k\b", r"\bcaption\b",
    r"\bvideo\b", r"\bmonth\b", r"\byear\b", r"\bannual\b",
    r"\bunlimited\b", r"\btrial\b",
]

HIGH_INTENT_SIGNALS = [
    r"\bsign up\b", r"\bsignup\b", r"\bregister\b", r"\bget started\b",
    r"\bsubscribe\b", r"\bi'm in\b", r"\bim in\b",
    r"\bbuy\b", r"\bpurchase\b", r"\btry it\b",
    r"\bonboard\b", r"\bsounds good\b.*\btry\b",
    r"\bi need this\b", r"\bthis is what i need\b", r"\bready to sign\b",
    r"\bwant to try\b", r"\bwant to sign\b",
    r"\bsign me up\b", r"\bwhere do i sign\b",
    r"\bi want.*sign\b", r"\bi want.*subscribe\b", r"\bi want.*try\b",
    r"\bi'd like.*sign\b", r"\bi'd like.*subscribe\b",
    r"\blet's go\b", r"\blets go\b",
]


def _match_signals(text: str, patterns: list) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def _signal_score(text: str) -> dict:
    """Returns weighted signal scores per intent."""
    return {
        Intent.GREETING: sum(1 for p in GREETING_SIGNALS if re.search(p, text.lower())),
        Intent.HIGH_INTENT_LEAD: sum(1 for p in HIGH_INTENT_SIGNALS if re.search(p, text.lower())),
        Intent.PRODUCT_INQUIRY: sum(1 for p in PRODUCT_SIGNALS if re.search(p, text.lower())),
    }


def classify_intent_heuristic(user_message: str) -> Intent:
    """
    Fast heuristic-based intent classification.
    Returns Intent enum.
    """
    scores = _signal_score(user_message)

    high = scores[Intent.HIGH_INTENT_LEAD]
    product = scores[Intent.PRODUCT_INQUIRY]
    greeting = scores[Intent.GREETING]

    # High intent takes top priority
    if high > 0:
        return Intent.HIGH_INTENT_LEAD

    if product > 0:
        return Intent.PRODUCT_INQUIRY

    if greeting > 0:
        return Intent.GREETING

    # Default for unknown short messages
    if len(user_message.strip()) < 20:
        return Intent.GREETING

    return Intent.PRODUCT_INQUIRY  # Default for longer unknown messages


def classify_intent_llm(user_message: str, conversation_history: list, llm_client) -> Intent:
    """
    LLM-based intent classification. Used as fallback or override for heuristics.
    Passes recent conversation context for better disambiguation.
    """
    recent_history = conversation_history[-6:] if conversation_history else []
    history_str = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in recent_history
    )

    prompt = f"""You are an intent classifier for a SaaS product (AutoStream).
Classify the user's latest message into exactly one of these intents:
- GREETING: casual hello or social opener with no product interest
- PRODUCT_INQUIRY: asking about features, pricing, policies, or capabilities
- HIGH_INTENT_LEAD: showing clear desire to sign up, buy, or try the product

Conversation history:
{history_str}

Latest user message: "{user_message}"

Respond with ONLY one word: GREETING, PRODUCT_INQUIRY, or HIGH_INTENT_LEAD.
"""

    try:
        response = llm_client.invoke(prompt)
        raw = response.content.strip().upper()
        if raw in Intent.__members__:
            return Intent(raw)
    except Exception:
        pass

    return classify_intent_heuristic(user_message)


def classify_intent(
    user_message: str,
    conversation_history: list,
    llm_client=None,
    use_llm: bool = True,
) -> Intent:
    """
    Master intent classification function.
    Uses heuristics first for speed; falls back to LLM for ambiguous cases.
    """
    heuristic_result = classify_intent_heuristic(user_message)

    # Short-circuit on high-confidence heuristic hits
    scores = _signal_score(user_message)
    total_signals = sum(scores.values())

    if heuristic_result == Intent.HIGH_INTENT_LEAD and scores[Intent.HIGH_INTENT_LEAD] >= 2:
        return Intent.HIGH_INTENT_LEAD

    if not use_llm or llm_client is None or total_signals >= 2:
        return heuristic_result

    # Defer to LLM for low-signal or ambiguous messages
    return classify_intent_llm(user_message, conversation_history, llm_client)
