"""AI-topic keyword definitions (fixed in the pre-registered plan, 2026-09-21)."""
import re

# Broad set: all AI-related title terms used for the primary analysis.
BROAD_PATTERNS = [
    r"\bAI\b", r"\bA\.I\.", r"\bartificial intelligence\b", r"\bmachine learning\b",
    r"\bdeep learning\b", r"\bneural net(?:work)?s?\b", r"\bGPT(?:-?\d+(?:\.\d+)?[a-z]?)?\b",
    r"\bChatGPT\b", r"\bLLMs?\b", r"\blarge language models?\b", r"\bOpenAI\b",
    r"\bDeepMind\b", r"\bAnthropic\b", r"\bchatbots?\b", r"\bgenerative AI\b",
    r"\bStable Diffusion\b", r"\bMidjourney\b", r"\bCopilot\b",
]
# Stable set: terms that existed throughout 2015-2026 (robustness check).
STABLE_PATTERNS = [
    r"\bAI\b", r"\bartificial intelligence\b", r"\bmachine learning\b",
    r"\bdeep learning\b", r"\bneural net(?:work)?s?\b",
]

# "AI" is matched case-sensitively to avoid words such as "ai" in other languages;
# every other term is case-insensitive.
def _compile(patterns: list[str]) -> list[re.Pattern]:
    out = []
    for p in patterns:
        flags = 0 if p in (r"\bAI\b", r"\bA\.I\.") else re.IGNORECASE
        out.append(re.compile(p, flags))
    return out

BROAD_RE = _compile(BROAD_PATTERNS)
STABLE_RE = _compile(STABLE_PATTERNS)


def is_ai(title: str, stable: bool = False) -> bool:
    regs = STABLE_RE if stable else BROAD_RE
    return any(r.search(title or "") for r in regs)


def mask_ai_terms(text: str) -> str:
    """Replace AI keywords in a comment with a neutral pronoun (topic-word masking check)."""
    for r in BROAD_RE:
        text = r.sub("it", text)
    return text
