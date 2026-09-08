"""Sonnet-only tuning layer for question_architect's PROMPT_TEMPLATE.

Testing phase, kept fully separate from the DeepSeek path (question_architect
imports and uses this ONLY when config.LLM_PROVIDER == "anthropic" — see the
selection at question_architect.py's PROMPT_TEMPLATE.format() call site).
DeepSeek's own PROMPT_TEMPLATE in question_architect.py is never edited by
this file and never imports from it.

Observed failure this layer exists to fix (live test, Sport Shoes / China,
2026-08-25): DeepSeek's PROMPT_TEMPLATE re-uses the literal `{segment}`
string mid-sentence in several instructions and examples (e.g. "EVERY
question must be about {segment}", "What do you mainly use {segment} for?").
When `segment` holds the Title-Case category string as typed by the caller
("Sport Shoes"), DeepSeek was observed to naturalise it in its own output
(lower-cased mid-sentence, reworded as "sports shoes"/"your shoes"). Claude
Sonnet followed the same instructions more literally and reproduced the
Title-Case string verbatim mid-sentence throughout the generated survey
("What do you mainly use Sport Shoes for?"), which reads as templated rather
than written by a person. This is a difference in how the two models follow
the SAME instruction, not a wording problem in the instruction's intent —
so the fix is an explicit rule Sonnet needs stated outright rather than a
change to what the instruction is asking for.
"""

from src.nodes.question_architect import PROMPT_TEMPLATE as _BASE_TEMPLATE

_SEGMENT_CASING_RULE = """\
SEGMENT-NAME CASING (Sonnet-specific — read carefully):
The {segment} placeholder below is filled with the category name exactly as
provided by the caller, which may be Title Case (e.g. "Sport Shoes"). When
you write a question or example that names the category mid-sentence, write
it the way an ordinary person would say it OUT LOUD — normal sentence casing,
not the literal placeholder casing. "What do you mainly use sport shoes
for?", never "What do you mainly use Sport Shoes for?". Capitalize the
category name only where normal English capitalization already requires it
(start of a sentence, a proper noun within it). This applies everywhere
{segment} appears in a question stem or option text; it does not change what
the question asks, only how the category name is cased when spoken aloud.

"""

# Injected once, right after the persona/twenty-five-years framing opens the
# template, so it lands before any example that reuses {segment} mid-sentence.
PROMPT_TEMPLATE = _BASE_TEMPLATE.replace(
    "WHAT TWENTY-FIVE YEARS SOUNDS LIKE",
    _SEGMENT_CASING_RULE + "WHAT TWENTY-FIVE YEARS SOUNDS LIKE",
    1,
)

assert PROMPT_TEMPLATE != _BASE_TEMPLATE, (
    "question_architect.PROMPT_TEMPLATE no longer contains the expected "
    "anchor text — update the .replace() target above to match its current "
    "wording so the Sonnet casing rule keeps landing in the prompt."
)
