"""Sonnet-only tuning layer for survey_simulator's judging PROMPT.

Testing phase, kept fully separate from the DeepSeek path (survey_simulator
imports and uses this ONLY when config.LLM_PROVIDER == "anthropic" — see the
selection at survey_simulator.py's PROMPT.format() call site in
_simulate_batch). DeepSeek's own PROMPT in survey_simulator.py is never
edited by this file.

Observed failure this layer exists to fix (live test, Sport Shoes / China,
2026-08-25): a Sonnet run showed 0 confident questions / 0% grounded_pct on
4 consecutive revision passes. Traced to src/llm.py's _RateLimitedStructured
(the generic with_structured_output path used for every non-DeepSeek
provider): a structured-output call that raises (malformed/truncated tool
call, schema validation failure) is NOT a rate-limit/timeout case, so it
propagated straight out of get_structured_llm(...).ainvoke(...) in
_simulate_batch, which catches it with a bare `except Exception: by_id = {}`
and silently falls back to `_fallback(q, confidence="low")` for the ENTIRE
batch. That fallback was being mistaken for a genuine low-confidence
judgement on every subsequent revision. The real fix for that specific
failure is the bounded retry now added to _RateLimitedStructured in
src/llm.py (Anthropic-only, does not touch DeepSeek's _DeepSeekStructured
path at all). This prompt-side file additionally:
  - carries the same {segment} literal-Title-Case fix as
    question_architect_prompts_sonnet.py, since this PROMPT also echoes
    {segment} into distribution_note guidance text.
  - adds an explicit instruction to keep each batch's JSON compact and
    complete rather than verbose, since a truncated tool call is one of the
    concrete ways the exception above can be triggered in the first place.
"""

from src.nodes.survey_simulator import PROMPT as _BASE_PROMPT

_SONNET_NOTES = """\
SONNET-SPECIFIC NOTES (read carefully):
- The {segment} value below may be Title Case as typed by the study author
  (e.g. "Sport Shoes"). Anywhere you name the category inside a
  distribution_note, write it the way a person would say it out loud in an
  ordinary sentence — normal casing, not the literal placeholder casing.
- Keep distribution_note SHORT (one sentence, under 30 words) and keep every
  other field terse. This response must complete as ONE valid tool call for
  the full batch of questions above — a long, padded distribution_note on
  early questions can push a later question's estimate past the output
  budget and truncate the whole batch. Terse and complete beats detailed and
  cut off.

"""

# Injected right after the "HARD RULES:" heading, before rule 1, so it lands
# before the batch of formatting requirements it's adding to.
PROMPT = _BASE_PROMPT.replace(
    "HARD RULES:\n",
    "HARD RULES:\n" + _SONNET_NOTES,
    1,
)

assert PROMPT != _BASE_PROMPT, (
    "survey_simulator.PROMPT no longer contains the expected anchor text "
    "('HARD RULES:\\n') — update the .replace() target above to match its "
    "current wording so the Sonnet notes keep landing in the prompt."
)
