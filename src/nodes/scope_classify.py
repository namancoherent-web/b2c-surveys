"""Node 1 — scope_classify (segment-only B2C gate).

# change for b2c questionarie

Runs ONCE per segment (before regional fan-out). Classifies B2C / B2B / other
with confidence. Hard-stops only high-confidence pure B2B (or clearly
non-consumer other). Ambiguous + hybrid (B2B-with-consumer-proxy) PROCEED.

``regions_to_simulate`` is always the fixed 5 continents — Global is A6 fan-in,
not a generated region.
"""

from __future__ import annotations

import os

from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import ScopeResult
from src.question_plan import GEOGRAPHIC_REGIONS
from src.state import SurveyState

# change for b2c questionarie — lenient by default (hybrids / ambiguous proceed)
B2C_GATE_STRICT = os.getenv("B2C_GATE_STRICT", "false").lower() in ("1", "true", "yes")

PROMPT_TEMPLATE = """\
You are a market research lead scoping a CONSUMER (B2C) survey pipeline.
Today's date is {today} — use it as your reasoning frame.

Input to scope. This may be a market category ("organic milk"), a description of
a business or product ("an app that books last-minute dentist appointments"), or
a company name on its own:
    "{market_segment}"

Do the following:
1. Normalize into a clean, canonical MARKET CATEGORY, whatever form the input
   took. A description or company name becomes the category it competes in
   ("an app that books last-minute dentist appointments" -> "Dental Appointment
   Booking Apps"). Drop "global"/"market" fluff. Everything downstream reads
   this field, so it must name a category, not a company.
2. Classify segment_type as EXACTLY one of: b2c | b2b | other
   - b2c: everyday consumers / households buy or use it (sneakers, milk, air fryers).
   - b2b: ONLY businesses/institutions buy it AND there is no meaningful household
     consumer end-user (industrial hydraulic seals, enterprise ERP, factory robots).
   - other: not a commercial consumer or B2B goods market (e.g. pure public policy).
3. Set classification_confidence: high | medium | low
   - high: unambiguous pure B2C or pure B2B.
   - medium/low: ambiguous category, mixed buyers, or you are unsure.
4. Answer WHO BUYS, WHO PAYS, WHO DECIDES. That is what separates the three
   cases below, and the classification_reason must state all three in 2-3
   sentences a human can read.

   is_hybrid=true — ONE buyer (a business or institution), but a clear consumer /
   patient / caregiver end-user stands in for them. The consumer does not pay.
   Example: prescription drugs — manufacturers sell to health systems, patients
   are the end users. Survey the consumer proxy.

   sells_to_both=true — TWO genuine buyers, each paying on their own account:
   consumers buy it for themselves AND businesses buy it for their teams.
   Example: a note-taking app on individual subscriptions AND enterprise seats;
   a food delivery platform with consumer orders AND business catering accounts.
   Write dual_motion_reason naming both motions and which is larger.

   These two are NOT the same and must never both be true. If the consumer pays,
   it is sells_to_both. If a business pays on the consumer's behalf, it is
   is_hybrid.

5. needs_clarification=true when you genuinely cannot tell — a bare company name
   you do not recognise, or a phrase that could plausibly be either a consumer or
   a business market. Write clarifying_question as the ONE question that would
   settle it (what the product does and who pays for it). Do NOT guess and do NOT
   ask a generic "tell me more". A wrong guess here produces an entire study for
   the wrong market, which is far more costly than asking.
6. Define a CONSUMER audience_profile (demographics lean, behaviors, qualifiers,
   exclusions, rationale). For is_hybrid, define the consumer-proxy and note it in
   assumptions. For sells_to_both, define the CONSUMER side. For pure B2B you may
   still sketch a proxy, but classification stays b2b.
7. Put low-confidence guesses into assumptions.

region must be "Global" (geographic fan-out is handled outside this node).

--- Example 1 (B2C) ---
Raw: "global organic milk market"
-> normalized_segment: "Organic Milk"
   segment_type: b2c, confidence: high, is_hybrid: false, sells_to_both: false
   reason: "Household shoppers buy organic milk, pay for it themselves, and
            decide at the shelf."

--- Example 2 (pure B2B — the gate stops this) ---
Raw: "enterprise CRM software"
-> normalized_segment: "Enterprise CRM Software"
   segment_type: b2b, confidence: high, is_hybrid: false, sells_to_both: false
   reason: "Companies buy it, procurement and IT pay, a buying committee decides.
            Employees use it but never purchase it."

--- Example 3 (consumer proxy — proceed) ---
Raw: "global insulin market"
-> normalized_segment: "Insulin"
   segment_type: b2b, confidence: medium, is_hybrid: true, sells_to_both: false
   reason: "Manufacturers sell to healthcare systems and insurers pay, but
            patients are the end users whose experience the study needs."

--- Example 4 (sells to both — proceed) ---
Raw: "a project management tool sold to freelancers and enterprise teams"
-> normalized_segment: "Project Management Software"
   segment_type: b2c, confidence: high, is_hybrid: false, sells_to_both: true
   dual_motion_reason: "Freelancers and individuals buy their own paid plans;
            companies separately buy team seats. The business motion is the
            larger revenue line, the consumer motion the larger user count."

--- Example 5 (too thin — ask) ---
Raw: "Meridian"
-> needs_clarification: true
   clarifying_question: "What does Meridian sell, and who pays for it — individual
            consumers, or businesses buying for their staff?"

Now scope the input given above.
"""


# change for b2c questionarie — A11: the message a rejected B2B market gets.
# Stating the methodological reason matters: this is not "unsupported", it is
# the wrong instrument for the job.
B2B_REFUSAL = (
    "This agent is scoped to B2C and hybrid B2C/B2B markets. {segment} is a "
    "business-to-business market, and B2B research needs a fundamentally "
    "different methodology — firmographic rather than behavioural segmentation, "
    "buying committees rather than a single decision maker, and a sales cycle "
    "measured in months. A consumer instrument would produce confident-looking "
    "numbers about the wrong population."
)


def decide_b2c_gate(
    *,
    segment_type: str,
    classification_confidence: str,
    is_hybrid: bool,
    classification_reason: str,
    strict: bool | None = None,
    sells_to_both: bool = False,
    needs_clarification: bool = False,
    clarifying_question: str = "",
    normalized_segment: str = "",
) -> dict:
    """Three-way gate: accepted | rejected | needs_clarification.

    # change for b2c questionarie — A11

    Order matters. Clarification is checked first: an input we cannot classify
    must not fall through to a confidence-based branch and get surveyed anyway.
    B2B is then rejected at ANY confidence — the old rule let a medium-confidence
    B2B market through, which is precisely the case most likely to be wrong.
    """
    strict = B2C_GATE_STRICT if strict is None else strict
    st = (segment_type or "").strip().lower()
    conf = (classification_confidence or "medium").strip().lower()
    reason = (classification_reason or "").strip()
    label = normalized_segment or "This market"

    # 1. Cannot classify — ask rather than guess.
    if needs_clarification:
        return {
            "gate_passed": False,
            "gate_decision": "needs_clarification",
            "status": "needs_clarification",
            "reason": (
                clarifying_question
                or "The input is too thin to classify. What does it sell, and who pays?"
            ),
            "clarifying_question": clarifying_question,
        }

    # 2. Straight consumer market.
    if st == "b2c":
        return {
            "gate_passed": True,
            "gate_decision": "accepted_b2c",
            "status": "accepted",
            "reason": reason or "Classified as B2C consumer market.",
        }

    # 3. Sells to consumers AND businesses — the consumer motion is in scope.
    if sells_to_both:
        return {
            "gate_passed": True,
            "gate_decision": "accepted_dual_motion",
            "status": "accepted",
            "reason": reason or "Sells to both consumer and business buyers — surveying the consumer motion.",
        }

    # 4. B2B market whose end user is a consumer proxy (patients, caregivers).
    if is_hybrid and not strict:
        return {
            "gate_passed": True,
            "gate_decision": "accepted_hybrid_proxy",
            "status": "accepted",
            "reason": reason or "B2B market with a consumer-proxy end user — proceeding.",
        }

    # 5. Everything else that is not B2C is out of scope, at ANY confidence.
    if st in ("b2b", "other"):
        return {
            "gate_passed": False,
            "gate_decision": "rejected_strict" if strict else f"rejected_pure_{st}",
            "status": "rejected",
            "reason": B2B_REFUSAL.format(segment=label) if st == "b2b" else (
                reason or f"{label} is not a consumer market — out of scope."
            ),
            "classification_note": reason,
        }

    if strict and st != "b2c":
        return {
            "gate_passed": False,
            "gate_decision": "rejected_strict",
            "status": "rejected",
            "reason": reason or f"Strict gate rejected segment_type={st}.",
        }

    # Unrecognised segment_type — treat as unclassifiable rather than assume B2C.
    return {
        "gate_passed": False,
        "gate_decision": "needs_clarification",
        "status": "needs_clarification",
        "reason": (
            f"Could not classify {label!r} as a consumer or business market. "
            "What does it sell, and who pays for it?"
        ),
        "clarifying_question": clarifying_question,
    }


def scope_classify(state: SurveyState) -> dict:
    """Classify segment once; set fixed 5 regions; apply B2C gate flags."""
    market_segment = state.get("market_segment")
    if not market_segment:
        raise ValueError("scope_classify requires 'market_segment' in state.")

    # Passthrough when classification was already provided (should not re-LLM).
    if state.get("classification_preseeded") and state.get("normalized_segment"):
        geo = state.get("region")
        return {
            "normalized_segment": state.get("normalized_segment"),
            "segment_type": state.get("segment_type") or "b2c",
            "classification_confidence": state.get("classification_confidence") or "high",
            "is_hybrid": bool(state.get("is_hybrid")),
            "classification_reason": state.get("classification_reason") or "",
            "audience_profile": state.get("audience_profile") or {},
            "region": geo or "Global",
            "regions_to_simulate": [geo] if geo and geo != "Global" else list(GEOGRAPHIC_REGIONS),
            "gate_passed": True,
            "gate_decision": state.get("gate_decision") or "accepted_preseeded",
            "status": "accepted",
        }

    today = current_date_context()["iso"]
    prompt = PROMPT_TEMPLATE.format(today=today, market_segment=market_segment)
    result: ScopeResult = get_structured_llm(ScopeResult, temperature=0).invoke(prompt)

    st = result.segment_type.value if hasattr(result.segment_type, "value") else str(result.segment_type)
    conf = (
        result.classification_confidence.value
        if hasattr(result.classification_confidence, "value")
        else str(result.classification_confidence or "medium")
    )
    gate = decide_b2c_gate(
        segment_type=st,
        classification_confidence=conf,
        is_hybrid=bool(result.is_hybrid),
        classification_reason=result.classification_reason or "",
        # change for b2c questionarie — A11
        sells_to_both=bool(getattr(result, "sells_to_both", False)),
        needs_clarification=bool(getattr(result, "needs_clarification", False)),
        clarifying_question=getattr(result, "clarifying_question", "") or "",
        normalized_segment=result.normalized_segment or "",
    )

    # Fixed 5 continents — Global is A6 fan-in only.
    regions_to_simulate = list(GEOGRAPHIC_REGIONS)

    return {
        "normalized_segment": result.normalized_segment,
        "segment_type": st,
        "classification_confidence": conf,
        "is_hybrid": bool(result.is_hybrid),
        # change for b2c questionarie — A11
        "sells_to_both": bool(getattr(result, "sells_to_both", False)),
        "dual_motion_reason": getattr(result, "dual_motion_reason", "") or "",
        "needs_clarification": bool(getattr(result, "needs_clarification", False)),
        "clarifying_question": getattr(result, "clarifying_question", "") or "",
        "region": "Global",
        "regions_to_simulate": regions_to_simulate,
        "classification_reason": result.classification_reason,
        "audience_profile": result.audience_profile.model_dump(),
        "gate_passed": gate["gate_passed"],
        "gate_decision": gate["gate_decision"],
        "status": gate["status"],
        "rejection_reason": None if gate["gate_passed"] else gate["reason"],
    }


def reject_stop(state: SurveyState) -> dict:
    """Terminal node — rejected non-B2C segment; no harvest/personas/etc."""
    # change for b2c questionarie
    reason = (
        state.get("rejection_reason")
        or state.get("classification_reason")
        or "Rejected by B2C gate."
    )
    return {
        "status": "rejected",
        "rejection_reason": reason,
        "final_survey": {
            "status": "rejected",
            "reason": reason,
            "segment_type": state.get("segment_type"),
            "classification_confidence": state.get("classification_confidence"),
            "is_hybrid": state.get("is_hybrid"),
            "gate_decision": state.get("gate_decision"),
            "market_segment": state.get("market_segment"),
            "normalized_segment": state.get("normalized_segment"),
            "classification_reason": state.get("classification_reason"),
        },
    }


def clarify_stop(state: SurveyState) -> dict:
    """Terminal node — input too thin to classify; ask instead of guessing.

    # change for b2c questionarie — A11

    Distinct from ``reject_stop``: a rejection is a decision, this is the absence
    of one. The artifact carries the question so a caller can re-run with an
    answer rather than having to work out what was missing.
    """
    question = (
        state.get("clarifying_question")
        or state.get("rejection_reason")
        or "What does this sell, and who pays for it — consumers or businesses?"
    )
    return {
        "status": "needs_clarification",
        "rejection_reason": question,
        "final_survey": {
            "status": "needs_clarification",
            "clarifying_question": question,
            "reason": question,
            "market_segment": state.get("market_segment"),
            "normalized_segment": state.get("normalized_segment"),
            "segment_type": state.get("segment_type"),
            "classification_confidence": state.get("classification_confidence"),
            "gate_decision": state.get("gate_decision"),
            "classification_reason": state.get("classification_reason"),
        },
    }


if __name__ == "__main__":
    import json

    # Gate unit checks (no LLM). Each case names the outcome it must produce.
    # change for b2c questionarie — A11: covers all three gate outcomes.
    CASES = [
        ("b2c accepted", "accepted", dict(
            segment_type="b2c", classification_confidence="high",
            is_hybrid=False, classification_reason="consumers buy and pay")),
        ("b2b high rejected", "rejected", dict(
            segment_type="b2b", classification_confidence="high",
            is_hybrid=False, classification_reason="OEM only",
            normalized_segment="Enterprise CRM Software")),
        # The old gate let this one through — the case most likely to be wrong.
        ("b2b medium rejected", "rejected", dict(
            segment_type="b2b", classification_confidence="medium",
            is_hybrid=False, classification_reason="looks like procurement")),
        ("consumer proxy accepted", "accepted", dict(
            segment_type="b2b", classification_confidence="medium",
            is_hybrid=True, classification_reason="patients are end users")),
        ("dual motion accepted", "accepted", dict(
            segment_type="b2c", classification_confidence="high",
            is_hybrid=False, sells_to_both=True,
            classification_reason="individuals and teams both pay")),
        ("b2b but sells to both accepted", "accepted", dict(
            segment_type="b2b", classification_confidence="high",
            is_hybrid=False, sells_to_both=True,
            classification_reason="enterprise seats plus individual plans")),
        ("thin input asks", "needs_clarification", dict(
            segment_type="b2c", classification_confidence="low",
            is_hybrid=False, needs_clarification=True,
            clarifying_question="What does Meridian sell, and who pays?",
            classification_reason="")),
        ("unknown type asks", "needs_clarification", dict(
            segment_type="", classification_confidence="low",
            is_hybrid=False, classification_reason="")),
    ]

    failures = []
    for name, expected, case in CASES:
        out = decide_b2c_gate(**case)
        ok = out["status"] == expected
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {out['status']} ({out['gate_decision']})")
        if not ok:
            failures.append(f"{name}: expected {expected}, got {out['status']}")
            print(json.dumps(out, indent=4))

    print("\n" + "=" * 58)
    print(f"  gate checks {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
    raise SystemExit(1 if failures else 0)
