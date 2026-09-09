"""Node 6 — validator_critic.

The data-integrity gate for the answered questions. It does NOT rewrite —
it audits, scores, computes grounded_pct, and emits targeted feedback so Node 4
(question_architect) or Node 5 (answer_estimator) can fix specific items. Keeping
critique separate from generation is what gives real quality control.

Mechanical checks run in CODE (deterministic): distribution math, null/range,
counts-vs-plan, grounded_pct, open-ended detection, and the clear-cut label
integrity rule (adjacent figure + empty label = misattribution). An LLM pass
handles the JUDGMENT checks (semantic cross-tab dedup, mis-mapped grounding,
nuanced label reasoning); a heuristic dedup runs as a fallback so dedup still
works if the LLM is unavailable.

Label integrity is the enforcement of the permissive-with-labeling contract:
a borrowed adjacent figure is allowed, but losing its grounding_label is
MISATTRIBUTION — the highest-priority failure.
"""

import os
import re

from pydantic import BaseModel

from src import narrative
from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import Critique, QuestionIssue
from src.question_plan import (
    CATEGORY_PLAN,
    GENERATION_MODE,
    MIN_TOTAL_QUESTIONS,
    QUESTIONS_PER_SECTION,
    QUESTIONS_PER_TAB_MIN,
    QUESTIONS_PER_TAB_TARGET,
    TOTAL_TARGET,
)
from src.question_similarity import (
    LOOSE_JACCARD,
    NEAR_EXACT_JACCARD,
    is_near_exact,
    token_set_jaccard,
    tokens as _sim_tokens,
)
from src.state import SurveyState

VALID_TYPES = {"single_choice", "multiple_choice", "ranking", "likert_5", "likert_7"}
_SUM_TO_100_TYPES = {"single_choice", "ranking", "likert_5", "likert_7"}
PASS_SCORE = 0.85

# change for b2c questionarie — SAME CONSTRUCT, DIFFERENT WORDS. Word
# overlap misses the duplicates that matter: "What would you give up first
# to get a lower price?" and "If it cannot have everything, which would you
# give up first?" share only 25% of their tokens but are the same question,
# and a respondent answers them identically. Match on the CONSTRUCT the stem
# measures instead of its wording — two questions measuring the same thing
# in one section is a wasted slot in a 6-question section.
#
# change for b2c questionarie -- CHECK (Section 30): hoisted to module level
# (was a local inside validator_critic()) so assembler.py's independent
# zero-tolerance-clean check at the final restore point can import the
# SAME table instead of risking a second copy drifting out of sync.
CONSTRUCTS = (
    ("trade_off", r"(?i)\b(give up|trade[- ]?off|sacrifice|forgo|"
                  r"cannot have everything|compromise on)\b"),
    # change for b2c questionarie -- CHECK (live, Sport Shoes/China,
    # 2026-09-06): "Would you buy the same pair again?" and "How likely are
    # you to buy sport shoes from the same brand again?" both shipped in the
    # same file -- the same repurchase-intent construct, worded two ways.
    # The regex only matched the "how likely...buy" phrasing; the simpler
    # "would you buy...again" phrasing (which the 6-year-old-simplicity
    # prompt rule now actively encourages the model to write) matched
    # nothing at all, so find_construct_duplicates() never saw them as the
    # same construct. Widened to a bag-of-concepts match so any phrasing of
    # "would/will/likely to buy/purchase again" is caught, regardless of
    # how plain the wording is -- simpler wording must not be a way to
    # dodge duplicate detection.
    ("purchase_intent", r"(?i)\b(how likely.{0,30}(buy|purchase|upgrade)|"
                        r"intend to (buy|purchase)|plan to (buy|purchase)|"
                        r"(would|will) you (buy|purchase).{0,20}again|"
                        r"buy.{0,20}same.{0,20}(brand|pair).{0,10}again|"
                        r"buy.{0,20}again)\b"),
    ("willingness_to_pay", r"(?i)\b(most you would pay|willing to pay|"
                           r"how much more would you pay|pay extra)\b"),
    # change for b2c questionarie -- CHECK (live, Sport Shoes/China): this
    # pattern required the literal lead-in "Overall, how satisf...", which
    # matched the anchor's OLD wording. The 2026-08-25 manager-wording pass
    # simplified the hardcoded anchor in standard_sections.py from "Overall,
    # how satisfied are you with X you currently use?" to "How satisfied are
    # you with your current X?" -- dropping "Overall," -- without updating
    # this regex. The anchor stopped matching its own construct, so
    # find_construct_duplicates() no longer saw it as a satisfaction
    # question at all, and the architect kept independently writing a
    # second (and sometimes third) satisfaction question every revision
    # with no covered-construct feedback to stop it. Widened to a bag-of-
    # concepts match so the construct is detected regardless of whether an
    # "Overall," lead survives a future wording pass.
    ("overall_satisfaction", r"(?i)how\s+satisf"),
    ("advocacy", r"(?i)\b(recommend)\b"),
    ("replacement_cycle", r"(?i)\b(how often.{0,25}replace|how long.{0,25}"
                          r"(keep|last)|replacement cycle)\b"),
    # change for b2c questionarie — "what matters", "what was most
    # important", "what must it do", "how do you judge quality" and "what
    # disappoints you" are ONE construct — attribute importance — asked five
    # ways. The last run spent 5 of 30 slots on it.
    # change for b2c questionarie — Issue 5. This used to lump five
    # DIFFERENT constructs into one "attribute_importance" bucket and cut
    # four of them. That is a B2B procurement instinct: in B2B a near-
    # identical question usually IS redundant because the buyer has one
    # practical answer. A B2C segmentation study deliberately probes the
    # same product from several psychological angles, and each feeds a
    # different decision:
    #     "what matters when you choose"  -> positioning
    #     "what it must DO to keep using" -> retention
    #     "what it must BE to consider"   -> trust / safety filter
    #     "how you judge quality"         -> quality heuristic
    #     "which claims you trust"        -> ad messaging
    # Those are five constructs, not one. Only the SAME angle asked twice
    # is redundant, so each angle now has its own narrow key and collides
    # only with itself.
    ("decision_driver",
     r"(?i)(what matters most|most important (factor|thing)|"
     r"mattered most when you (chose|choose|picked))"),
    ("retention_condition",
     r"(?i)(must .{0,30}\bdo\b.{0,30}(keep|continue) using|"
     r"to keep using it)"),
    ("quality_heuristic",
     r"(?i)(how do you judge|judge the quality|tells you it is (well made|"
     r"high quality)|what tells you it is working)"),
    ("claim_credibility",
     r"(?i)(which claims|claims would make you|claims you trust)"),
    # change for b2c questionarie -- "most disappoints you", "which
    # frustrations", "which problems" are the same construct (the
    # complaint/friction battery) however the verb or noun is phrased.
    # Widened to catch "disappoints" anywhere in the stem, not only right
    # after "what", which is what let Q22/Q23/Q24 all ship unmerged.
    # change for b2c questionarie -- CHECK (live, Skincare/Germany,
    # 2026-09-08): "what would make you SWITCH from X" and "what
    # disappoints you about X" shipped as two separate questions with
    # near-identical option lists ("it causes a reaction/irritation" vs
    # "they irritate my skin", "stops working" vs "don't deliver results").
    # The pain point IS the switch trigger for a repeat-use consumable --
    # this is one construct, not two, however the stem is framed (complaint
    # vs. hypothetical future action). Merged "switch" phrasing into this
    # same construct so find_construct_duplicates() catches the pair.
    ("disappointment",
     r"(?i)(disappoints?\b|which (frustrations|problems)|"
     r"problems .{0,20}(experienced|had)|biggest disappointment|"
     r"(would make you|makes? you) switch|switch (from|to|away)|"
     r"switch to a different)"),
    ("usage_frequency",
     r"(?i)(how often do you (wear|use)|in the last \d+ days,? how many days)"),
    # change for b2c questionarie -- CHECK (Section 28c): "how confident
    # are you that you can find X that fits" and "how often do you have
    # trouble finding X that fits" are the same construct (fit-finding
    # difficulty) with essentially zero token overlap (jaccard 0.125 on
    # the real pair) -- confirmed BOTH questions were independently
    # flagged as carrying no segmentation signal at all
    # (nonDiscriminatingQuestions), yet neither the option-overlap check
    # nor any prior construct pattern caught them as duplicates of each
    # other. A confidence-scale framing and a trouble-frequency framing
    # of the same fit problem are not two distinct business questions.
    # change for b2c questionarie -- widened after "confident are you IN
    # FINDING X that fits" and "struggle TO FIND X that fits" both slipped
    # past the original phrase-sequence version (only "confident...find/get"
    # and "trouble/difficulty finding" were covered, missing "in finding"
    # and "struggle to find" entirely -- confirmed live, the pair still
    # shipped together on the very next generation after the original fix).
    # Rebuilt as a bag-of-concepts match: any difficulty/frequency word,
    # any find-related verb form, and "fit" all present in the same
    # sentence, in ANY order -- robust to whichever specific verb phrasing
    # the model picks, rather than chasing each new variant one at a time.
    ("fit_difficulty",
     r"(?i)(?=.*\b(confiden\w*|trouble|difficult\w*|struggl\w*|hard(?:er)?)\b)"
     r"(?=.*\bfind\w*\b)(?=.*\bfit\w*\b).*"),
)


def find_construct_duplicates(answered: list) -> dict[str, list]:
    """Group answered questions by shared construct (CONSTRUCTS table).

    Returns {construct_name: [question_dict, ...]} for every construct
    matched by 2+ questions -- i.e. only the groups that ARE duplicates.
    Shared by validator_critic() (during the revision loop) and
    assembler.py's zero-tolerance-clean check (at final restore) so both
    call sites agree on what counts as a known duplicate.
    """
    by_construct: dict[str, list] = {}
    for q in answered:
        if (q.get("question_layer") or "") in ("screening", "profiling"):
            continue
        text = q.get("text") or ""
        for name, pat in CONSTRUCTS:
            if re.search(pat, text):
                by_construct.setdefault(name, []).append(q)
                break
    groups = {name: qs for name, qs in by_construct.items() if len(qs) >= 2}

    # change for b2c questionarie -- CHECK (live, Japan health devices
    # 2026-09-09): this generic CONSTRUCTS table has ONE "overall_satisfaction"
    # bucket for the whole survey, but the 23-theme framework deliberately
    # has TWO distinct satisfaction slots -- core_function_satisfaction
    # ("how satisfied with HOW WELL it works", Section 3) and
    # overall_satisfaction ("how satisfied with your CURRENT device",
    # Section 4). Both matched the old bucket and were flagged as
    # duplicates, even though the framework's own slot list says they are
    # meant to coexist. If every question in a construct-duplicate group
    # maps to a DIFFERENT REQUIRED_SLOTS theme, the framework is the more
    # specific authority here -- drop the group.
    _filtered: dict[str, list] = {}
    for name, qs in groups.items():
        _slot_keys = set()
        for q in qs:
            _t = q.get("text") or ""
            _k = next(
                (k for slots in REQUIRED_SLOTS.values() for k, _l, p in slots
                 if re.search(p, _t)),
                None,
            )
            _slot_keys.add(_k)
        if None not in _slot_keys and len(_slot_keys) == len(qs):
            continue  # every question maps to a different named theme
        _filtered[name] = qs
    return _filtered


# change for b2c questionarie -- CHECK (user directive, 2026-09-08): the 16
# required question types (4 per behavioural section) existed ONLY as
# advisory text in question_architect.py's _TAB_GUIDANCE. Nothing in code
# ever checked whether they were actually generated, which is why -- across
# several live runs -- the price question went missing entirely, the NPS
# question shipped as a 5-option single_select instead of an 11-point 0-10
# scale, and a section quietly shipped 3 questions instead of 4. Prompt
# text alone is a suggestion; this table is the gate.
#
# Each entry: (slot_key, human_label, regex that must match >=1 question
# stem in that section). Keyed by CANONICAL section id so a market-specific
# section label (e.g. "Product / Ecosystem Experience") still maps back.
# change for b2c questionarie -- CHECK (user directive, 2026-09-09): expanded
# from 16 slots (4 per section) to the 23-slot framework, 5 / 7 / 6 / 5.
# Section 4 now carries a SEPARATE switching_trigger alongside top_pain_point
# (they were deliberately merged under the old 16-slot framework), and
# advocacy is explicitly CATEGORY advocacy asked last.
REQUIRED_SLOTS: dict[str, tuple] = {
    # --- Section 1: 5 questions ------------------------------------------
    "consumer_profile": (
        ("usage_frequency", "usage frequency (how often/how many days)",
         r"(?i)how (often|many (days|times))\b"),
        # Widened after the Japan run: "What is the main reason you take a
        # reading with your device?" is a valid primary-use-case question but
        # matched nothing, so the gate rejected correct output.
        ("primary_use_case", "primary use case (what they mainly use it for)",
         r"(?i)(mainly|mostly|most often|primarily)\s+use|"
         r"what do you use .{0,40}\bfor\b|"
         r"(main|biggest|number one) reason you\b|"
         r"what do you .{0,25}\bit for\b"),
        ("ownership_tenure", "ownership tenure (how long they have had it)",
         r"(?i)how long (have|has) you|how long .{0,25}(had|owned|kept|been using)"),
        # NOTE: must NOT match "what do you mainly use X for?" -- that is
        # primary_use_case. Requires a place/occasion cue (when/where/which
        # situations), never a bare "use ... for".
        # change for b2c questionarie -- CHECK (live, UK vitamins 2026-09-09):
        # this only accepted the verbs "use/wear", so the model's perfectly
        # correct "Where do you usually TAKE your daily vitamins?" was
        # rejected four times and the slot was reported permanently missing.
        # The consumption verb is category-specific (take / drink / apply /
        # wash with / put on), so matching a fixed verb list is brittle by
        # design. Anchor on the place/occasion cue plus "usually", which is
        # what actually makes it a usage-context question, and keep the
        # negative lookahead so "what do you mainly use it FOR" still belongs
        # to primary_use_case.
        # The cue that separates this from the Section 2 "where did you buy /
        # where did you look" questions is TENSE, not the verb: usage context
        # is habitual present ("where/when do you USUALLY ..."), purchase is
        # a past event ("where DID you ..."). Verified against both.
        # Separating this from Section 2's "where" questions needs BOTH cues:
        #  - habitual present ("do you usually"), not past ("did you"), and
        #  - not a shopping verb -- "where do you usually BUY" is the
        #    acquisition channel, not usage context.
        ("usage_context", "usage context (situations/occasions/places)",
         r"(?i)(\b(when|where)\b(?!.{0,40}\bmainly\b)"
         r"(?!.{0,45}\b(buy|bought|purchase|shop|order|sign up|subscribe)\b)"
         r".{0,30}\bdo you\b.{0,25}\busually\b"
         r"|\b(which|what|in which)\b.{0,15}\b(situations?|occasions?|places?|settings?)\b)"),
        # Widened after the UK run: "Which FORM of vitamins do you use?" is
        # exactly this theme; "form" and "flavour/strength" were missing.
        ("item_variant", "item variant (which format/size/type they use)",
         r"(?i)(which|what) (type|kind|format|form|size|version|variant|"
         r"style|strength|flavour|flavor)\b|"
         # change for b2c questionarie -- CHECK (live, UK vitamins
         # 2026-09-09): the model twice wrote a correct variant question as
         # "Which of these best MATCHES/DESCRIBES the type you take?" and
         # both were rejected, leaving the slot permanently unfilled.
         r"which (of these |one of these )?best (matches|describes)\b|"
         # "which ... do you use" must NOT swallow "in which SITUATIONS do
         # you use it" -- that is usage_context. Exclude occasion words.
         r"which (?!.{0,20}\b(situations?|occasions?|places?|settings?)\b)"
         r".{0,30}\bdo you (use|have|own|take|choose)\b"),
    ),
    # --- Section 2: 7 questions ------------------------------------------
    "buying_behavior": (
        # Widened after the Japan run: "What made you first decide you needed
        # a device?" is a valid trigger question -- allow more words between
        # "you" and the verb, and accept "need/needed/start".
        ("acquisition_trigger", "acquisition trigger (what made them buy)",
         r"(?i)(what (made|makes)|why did|what prompted) you\s+"
         r"(\w+\s+){0,5}?(buy|get|choose|purchase|need|needed|start)"),
        ("shopping_timeframe", "shopping timeframe (how long they researched)",
         r"(?i)how long (did|do) you .{0,30}(spend|take|research|look|compar)|"
         r"how much time .{0,25}(research|decid|compar|look)"),
        # change for b2c questionarie -- CHECK (live, Japan health devices
        # 2026-09-09): the old second branch matched any question containing
        # "information ... before", so the SHOPPING TIMEFRAME question ("How
        # long did you look for information before buying?") also satisfied
        # this slot. One question covered two themes, the section looked
        # complete at 6/7, and the real information-sources gap was hidden.
        # This theme is about WHERE they looked, so it must name a source or
        # ask "where" -- never merely mention the word "information".
        ("information_sources", "information sources (where they learned about it)",
         r"(?i)where (did|do) you\s+(\w+\s+){0,3}?"
         r"(look|learn|read|research|find out|go to (learn|find))|"
         r"(which|what)\s+(sources?|websites?|places?)\b|"
         r"(where|which|what).{0,30}(reviews?|advice|recommendations?)\b"),
        # Widened after the UK run: a SUBSCRIPTION is "signed up for", not
        # "bought", so "Where did you sign up for your subscription?" is the
        # channel question for this category and must match.
        ("acquisition_channel", "acquisition channel (where they bought it)",
         r"(?i)where (did|do) you\s+(\w+\s+){0,3}?"
         r"(buy|get|purchase|order|shop|sign up|subscribe|join)"),
        ("price_paid", "price paid (how much they spent)",
         r"(?i)how much did you (spend|pay)|how much .{0,20}(spend|pay)\b"),
        ("top_decision_driver", "top decision driver (what mattered most)",
         r"(?i)(matter(s|ed)? most|most important|biggest influence|"
         r"single most)"),
        # Widened after the Japan run: "Before buying, what other ways to
        # handle your health did you think about?" is exactly this theme but
        # used "other ways" + "think about" rather than "consider".
        # change for b2c questionarie -- CHECK (live, UK vitamins 2026-09-09):
        # the old second branch matched "look(ed) at ... before", so the
        # SHOPPING TIMEFRAME question ("How long did you look at different
        # subscriptions before you picked one?") was counted as this theme.
        # The slot then looked filled and the guarantee never topped it up,
        # hiding a real gap. This theme needs an explicit ALTERNATIVES
        # concept -- other options/ways/brands/none-of-these -- not merely
        # the words "look at ... before".
        ("alternative_consideration", "alternative consideration set (what else they considered)",
         r"(?i)(what else|which other|other (options|ways|choices|kinds|"
         r"products?|brands?|methods?)|alternatives?|instead of|"
         r"anything else)\b|"
         r"(consider(ed)?|think about|thought about|compare[d]?)\b"
         r".{0,45}\b(other|alternatives?|besides|instead|else)\b"),
    ),
    # --- Section 3: 6 questions ------------------------------------------
    "preferences_expectations": (
        # Widened after the Japan run: "How happy are you with how well your
        # device gives accurate readings?" is core-function satisfaction, but
        # was being claimed by overall_satisfaction instead. The "with how
        # well" cue is what distinguishes the two: overall satisfaction is
        # about the product as a whole, this one is about how it PERFORMS.
        ("core_function_satisfaction", "core function satisfaction (rating of main capabilities)",
         r"(?i)how (well|satisf).{0,40}(work|perform|do|does|job)|"
         r"rate .{0,30}(performance|how well)|"
         r"how (happy|satisf)\w*\s+are you\s+with how well\b"),
        ("usability_ux", "usability / ease of use",
         r"(?i)(easy|easier|difficult|hard|simple|straightforward)\b.{0,30}"
         r"(to use|to set up|to operate|to figure)|how easy\b"),
        ("quality_signal", "quality signal (what tells them it is well made)",
         r"(?i)(well made|good quality|high quality|"
         r"tells you .{0,40}(quality|work|made|good|last)|"
         r"how (can|do) you tell .{0,40}(good|quality|work|last|made))"),
        ("essential_vs_nonessential", "essential vs non-essential features",
         r"(?i)(must|need to) (a |an )?.{0,30}(have|do|include)|"
         r"what (features|must)\b|which features|"
         r"(could not|couldn't|cannot) (do without|live without)|rarely use"),
        ("value_for_money", "perceived value for money",
         r"(?i)(value for money|worth (the|what)|worth paying|"
         r"good value|for (the|what) you paid)"),
        # change for b2c questionarie -- CHECK (live, UK vitamins 2026-09-09):
        # only three literal phrasings were accepted, so the natural
        # "better or worse than you expected" and "compare WITH what you
        # expected" both failed. Anchor on the word "expect" plus any
        # comparison cue instead of fixed phrases.
        ("expectation_match", "expectation match (performance vs expected)",
         r"(?i)(as (well as )?you expected|compare[ds]?\s+(to|with)\s+"
         r".{0,25}expect|meet .{0,25}expectations?|live(d)? up to|"
         r"(better|worse|more|less)\s+.{0,25}than you expected|"
         r"than (you )?expected)"),
    ),
    # --- Section 4: 5 questions ------------------------------------------
    "satisfaction_future_intent": (
        # Must NOT swallow core_function_satisfaction ("how happy are you
        # with HOW WELL it works") -- that is a Section 3 theme. The negative
        # lookahead keeps the two apart; without it the broader pattern wins
        # by table order and Section 3's slot is reported missing.
        ("overall_satisfaction", "overall satisfaction",
         r"(?i)how (satisf|happy are you|pleased are you)"
         r"(?!.{0,25}\bwith how well\b)"),
        # Widened after the Japan run: "What bothers you most about X?" is
        # the pain-point question but the old pattern required the word
        # "problem" to appear before "bothers".
        ("top_pain_point", "top pain point (biggest frustration/unmet need)",
         r"(?i)(biggest (problem|issue|frustration)|most often disappoints|"
         r"disappoints?\b|problem .{0,25}(bothers|most)|"
         r"most frustrating|what annoys|bothers you (the )?most)"),
        ("switching_trigger", "switching trigger (what would make them switch away)",
         r"(?i)(would make you switch|make you (switch|stop|change)|"
         r"switch to (a |an )?(different|another)|stop using)"),
        ("future_intent", "future acquisition intent (likely to buy the same again)",
         r"(?i)how likely are you to (buy|purchase|get)|"
         r"would you buy .{0,25}again|buy .{0,20}same .{0,20}again"),
        ("category_advocacy", "category advocacy (recommend, 0-10 scale)",
         r"(?i)(recommend|tell a friend|tell .{0,15}(others|someone))"),
    ),
}


def _canonical_tab(q: dict) -> str:
    """Canonical section id for a question, falling back to its raw tab."""
    return (q.get("beat_canonical") or q.get("tab") or "").strip()


def find_missing_required_slots(answered: list, section_ids: dict) -> dict:
    """Which required question types are MISSING, per canonical section.

    ``section_ids`` maps canonical id -> the market's actual section id, so a
    renamed section still resolves. Returns
    {canonical_id: [(slot_key, human_label), ...]} for sections that are
    short. Shared by validator_critic() (to force a regeneration with
    explicit feedback) and assembler.py's zero-tolerance gate (so a pass
    missing a required question can never be crowned "best").
    """
    by_canon: dict[str, list] = {}
    _substituted: dict[str, set] = {}
    for q in answered:
        if (q.get("question_layer") or "") in ("screening", "profiling"):
            continue
        canon = _canonical_tab(q)
        # Map a market-specific section id back to its canonical id.
        for cid, market_id in (section_ids or {}).items():
            if canon == market_id:
                canon = cid
                break
        by_canon.setdefault(canon, []).append(q.get("text") or "")
        # change for b2c questionarie -- CHECK (user directive, 2026-09-09):
        # a theme the model deliberately substituted (because forcing it
        # would produce a useless question for this category) is NOT a gap.
        # Without this the gate would demand the bad question back and the
        # run would loop until the revision cap.
        _sub = q.get("_substituted_for")
        if _sub:
            _substituted.setdefault(canon, set()).add(_sub)

    missing: dict[str, list] = {}
    for canon, slots in REQUIRED_SLOTS.items():
        stems = by_canon.get(canon) or []
        subs = _substituted.get(canon) or set()
        gaps = [
            (key, label) for key, label, pat in slots
            if label not in subs and not any(re.search(pat, s) for s in stems)
        ]
        if gaps:
            missing[canon] = gaps
    return missing


def find_offtheme_questions(answered: list, section_ids: dict) -> dict:
    """Questions that match NO required slot for the section they sit in.

    # change for b2c questionarie -- CHECK (user directive, 2026-09-09):
    # find_missing_required_slots() only checks that every listed theme is
    # PRESENT. It says nothing about extra questions, so a section could
    # satisfy all its slots and still carry an off-theme question -- which
    # is how surveys kept picking up questions the framework never asked
    # for. The manager signed off on these themes specifically, so the slot
    # list is closed: anything outside it is a defect.
    #
    # A question is off-theme when it matches none of ITS OWN section's slot
    # patterns. Matching another section's pattern is reported too (it means
    # the question is filed in the wrong section), because that is a more
    # useful message for the architect than a bare "unlisted theme".
    Returns {canonical_id: [(question_text, note), ...]}.
    """
    by_canon: dict[str, list] = {}
    for q in answered:
        if (q.get("question_layer") or "") in ("screening", "profiling"):
            continue
        canon = _canonical_tab(q)
        for cid, market_id in (section_ids or {}).items():
            if canon == market_id:
                canon = cid
                break
        # change for b2c questionarie -- CHECK (user directive, 2026-09-09):
        # a deliberate substitute is expected NOT to match any slot pattern,
        # so it must be exempt here or the off-theme gate would reject the
        # very question the substitution rule asked for.
        if q.get("_substituted_for"):
            continue
        by_canon.setdefault(canon, []).append(q.get("text") or "")

    offtheme: dict[str, list] = {}
    for canon, stems in by_canon.items():
        own = REQUIRED_SLOTS.get(canon)
        if not own:
            continue  # unknown section (e.g. profiling) -- not gated here
        for stem in stems:
            if any(re.search(pat, stem) for _k, _l, pat in own):
                continue
            # Does it belong to a DIFFERENT section's slot list?
            belongs_to = next(
                (
                    other
                    for other, slots in REQUIRED_SLOTS.items()
                    if other != canon
                    and any(re.search(pat, stem) for _k, _l, pat in slots)
                ),
                None,
            )
            note = (
                f"belongs in section '{belongs_to}', not here"
                if belongs_to else
                "matches none of this section's required themes"
            )
            offtheme.setdefault(canon, []).append((stem, note))
    return offtheme


def find_nps_shape_issue(answered: list) -> str | None:
    """The advocacy/NPS question must be an 11-point 0-10 scale.

    Shape-detected the same way assembler._is_rating_0_10 does it (that is
    what actually turns it into rating_0_10 at publication), so a 5-option
    "Would you recommend...?" is caught here rather than shipping as an
    ordinary single_select.
    """
    for q in answered:
        if (q.get("question_layer") or "") in ("screening", "profiling"):
            continue
        text = q.get("text") or ""
        if not re.search(r"(?i)(recommend|tell a friend)", text):
            continue
        opts = q.get("options") or []
        if len(opts) == 11 and str(opts[0]).strip().startswith("0") \
                and str(opts[-1]).strip().startswith("10"):
            return None  # correct shape found
        return (
            f"advocacy/NPS question must be an 11-option 0-10 scale "
            f"(\"0 - Not at all likely\" ... \"10 - Extremely likely\"), "
            f"got {len(opts)} options: {text[:70]}"
        )
    return None


# Soft semantic dedup — higher threshold + segment-aware tokens so on-topic
# questions aren't mis-flagged just for sharing the segment name (BUG 5).
# change for b2c questionarie — thresholds live in question_similarity.py
_DEDUP_JACCARD = LOOSE_JACCARD
_MIN_SHARED_TOKENS = 2
# change for b2c questionarie — floor from QUESTIONS_PER_TAB_MIN × 4 (was 90).
_MIN_TOTAL_QUESTIONS = MIN_TOTAL_QUESTIONS

_STOP = {
    "how", "what", "which", "who", "when", "where", "why", "do", "does", "you",
    "your", "the", "of", "for", "and", "are", "is", "to", "in", "on", "with",
    "most", "often", "please", "select", "all", "that", "apply", "following",
    "best", "describes", "would", "have", "any", "this", "these", "from", "buy",
}

# Geography of residence is a UI filter, not a survey instrument.
_GEO_RESIDENCE_RE = re.compile(
    r"(?i)\b("
    r"which\s+region\s+do\s+you\s+(live|reside)|"
    r"in\s+which\s+region\s+do\s+you\s+(currently\s+)?(live|reside)|"
    r"where\s+do\s+you\s+(currently\s+)?(live|reside)|"
    r"what\s+(is\s+your|region\s+are\s+you\s+in)|"
    r"your\s+(current\s+)?(country|region)\s+of\s+residence|"
    r"where\s+are\s+you\s+(based|located)|"
    r"select\s+your\s+(region|country)"
    r")\b"
)


def _is_geo_residence_question(text: str) -> bool:
    return bool(_GEO_RESIDENCE_RE.search(text or ""))


# change for b2c questionarie — ban age / income demographics as survey items
# change for b2c questionarie — A16: Profiling already asks household make-up
# and gender at the end, for cross-tabs. A behavioural section asking the same
# thing spends a slot on a fact the study already has, and the two answers can
# disagree. Age and income are covered by _AGE_INCOME_RE below.
_PROFILING_DUPLICATE_RE = re.compile(
    r"(?i)("
    r"(who|what|which).{0,30}\b(else\s+)?(lives?|live)\s+(in|with)\s+your\s+(household|home)|"
    r"describe\s+your\s+household|"
    r"how\s+many\s+people\s+(live|are)\s+in\s+your\s+(household|home)|"
    r"what\s+is\s+your\s+gender|"
    r"household\s+(composition|make[- ]?up)"
    r")"
)


def _duplicates_profiling(text: str) -> bool:
    """True when a behavioural question re-asks a Profiling demographic."""
    return bool(_PROFILING_DUPLICATE_RE.search(text or ""))


_AGE_INCOME_RE = re.compile(
    r"(?i)\b("
    r"(what|which).{0,40}\b(age|age\s*group|age\s*band|how\s+old)\b|"
    r"\b(your|household'?s?|personal|annual|monthly)\s+(total\s+)?(income|salary|earnings)\b|"
    r"\bincome\s+(range|bracket|band|level|group)\b|"
    r"\bhow\s+old\s+are\s+you\b|"
    r"\bage\s+group\b|"
    r"\bhousehold'?s?\s+total\s+(annual\s+)?income\b"
    r")"
)

# change for b2c questionarie — ban brand / company name questions
_BRAND_COMPANY_RE = re.compile(
    r"(?i)\b("
    r"which\s+(brand|brands|company|companies|manufacturer|manufacturers|label|labels)\b|"
    r"what\s+(brand|brands|company|companies|manufacturer)\b|"
    r"(favorite|favourite|preferred|usual|current|main)\s+brand\b|"
    r"brand\s+(do\s+you|you\s+(buy|use|prefer|choose|purchase|trust))\b|"
    r"name\s+of\s+(the\s+)?(brand|company|manufacturer)\b|"
    r"from\s+which\s+(brand|company|manufacturer)\b|"
    r"which\s+of\s+these\s+(brands|companies|manufacturers)\b"
    r")"
)


def _is_age_or_income_question(text: str) -> bool:
    return bool(_AGE_INCOME_RE.search(text or ""))


def _is_brand_name_question(text: str, options: list | None = None) -> bool:
    if _BRAND_COMPANY_RE.search(text or ""):
        return True
    if re.search(r"(?i)\b(select|choose|pick)\s+(your\s+)?(brand|company)\b", text or ""):
        return True
    if re.search(r"(?i)\b(brand|company|manufacturer)\b", text or "") and options:
        brandish = sum(
            1 for o in options
            if re.match(r"^[A-Z][A-Za-z0-9&\-'.]{1,24}$", str(o or "").strip())
        )
        if brandish >= 3:
            return True
    return False


# CHECK 18b (B2C master rules, Section 27b/27c) -- the ban on brand/company
# names is not limited to a question ASKING about brands (_is_brand_name_
# question above); a channel/format option like "Online marketplace (e.g.,
# Shopee, Lazada)" is not about brands at all, but the parenthetical example
# still names real companies, breaking category-level neutrality just as
# much. Confirmed live: two options in a Running Shoes/China survey named
# Shopee, Lazada, Decathlon and Sports Direct as examples -- none of which
# even operate meaningfully in China, compounding the violation with a
# regional mismatch. Distinguishing a real proper-noun example list ("e.g.,
# Shopee, Lazada") from a legitimate attribute example list ("e.g., 4K,
# HDR" or "e.g., cushioning, durability") without a brand database: real
# company names are essentially never in this common-word list, whereas
# attribute/format examples usually are, or are all-caps acronyms/numbers.
_COMMON_PARENTHETICAL_WORDS = {
    "hdr", "4k", "hd", "uhd", "led", "oled", "usb", "wifi", "gps", "nfc",
    "cushioning", "durability", "comfort", "breathability", "support",
    "style", "color", "colour", "size", "fit", "weight", "price", "quality",
    "morning", "evening", "daily", "weekly", "monthly", "indoor", "outdoor",
    "cash", "card", "credit", "debit", "cotton", "leather", "mesh", "foam",
}


def _named_example_issue(options: list) -> str | None:
    """Flag an option whose "(e.g., X, Y)" example list looks like real
    company/brand/platform names rather than generic attribute examples."""
    for o in options or []:
        text = str(o or "")
        for m in re.finditer(r"\(e\.?g\.?,?\s*([^)]+)\)", text, re.IGNORECASE):
            items = [it.strip() for it in re.split(r",|/| and ", m.group(1)) if it.strip()]
            if len(items) < 2:
                continue
            proper_noun_like = sum(
                1 for it in items
                if re.match(r"^[A-Z][a-zA-Z''.\-]{2,20}(\s+[A-Z][a-zA-Z''.\-]{2,20})?$", it)
                # An all-caps token (UPI, COD, HDR) is almost always an
                # acronym/abbreviation, not a brand name -- a real brand in
                # title case ("Shopee") or Sentence case has at least one
                # lowercase letter after its first.
                and not it.replace(" ", "").isupper()
                and it.lower().split()[0] not in _COMMON_PARENTHETICAL_WORDS
            )
            if proper_noun_like >= 2:
                return (
                    f"option {text!r} names real companies/platforms as "
                    "examples -- the category-level rule bans brand, "
                    "company or platform names EVEN as a parenthetical "
                    "'e.g.' example; use a generic description instead "
                    "(\"Online marketplace\", \"Sports chain retailer\") "
                    "with no named examples at all"
                )
    return None


# change for b2c questionarie — ban matrix stems with scale-only options
_GRID_STEM_RE = re.compile(
    r"(?i)(each of the following|how important are each|rate the importance of each|"
    r"importance of each of|for each of the following)"
)
_SCALE_OPTION_HINTS = (
    "not at all important", "slightly important", "moderately important",
    "very important", "extremely important", "strongly disagree", "disagree",
    "agree", "strongly agree", "very dissatisfied", "dissatisfied",
    "satisfied", "very satisfied", "neutral",
)


def _is_broken_grid_question(text: str, options: list, qtype: str = "") -> bool:
    """True when a matrix stem cannot be answered by the options given.

    Two ways this breaks, and the original only caught the first:

    1. Stem promises a grid, options are ONLY a rating scale — there is nothing
       to rate.
    2. Stem promises a grid ("how satisfied are you with EACH of the following")
       and the options ARE the attribute list, but the question is typed
       single_choice — so it renders as "pick one", and the chart reads as a
       share when the stem asked for a rating per attribute.
    """
    if not _GRID_STEM_RE.search(text or ""):
        return False
    if not options or len(options) < 3:
        return False
    hits = 0
    for o in options:
        key = " ".join(str(o).lower().split())
        if any(h in key for h in _SCALE_OPTION_HINTS):
            hits += 1
    if hits >= max(3, len(options) - 1):
        return True
    # change for b2c questionarie — attribute options under a grid stem, asked
    # as a single pick. The answer format contradicts the question.
    return hits == 0 and qtype in ("single_choice", "ranking")


# change for b2c questionarie — numeric recall beyond respondent capacity.
# Asking a consumer to total up occasions or spend across a multi-week window
# produces an estimate, not a measurement: nobody counts their own workouts or
# app spend. Professional instruments ask relative frequency instead (Pew's
# "Always or almost always / Often / Sometimes / Rarely / Never"), so the answer
# is something a person can actually retrieve. A count is only acceptable when
# the number is small, salient and standing (how many services you PAY for right
# now), which is a stock question, not a recall-over-a-window question.
_RECALL_STEM_RE = re.compile(
    r"(?i)\b(how many|how much)\b"
)
# "How often … in the last 30 days" is fine when the options are a frequency
# scale — the window is just a framing cue and the respondent answers with a
# rate, not a tally. It only breaks when the options demand a count.
_RECALL_COUNT_OPTION_RE = re.compile(
    r"(?i)^\s*(\d[\d,]*\s*(-|–|to)\s*\d[\d,]*|\d[\d,]*\s*(or (more|fewer|less))?|"
    r"(more|less|fewer) than \d|under \d|over \d|[\$€£]\s?\d)"
)
_RECALL_WINDOW_RE = re.compile(
    r"(?i)("
    r"\b(in|over|during|within)\s+the\s+(last|past)\s+\d+\s*(day|days|week|weeks|month|months|year|years)\b|"
    r"\b(in|over|during)\s+the\s+(last|past)\s+(day|week|month|year|fortnight)\b|"
    r"\bin\s+total\b|"
    r"\baltogether\b|"
    r"\bper\s+(month|year|week)\b"
    r")"
)
# A standing state ("do you currently pay for") is recallable even with a count.
_RECALL_STOCK_RE = re.compile(
    r"(?i)\b(currently|right now|at the moment|do you (now )?(own|have|pay for|subscribe))\b"
)


def _is_numeric_recall_question(text: str, options: list | None = None) -> bool:
    """True when the stem demands a total the respondent cannot actually recall.

    Requires BOTH a counting stem ("how many"/"how much") and an explicit
    look-back window — "how many times did you X in the last 30 days". A bare
    "how many services do you pay for" is a stock question and passes, as does
    anything phrased around the respondent's current standing state.

    "How often … in the last 30 days" is deliberately NOT a counting stem: with
    a frequency scale for options it is answerable and common in professional
    instruments. It is caught here only when its options are numeric bands,
    which turns it back into a tally.
    """
    t = text or ""
    if not _RECALL_WINDOW_RE.search(t):
        return False
    if _RECALL_STOCK_RE.search(t):
        return False
    if _RECALL_STEM_RE.search(t):
        return True
    # "how often" + numeric-band options == a count wearing a frequency label.
    if re.search(r"(?i)\bhow often\b", t) and options:
        numeric = sum(
            1 for o in options
            if _RECALL_COUNT_OPTION_RE.match(str(o or "").strip())
        )
        return numeric >= max(2, len(options) - 1)
    return False


# change for b2c questionarie — scale polarity must be consistent survey-wide.
# The architect prompt already says "use the same direction across the survey",
# but nothing enforced it, so satisfaction ran negative-first while likelihood
# ran positive-first in the same instrument. Mixed polarity is a real data
# defect: a respondent who has learned the left edge means "bad" mis-clicks when
# a later scale silently flips, and top-2-box comparisons across sections stop
# being like-for-like.
_POLARITY_NEGATIVE_HEADS = (
    "very dissatisfied", "extremely dissatisfied", "very unlikely",
    "extremely unlikely", "very unimportant", "not at all important",
    "not at all likely", "not at all satisfied", "strongly disagree",
    "much worse", "very poor", "never",
)
_POLARITY_POSITIVE_HEADS = (
    "very satisfied", "extremely satisfied", "very likely", "extremely likely",
    "very important", "extremely important", "strongly agree", "much better",
    "excellent", "always", "always or almost always",
)


def _scale_polarity(options: list) -> str | None:
    """Return 'neg_first' / 'pos_first' for an ordinal scale, else None.

    Only fires on genuine ordered scales — categorical lists that happen to
    start with a scale-ish word return None and are left alone.
    """
    if not options or len(options) < 4:
        return None
    head = " ".join(str(options[0] or "").lower().split())
    tail = " ".join(str(options[-1] or "").lower().split())
    starts_neg = any(head.startswith(h) for h in _POLARITY_NEGATIVE_HEADS)
    ends_pos = any(tail.startswith(h) for h in _POLARITY_POSITIVE_HEADS)
    starts_pos = any(head.startswith(h) for h in _POLARITY_POSITIVE_HEADS)
    ends_neg = any(tail.startswith(h) for h in _POLARITY_NEGATIVE_HEADS)
    if starts_neg and ends_pos:
        return "neg_first"
    if starts_pos and ends_neg:
        return "pos_first"
    return None


# change for b2c questionarie — respondent-facing text must be plain consumer
# language. Research vocabulary ("attributes", "purchase journey") is how the
# trade talks to itself, not how you ask a shopper a question; passive voice and
# throat-clearing openers ("Thinking about your overall experience,") make a
# stem longer and harder without adding meaning.
_JARGON_RE = re.compile(
    r"(?i)\b("
    r"purchase\s+journey|decision[- ]driver|touch\s?point|consideration\s+set|"
    r"value\s+proposition|pain\s+point|attribute|criteri(?:a|on)|"
    r"dimension|demographic|utilis|utiliz|leverage|engagement|ecosystem|"
    r"omni[- ]?channel|end[- ]user|consumer\s+segment|purchase\s+occasion|"
    r"decision[- ]making\s+process|brand\s+equity|share\s+of\s+wallet"
    r")\b"
)
# "is/are/was/were <past participle> by" — the passive that matters here.
_PASSIVE_RE = re.compile(
    # Allow words between the auxiliary and the participle ("are running shoes
    # purchased by you"), which is exactly how an inverted question phrases it.
    r"(?i)\b(is|are|was|were|been|being)\b[^?.!]{0,40}?\b\w+(?:ed|en)\s+by\b"
)
_THROAT_CLEARING_RE = re.compile(
    r"(?i)^\s*("
    r"thinking\s+(about|back|now)|in\s+your\s+opinion|generally\s+speaking|"
    r"considering\s+(your|the)|reflecting\s+on|with\s+regard\s+to|"
    r"when\s+it\s+comes\s+to\s+your\s+overall|as\s+you\s+think\s+about|"
    r"when\s+considering|if\s+you\s+think\s+about"
    r")\b"
)
# change for b2c questionarie -- a leading timeframe clause ("In the past
# year,", "In the last 12 months,") adds nothing when the question is really
# asking about a standing habit. Compare with a comparison stem
# ("Compared to a year ago...") which IS analytically required when the point
# is change over time -- that one is a genuine construct, not filler, so it is
# deliberately NOT in this pattern.
_UNNECESSARY_TIMEFRAME_LEAD_RE = re.compile(
    r"(?i)^\s*(in\s+the\s+(past|last)\s+(year|\d+\s*(months?|weeks?|days?)),?\s*"
    r"|over\s+the\s+(past|last)\s+(year|\d+\s*(months?|weeks?)),?\s*)"
)


# change for b2c questionarie — a stem with a missing head noun. The last run
# shipped "Which must a smartwatch do for you to consider buying it?" and "What
# was the most important when you chose your smartwatch?" — both are missing a
# word and read as broken English in a client deliverable.
_BROKEN_STEM_RE = re.compile(
    r"(?i)^\s*("
    r"which\s+(must|should|does|do|can|will)\b|"          # "Which must a X do…"
    r"what\s+was\s+the\s+most\s+\w+\s+when\b|"            # "What was the most important when…"
    r"what\s+is\s+the\s+most\s+\w+\s+when\b|"
    r"which\s+of\s+(must|should)\b"
    r")"
)
# The audience definition already establishes the respondent buys the category,
# so asking who decides can only contradict it.
_DECIDER_RE = re.compile(
    # "Who decided which X to buy?" slipped past the household-only form.
    r"(?i)\bwho\s+(else\s+)?(in\s+your\s+(household|home)\s+)?(usually\s+)?"
    r"(decides?|decided|chooses?|chose|buys?|bought|picks?|picked|selects?)\b"
)


# change for b2c questionarie — measured readability, not a guess. textstat
# gives a Flesch-Kincaid grade level; a consumer questionnaire that reads above
# roughly 9th grade is asking people to parse a sentence instead of answering a
# question. Optional import so a missing package degrades to the regex rules
# rather than breaking generation.
try:  # pragma: no cover - availability differs per environment
    import textstat as _textstat
except Exception:  # noqa: BLE001
    _textstat = None

_MAX_READING_GRADE = float(os.getenv("MAX_READING_GRADE", "9"))


def _readability_issue(text: str) -> str | None:
    """Return a reason when a stem reads harder than a consumer survey should."""
    t = (text or "").strip()
    if not _textstat or len(t.split()) < 6:
        return None
    try:
        grade = _textstat.flesch_kincaid_grade(t)
    except Exception:  # noqa: BLE001
        return None
    if grade > _MAX_READING_GRADE:
        return (
            f"reads at about grade {grade:.0f} — too hard for a consumer "
            "questionnaire; shorten the sentence and use commoner words"
        )
    return None


# change for b2c questionarie — Issue 1. An India survey shipped 106 dollar
# symbols alongside 62 rupee ones. The prompt asks for the local currency, but
# a prompt is not a guarantee, so wrong-currency options are flagged in code.
_CURRENCY_SYMBOLS = ("$", "€", "£", "₹", "¥", "R$", "C$", "A$", "kr", "SAR", "AED")

# change for b2c questionarie -- CHECK (Section 30b): a question about what
# someone PAID/PAYS and a question about what they EXPECT/WOULD pay for the
# same product may legitimately be two distinct questions (actual habitual
# spend vs aspirational/expected price), but if their price bands don't
# align, the two findings can never be reconciled against each other in
# analysis -- confirmed live: "Under Y300/Y300-599/.../Y1200+" vs "Under
# Y200/Y200-399/.../Y800+" on the same product category, same currency.
_PRICE_POINT_STEM_RE = re.compile(
    r"(?i)\b(how much (do|would|did) you (usually |typically |)"
    r"(pay|spend)|what (do|would|did) you (usually |typically |)"
    r"(pay|spend|expect to pay)|"
    # covers the indirect "how would you describe what you expect to pay"
    # wrapper too, so this fires even before/regardless of the padded-
    # opener cleanup (CHECK Section 30c) simplifying the wording.
    r"describe what you (expect|plan|intend) to (pay|spend))\b"
)
_AMOUNT_RE = re.compile(
    r"[\d][\d,]*(?:\.\d+)?"
)


def _extract_amounts(options: list) -> list:
    """All numeric amounts across an option list, in appearance order."""
    return [
        int(m.replace(",", "").split(".")[0])
        for o in (options or [])
        for m in _AMOUNT_RE.findall(str(o or ""))
    ]


def _price_band_mismatch_issue(qid: str, options: list, seen: dict) -> str | None:
    """Flag a second price-point question whose bands don't match the first."""
    amounts = _extract_amounts(options)
    if len(amounts) < 3:
        return None
    if not seen:
        seen[qid] = amounts
        return None
    first_id, first_amounts = next(iter(seen.items()))
    if amounts == first_amounts:
        return None
    return (
        f"prices the same purchase using different bands than "
        f"{first_id} ({amounts} vs {first_amounts}) -- if this is a "
        "genuinely distinct question (actual spend vs expected price), "
        "align the band boundaries so the two findings can be compared; "
        "if it tests the same construct, merge them instead"
    )


# change for b2c questionarie -- CHECK (Section 29a, adopted from the
# external India-skincare reference build): "how much do you spend on X"
# with no stated period is not answerable consistently -- one respondent
# might read it as per-purchase, another as per-month, another as per-year.
# A ONE-TIME purchase price question ("how much did you pay for your LAST
# pair") is inherently unambiguous and does not need this -- only a
# recurring/aggregate SPEND question does.
_RECURRING_SPEND_STEM_RE = re.compile(
    r"(?i)\bhow much (do|would) you (usually |typically |)spend\b"
)
_STATED_PERIOD_RE = re.compile(
    r"(?i)\b(per|each|every)\s+(day|week|month|year)|"
    r"\b(daily|weekly|monthly|yearly|annually)\b|"
    r"\bin\s+(a|one)\s+(typical\s+)?(week|month|year)\b|"
    r"\bover\s+the\s+(past|last)\s+(\d+\s*)?(days?|weeks?|months?|years?)\b|"
    r"\bfor\s+(a|one)\s+(single\s+)?(purchase|pair|item)\b"
)


def _spend_missing_period_issue(text: str) -> str | None:
    """Flag a recurring-spend question with no stated time period."""
    if not _RECURRING_SPEND_STEM_RE.search(text or ""):
        return None
    if _STATED_PERIOD_RE.search(text or ""):
        return None
    return (
        "asks how much the respondent spends with no stated time period -- "
        "make clear whether this is per purchase, per month, per year, or "
        "another stated window; an unqualified 'how much do you spend' is "
        "not answerable consistently across respondents"
    )


# change for b2c questionarie -- CHECK (Section 30, currency): a CORE-layer
# question is cached ONCE per segment and reused across every country in a
# region (its cache key is literal "core", never geography-scoped) -- ANY
# hardcoded currency symbol in a core option is wrong for every country
# except (by coincidence) whichever one triggered the cache miss that
# generated it. Confirmed live: a core spend question hardcoded USD bands
# into a China-region file. This is stricter than _currency_issue (which
# only flags a MISMATCH against the expected currency) because at the core
# layer there is no such thing as a correct hardcoded currency -- the
# layer's whole design requires relative/currency-neutral framing.
def _core_layer_currency_issue(options: list) -> str | None:
    """Flag ANY currency symbol in a core-layer question's options."""
    joined = " ".join(str(o or "") for o in (options or []))
    if not joined:
        return None
    for sym in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True):
        if sym in joined:
            return (
                f"is a CORE-layer question (cached and reused across every "
                f"country) but hardcodes the currency symbol {sym!r} -- core "
                "money questions must use relative price framing "
                "(Budget/Mid-range/Premium) with no currency symbol at all; "
                "a concrete currency-denominated band belongs only in the "
                "regional MODULE layer"
            )
    return None


def _currency_issue(options: list, expected: str) -> str | None:
    """Flag options priced in a currency this market does not use."""
    if not expected:
        return None
    joined = " ".join(str(o or "") for o in (options or []))
    if not joined:
        return None
    # Longest-first so "R$"/"C$"/"A$" are not mistaken for a bare "$".
    for sym in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True):
        if sym in joined and sym != expected:
            if sym == "$" and expected in ("R$", "C$", "A$"):
                continue  # the expected symbol contains a dollar sign
            return (
                f"prices options in {sym} but this market uses {expected} — "
                "rewrite the amounts in the local currency at local price levels"
            )
    return None


# change for b2c questionarie — Issue 9c. Subject-verb agreement, checked where
# it actually slips: a plural subject several words from its verb.
# "Which aspects of your current skincare products most satisfies you?"
_SV_DISAGREE_RE = re.compile(
    r"(?i)\b(aspects|features|options|factors|products|things|claims|reasons|"
    r"problems|benefits|frustrations|concerns|qualities)\b[^?.!]{0,40}?\b"
    r"(satisfies|matters|makes|helps|appeals|drives|influences|bothers|works|"
    r"disappoints|annoys|frustrates|delights|convinces)\b"
)
# Issue 9b — "colleague" is a workplace NPS convention. Personal-care and
# household categories are recommended to friends and family, not colleagues.
_PERSONAL_CATEGORY_RE = re.compile(
    r"(?i)\b(skin ?care|hair ?care|cosmetic|beauty|shampoo|moisturis|moisturiz|"
    r"serum|fragrance|deodorant|toothpaste|grocer|food|beverage|snack|milk|"
    r"apparel|clothing|footwear|shoe|mattress|furniture|pet ?food)\b"
)


def _register_issue(text: str, segment: str) -> str | None:
    """Flag a workplace recommendation frame on a personal-life category."""
    t = text or ""
    if "colleague" not in t.lower():
        return None
    if _PERSONAL_CATEGORY_RE.search(f"{t} {segment}"):
        return ("asks about recommending to a \"colleague\" — a workplace frame; "
                "for a personal or household product say \"a friend or family "
                "member\"")
    return None


def _agreement_issue(text: str) -> str | None:
    """Flag plural-subject / singular-verb disagreement in a stem."""
    if _SV_DISAGREE_RE.search(text or ""):
        return ("has a plural subject with a singular verb — read it aloud and "
                "fix the agreement (\"aspects ... satisfy\", not \"satisfies\")")
    return None


# change for b2c questionarie -- Issue 9a. A single-select list must answer
# ONE dimension of the stem. "When do you apply it?" mixing TIME-OF-DAY options
# ("morning only", "evening only") with OCCASION options ("only before going
# out", "only when skin feels dry") is not mutually exclusive -- someone who
# only applies before going out could do that in the morning or evening.
_TIME_OF_DAY_OPT_RE = re.compile(
    r"(?i)\b(morning|evening|night|afternoon)\b"
)
_OCCASION_OPT_RE = re.compile(
    r"(?i)\b(only when|only before|only after|whenever|as needed|"
    r"when i remember|when my skin|before going out|before bed)\b"
)


# change for b2c questionarie -- the audience definition qualifies every
# respondent as buying THEIR OWN skin care / product, so an option implying
# they buy exclusively for someone else contradicts the screener the same way
# a "non-user" option does. "Children in my household only" on "who do you buy
# for" is the same class of bug as offering "I don't use this" on a usage
# question, just phrased as an exclusive third-party purchase instead.
_EXCLUSIVE_OTHER_PERSON_RE = re.compile(
    r"(?i)^\s*(children|kids|my (partner|spouse|child|kids)|"
    r"(other|another) (person|family member|adult))s?\s+"
    r"(in my household\s+)?only\s*$"
)


# change for b2c questionarie -- two questions can word their STEMS
# completely differently ("which products have you used" vs "which formats are
# you willing to use") while their OPTION LISTS are the same underlying
# category rundown (cleanser, serum, mask...). Text-similarity on the stem
# alone misses this; catch it on the option sets instead.
def _singularize(word: str) -> str:
    """Crude stem so 'cleansers'/'cleanser' and 'masks'/'mask' compare equal."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _option_overlap_issue(qid: str, options: list, seen: dict) -> str | None:
    """Flag a second question whose option list heavily overlaps an earlier one."""
    toks = frozenset(
        _singularize(w)
        for o in (options or []) for w in _sim_tokens(str(o or ""))
    )
    if len(toks) < 3:
        return None
    for prior_id, prior_toks in seen.items():
        if not prior_toks:
            continue
        overlap = len(toks & prior_toks) / max(1, min(len(toks), len(prior_toks)))
        if overlap >= 0.5:
            return (
                f"has an option list that overlaps {prior_id}'s by "
                f"{overlap:.0%} -- even though the wording differs, this is "
                "the same category list asked twice; keep the more useful "
                "framing and drop this one"
            )
    seen[qid] = toks
    return None


def _audience_contradiction_issue(text: str, options: list) -> str | None:
    """Flag an option implying the respondent isn't the buyer/user at all."""
    if not re.search(r"(?i)who.{0,45}(buy|use).{0,15}for\b", text or ""):
        return None
    for o in options or []:
        if _EXCLUSIVE_OTHER_PERSON_RE.match(str(o or "")):
            return (
                f"offers {str(o)!r}, which means the respondent buys this "
                "exclusively for someone else -- the audience definition "
                "qualifies every respondent as buying for themselves, so this "
                "option contradicts the screener; drop it or add 'for myself' "
                "to it"
            )
    return None


def _mixed_dimension_issue(options: list) -> str | None:
    """Flag a single-select list combining two incompatible answer dimensions."""
    opts = [str(o or "") for o in (options or [])]
    if len(opts) < 3:
        return None
    time_hits = sum(1 for o in opts if _TIME_OF_DAY_OPT_RE.search(o))
    occ_hits = sum(1 for o in opts if _OCCASION_OPT_RE.search(o))
    if time_hits >= 1 and occ_hits >= 1:
        return (
            "mixes WHEN-of-day options with WHY/occasion options in one list -- "
            "these are not mutually exclusive (someone who applies 'before going "
            "out' could do that in the morning or evening); split into two "
            "questions"
        )
    return None


# Issue 9e / CHECK 36 (B2C_SURVEY master rules, Section 25b) -- a multi-select
# question about must-haves, requirements, concerns, features sought, or
# preferences needs a genuine opt-out, or a low-involvement respondent has no
# honest answer. The stem list was originally scoped to "must have/be" only
# and missed plain "which features do you look for" / "concerns" / "matters
# to you" phrasing -- confirmed missing on the Running Shoes (China) survey's
# "Which running shoe features do you look for..." question, which had no
# opt-out at all despite testing the same must-have/preference construct.
# change for b2c questionarie -- two rounds of "which X must/problems ..."
# missed real cases because an adjective sat between "which" and the keyword
# ("which FEATURES must", "which FIT problems"). Widened every "which ..."
# branch to tolerate 0-20 characters of anything before the keyword, so the
# keyword's POSITION in the sentence no longer matters, only its presence.
_MUSTHAVE_STEM_RE = re.compile(
    r"(?i)\b("
    r"which .{0,20}\bmust\b|"
    r"which .{0,20}(matter|required|need to have)|"
    r"(features?|attributes?) (do|would) you (look for|want|expect)|"
    r"(biggest |main |top )?concerns? (do you have|about)|"
    r"which .{0,20}(preferences?|requirements?)\b|"
    r"which .{0,20}(problems|issues|frustrations) have you (had|experienced)|"
    r"what .{0,20}problems have you (had|experienced)"
    r")\b"
)


def _missing_optout_issue(text: str, qtype: str, options: list) -> str | None:
    """Flag a must-have list with no 'none of these matter to me' option."""
    if qtype != "multiple_choice":
        return None
    if not _MUSTHAVE_STEM_RE.search(text or ""):
        return None
    opts = [str(o or "").lower() for o in (options or [])]
    if any("none of these" in o or "not important to me" in o for o in opts):
        return None
    return (
        "is a must-have list with no opt-out -- add 'None of these matter to "
        "me' so a low-involvement respondent has an honest answer"
    )


# change for b2c questionarie -- Issue 4/9d. Two questions sharing this exact
# template ("Which must X do/be...") differed in type (single vs multi-select)
# with nothing in the text to signal it -- caught by comparing type against
# the shared-template group, not by wording alone.
_MUST_TEMPLATE_RE = re.compile(
    r"(?i)^which must (a|an|your) [\w -]+ (do|be) "
)


# CHECK (B2C master rules, Section 26i) -- a multi-select question's options
# are independent select-all rates (each is its own probability), so the
# total across options is not meaningful the way a single_choice total is
# and normally lands well over 100% once real respondents pick more than
# one. A multi-select total landing at/near 100% while sibling multi-select
# questions in the SAME survey clearly exceed 100% is the tell that this one
# question's percentages were generated as a constrained share (single_choice
# habit) rather than independent probabilities -- confirmed on two questions
# in the Running Shoes (China) survey while others correctly totalled
# 230%/236%.
_NEAR_100_TOLERANCE = 3.0    # percentage points either side of 100
_CLEAR_EXCESS_THRESHOLD = 115.0  # a sibling must clearly exceed 100 to anchor the comparison


def _suspicious_multiselect_sum_issue(
    qid: str, total: float, sibling_totals: dict,
) -> str | None:
    """Flag qid's multi-select total if it lands near 100% while at least one
    OTHER multi-select question in the same survey clearly exceeds 100%."""
    if abs(total - 100.0) > _NEAR_100_TOLERANCE:
        return None
    others = [t for other_qid, t in sibling_totals.items() if other_qid != qid]
    if not any(t >= _CLEAR_EXCESS_THRESHOLD for t in others):
        return None
    return (
        f"sums to {total:.1f}%, suspiciously close to 100%, while other "
        "multi-select questions in this survey clearly exceed 100% -- "
        "re-generate this question's percentages as independent per-option "
        "select-all rates, not a constrained 100% share"
    )


# change for b2c questionarie -- a narrow micro-topic (packaging, a single
# product subtype) earns at most ONE question in a category-level survey unless
# that topic is the survey's actual subject. Two packaging questions, or a
# question scoped to one subtype ("cleanser") inside a broader "skin care
# products" survey, take a disproportionate share of a ~30-question budget for
# a topic that is rarely the client's primary decision.
_MICRO_TOPIC_RE = re.compile(
    r"(?i)\b(packaging|pump|dispenser|tube|bottle|jar|container|refill)\b"
)


def _micro_topic_issue(qid: str, text: str, seen: dict) -> str | None:
    """Flag the second (or later) question on a narrow micro-topic."""
    m = _MICRO_TOPIC_RE.search(text or "")
    if not m:
        return None
    key = "micro_topic:" + m.group(1).lower()
    prior = seen.get(key)
    seen[key] = qid
    if prior:
        return (
            f"is a second question on '{m.group(1)}', matching {prior} -- a "
            "narrow topic like this earns one question in a category-level "
            "survey unless it is the survey's stated subject; drop one"
        )
    return None


# CHECK (B2C master rules, Section 27d, re-raising 26c) -- a prompt-only fix
# for this did not persist into the next generation (confirmed: the almost
# identical "what is the main purpose of your runs?" question reappeared
# unchanged in a later Running Shoes run after the same defect was already
# flagged once). This is the deterministic backstop the addendum asked for:
# a "why do you do the ACTIVITY" motivation question, where none of the
# options mention the product/category at all, is off-category regardless of
# how plausible it reads on its own -- it would work unchanged in a survey
# about the activity itself (running, cooking, driving) rather than the
# product being studied.
# change for b2c questionarie -- widened after "What is the main reason you
# run?" slipped through: the original pattern required a preposition
# between "reason" and the activity verb ("reason FOR running"), but real
# phrasing often has none at all ("reason you run", "why do you run").
# Added a second alternative for "why (do you)" stems, which have the same
# structure without ever using the word "reason"/"purpose".
_ACTIVITY_MOTIVATION_STEM_RE = re.compile(
    r"(?i)\b("
    r"(main\s+)?(purpose|reason)s?\s+(of|for|do\s+you|behind|you)?\s*.{0,15}?\b"
    r"(runs?|running|jogs?|jogging|exercis\w*|work\s?outs?|trains?|training|"
    r"cooks?|cooking|drives?|driving|travels?|traveling|travelling|shops?|"
    r"shopping|plays?|playing)\b"
    r"|"
    r"why\s+(do\s+you|does\s+someone)\s+"
    r"(run|jog|exercise|work\s?out|train|cook|drive|travel|shop|play)\b"
    r")"
)


def _off_category_activity_issue(text: str, options: list, segment: str) -> str | None:
    """Flag an activity-motivation question with no option naming the product."""
    if not _ACTIVITY_MOTIVATION_STEM_RE.search(text or ""):
        return None
    cat_tokens = _category_tokens(segment)
    if not cat_tokens:
        return None
    joined_options = " ".join(str(o or "") for o in (options or []))
    if cat_tokens & _tokens(text or "") or cat_tokens & _tokens(joined_options):
        return None  # the product IS named somewhere -- not off-category
    return (
        f"asks why the respondent does the ACTIVITY, with no option "
        f"mentioning {segment!r} at all -- this question would read "
        "identically in a survey about the activity itself (fitness, "
        "cooking, driving) rather than the product being studied; remove "
        "it or reframe the options around a product-relevant consequence "
        "of the motivation, not the motivation in the abstract"
    )


def _narrow_subcategory_issue(text: str, segment: str) -> str | None:
    """Flag a question scoped to one product subtype inside a broader category.

    "How often do you use a CLEANSER" inside a "skin care products" survey asks
    about one component of the category, not the category itself -- narrower
    than the audience the study defines.
    """
    seg_low = (segment or "").lower()
    if "product" not in seg_low and "care" not in seg_low:
        return None  # only meaningful when the segment is itself a broad category
    SUBTYPES = (
        "cleanser", "toner", "serum", "eye cream", "face mask", "moisturi",
        "sunscreen", "conditioner", "shampoo",
    )
    t = (text or "").lower()
    hit = next((s for s in SUBTYPES if s in t), None)
    if not hit:
        return None
    # A question that LISTS the subtype as one of several routine steps is
    # fine ("which steps... cleanser, toner, serum"); only flag when the
    # subtype IS the subject the stem asks about.
    if re.search(rf"(?i)\b(use|apply|buy|choose)\b[^.?!]{{0,20}}\b{re.escape(hit)}", t):
        return (
            f"asks specifically about '{hit}', a single product subtype, when "
            f"the survey covers the broader category '{segment}' -- ask about "
            "the category as a whole unless this subtype is a defined research "
            "objective"
        )
    return None


def _template_type_collision(qid: str, text: str, qtype: str,
                             seen: dict) -> str | None:
    """Flag when the SAME sentence template is used for two different types."""
    m = _MUST_TEMPLATE_RE.match(text or "")
    if not m:
        return None
    key = "which_must_template"
    prior = seen.get(key)
    seen[key] = (qid, qtype)
    if prior and prior[1] != qtype:
        return (
            f"shares the exact template of {prior[0]} but is a different "
            f"answer type ({qtype} vs {prior[1]}) -- a reader cannot tell them "
            "apart from the wording; reword this one so 'select all that "
            "apply' or similar is visible in the sentence"
        )
    return None


# change for b2c questionarie -- Issue 8. A distribution note is only useful
# if it references the FIXED segment set consistently. Catches a drifted
# spelling, a renamed segment mid-survey, or a persona invented on the fly.
_TITLECASE_PHRASE_RE = re.compile(
    r"\b[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?(?:\s[A-Z][a-z]+){1,3}\b"
)


# Static fallback vocabulary, widened after "Sensitive-Skin Sufferers" slipped
# through with an ending ("Sufferers") not on the original list.
_PERSON_NOUN_SUFFIXES_RE = re.compile(
    r"(?i)(users?|buyers?|shoppers?|owners?|consumers?|customers?|households?|"
    r"subscribers?|members?|sufferers?|seekers?|adopters?|enthusiasts?|"
    r"loyalists?|minimalists?|traditionalists?|explorers?|novices?|"
    r"beginners?|professionals?|experts?|purists?|skeptics?|advocates?|"
    r"switchers?|holdouts?|drivers?|decision.makers?)$"
)


def _segment_drift_issue(note: str, declared_segments: set) -> str | None:
    """Flag a distribution note naming a group outside the declared segment set.

    # change for b2c questionarie -- the suffix whitelist is derived from the
    # declared segments' OWN final word, not just a static list, so a persona
    # vocabulary the static list has never seen (e.g. "Sufferers") is still
    # caught as long as at least one DECLARED segment uses a recognised
    # person-noun ending to anchor the pattern.
    """
    if not note or not declared_segments:
        return None
    declared_endings = {
        seg.split()[-1].lower() for seg in declared_segments if seg.split()
    }
    for phrase in _TITLECASE_PHRASE_RE.findall(note):
        if phrase in declared_segments:
            continue
        last_word = phrase.split()[-1].lower()
        looks_personish = (
            bool(_PERSON_NOUN_SUFFIXES_RE.search(phrase))
            or last_word in declared_endings
        )
        if looks_personish:
            return (
                f"distribution note names {phrase!r}, which is not one of the "
                "survey's declared segments -- reuse the exact declared names, "
                "never introduce or rename one mid-survey"
            )
    return None


# change for b2c questionarie -- a profiling question (age/gender/household/
# income) exists ONLY for cross-tabulation. Its distribution note must never
# attribute a behavioural persona to a demographic slice ("Younger
# High-Involvement Users skew toward 18-34") or invent a demographic persona
# ("family-oriented segments"). Zero tolerance: unlike a behavioural question
# where a declared persona is expected, a profiling note should carry NONE.
def _profiling_persona_leak_issue(note: str, declared_segments: set) -> str | None:
    """Flag ANY persona-shaped name in a profiling question's distribution note."""
    if not note:
        return None
    for phrase in _TITLECASE_PHRASE_RE.findall(note):
        last_word = phrase.split()[-1].lower() if phrase.split() else ""
        if phrase in declared_segments or _PERSON_NOUN_SUFFIXES_RE.search(phrase):
            return (
                f"is a profiling question but its distribution note names "
                f"{phrase!r} -- profiling notes must describe the modelled "
                "demographic split only, never attribute it to a behavioural "
                "persona; omit the note or make it purely descriptive"
            )
    # Also catch demographic pseudo-personas that don't end in a person-noun
    # suffix at all: "family-oriented segments", "younger users" phrased as a
    # named group rather than a plain description.
    if re.search(r"(?i)\b(family[- ]oriented|budget[- ]conscious|premium|"
                 r"female[- ]led|male[- ]led|middle[- ]income|younger|older)\s+"
                 r"(segments?|users?|group|cohort)\b", note):
        return (
            "is a profiling question but its distribution note invents a "
            "demographic pseudo-persona (e.g. 'family-oriented segment') -- "
            "describe the demographic split plainly, do not name a new group"
        )
    return None


# change for b2c questionarie -- a distribution note may describe a MODELLED
# RESPONSE PATTERN ("higher-spending responses are more common among X"), but
# must never assert that a market trend, region, income, or age level CAUSES
# an answer -- the survey does not measure causation, only correlation within
# the panel.
_CAUSAL_CLAIM_RE = re.compile(
    r"(?i)\b("
    r"growth (in|of) .{0,20}(pushes|drives|increases|causes)|"
    r"(income|age|region|growth|trend) .{0,15}(explains|causes|drives|leads to)|"
    r"(women|men|younger|older) (drive|cause)s?\b"
    r")"
)


def _causal_claim_issue(note: str) -> str | None:
    """Flag a distribution note asserting causation the survey cannot support."""
    m = _CAUSAL_CLAIM_RE.search(note or "")
    if not m:
        return None
    return (
        f"makes an unsupported causal claim ({m.group(0)!r}) -- describe the "
        "modelled response pattern instead (\"higher-spending responses are "
        "more common among X\"), not what caused it"
    )


def _plain_language_issue(text: str) -> str | None:
    """Return a reason when a stem is not plain, active consumer language."""
    t = (text or "").strip()
    if not t:
        return None
    if _BROKEN_STEM_RE.match(t):
        return (
            "is missing a word and does not parse as English — read it aloud "
            "and add the missing noun (e.g. \"Which of these must a X do…\", "
            "\"What was the most important FACTOR when…\")"
        )
    if _DECIDER_RE.search(t):
        return (
            "asks who decides the purchase, but the audience definition already "
            "says this respondent buys the category — the answer is fixed, so "
            "the question cannot discriminate"
        )
    m = _JARGON_RE.search(t)
    if m:
        return f"uses research jargon {m.group(0)!r} — say it the way a shopper would"
    if _PASSIVE_RE.search(t):
        return "is written in the passive voice — ask it directly (\"do you…\")"
    if _THROAT_CLEARING_RE.match(t):
        return "opens with throat-clearing — delete the preamble and ask the thing"
    if _UNNECESSARY_TIMEFRAME_LEAD_RE.match(t):
        return (
            "opens with a timeframe clause ('In the past year,'/'In the last "
            "N months,') in front of a question about a standing habit, not a "
            "change over time — drop the clause and ask the habit directly "
            "(\"How often do you usually...\" not \"In the past year, how "
            "often have you...\")"
        )
    return None


# change for b2c questionarie — beat/topic mismatch breaks the reading flow.
# Questions are ordered by their beat, so a question tagged to the wrong beat
# gets sorted into the wrong place and the survey reads as if it jumped topic.
# Observed: "How likely are you to buy another pair…" (purchase intent) tagged
# `overall_satisfaction`, which put an intent question at the head of the
# satisfaction section while the real satisfaction item sat 8 questions later.
# Each entry: canonical beat id -> (subject regex the stem MUST match,
# regex for a subject that clearly belongs to a DIFFERENT beat).
_BEAT_SUBJECT_RULES = {
    # An "overall satisfaction" beat must carry an actual satisfaction reading.
    # Intent and switching questions are their OWN beats later in the arc, so a
    # stem about buying again or switching away is in the wrong slot — it pushes
    # the real satisfaction item out of the section head.
    "overall_satisfaction": (
        r"(?i)\b(satisf|dissatisf|happy|pleased|rate\s+your\s+experience|"
        r"how\s+(good|well))\b",
        r"(?i)\b(how\s+likely\s+are\s+you\s+to\s+(buy|purchase|repurchase|switch|"
        r"change|move)|intend\s+to\s+buy|next\s+(purchase|\d+\s+months?)|"
        r"switch\s+to\s+a\s+different|replace)\b",
    ),
    "advocacy": (
        r"(?i)\b(recommend|tell\s+(a\s+)?(friend|others)|word\s+of\s+mouth)\b",
        r"(?i)\b(how\s+much\s+do\s+you\s+(spend|pay)|where\s+do\s+you\s+buy)\b",
    ),
    "switching": (
        r"(?i)\b(switch|change|move\s+away|stop\s+(buying|using)|different)\b",
        r"(?i)\b(how\s+satisf|how\s+often\s+do\s+you\s+(use|run|wear))\b",
    ),
    "value_expectation": (
        r"(?i)\b(pay|price|spend|cost|worth|value|budget|\$|€|£)\b",
        r"(?i)\b(how\s+satisf|recommend)\b",
    ),
}


# CHECK (B2C master rules, Section 27a) -- category_entry is the survey's
# opening beat by design (see narrative.py's CANONICAL_BEATS comment), but
# "the category" is ambiguous between the PRODUCT and the activity/occasion
# around it. Confirmed live: a Running Shoes survey's category_entry beat
# filled with running-frequency/purpose questions that never mentioned shoes,
# so nothing anchored the survey to its actual subject until question 5. The
# prompt now says to name the product explicitly in this beat; this is the
# deterministic backstop in case that instruction is not followed.
def _category_entry_anchoring_issue(text: str, segment: str) -> str | None:
    """Flag the survey's opening (category_entry) question if its stem never
    names the product/category — only the activity or occasion around it."""
    cat_tokens = _category_tokens(segment)
    if not cat_tokens:
        return None
    if cat_tokens & _tokens(text or ""):
        return None
    return (
        "is the survey's opening question but never names "
        f"{segment!r} (or a word from it) — it reads as a question about the "
        "activity/occasion around the product, not the product itself, so "
        "nothing anchors the survey to its actual subject until a later "
        "question does"
    )


def _beat_subject_mismatch(text: str, beat_canonical: str) -> str | None:
    """Return a reason when a stem plainly does not match its beat's subject."""
    rule = _BEAT_SUBJECT_RULES.get((beat_canonical or "").strip())
    if not rule:
        return None
    must, wrong = rule
    t = text or ""
    if re.search(must, t):
        return None
    if re.search(wrong, t):
        return (
            f"beat '{beat_canonical}' expects a question about that subject, "
            "but this stem is about something else"
        )
    return None


# change for b2c questionarie — Phase A2 hygiene detectors
_DOUBLE_BARREL_RE = re.compile(
    r"(?i)\b("
    r"(?:how|rate|rating|satisfied|satisfaction|important|importance|"
    r"prefer|preference|experience|opinion|feel|evaluate|evaluation)"
    r".{0,80}?\b(?:and|&)\b.{0,80}"
    r"|(?:price|quality|service|value|brand|design|features?|support|"
    r"reliability|taste|packaging|delivery|availability)"
    r"\s+and\s+"
    r"(?:price|quality|service|value|brand|design|features?|support|"
    r"reliability|taste|packaging|delivery|availability)"
    r")"
)
_LEADING_RE = re.compile(
    r"(?i)\b("
    r"wouldn't you|don't you agree|dont you agree|isn't it true|"
    r"obviously|clearly you|everyone knows|surely you|"
    r"don't you think|dont you think|you must agree"
    r")\b"
)
_ESCAPE_RE = re.compile(
    r"(?i)\b("
    r"other(\s+please\s+specify)?|none(\s+of\s+the\s+above)?|"
    r"n/?a|not\s+applicable|prefer\s+not|not\s+sure|no\s+opinion|"
    r"does\s+not\s+apply|something\s+else"
    r")\b"
)
_SCALE_COMPLETE_RE = re.compile(
    r"(?i)\b("
    r"years?|age|income|\$|usd|eur|weekly|monthly|daily|rarely|never|"
    r"always|often|sometimes|hour|minute|day|week|month|year|"
    r"very\s+(dissatisfied|satisfied|unlikely|likely)|"
    r"strongly\s+(disagree|agree)|neither|neutral"
    r")\b"
)
# change for b2c questionarie — numeric / currency / measurement bands. These
# are exhaustive by construction and must not be asked for an 'Other' option.
_NUMERIC_BAND_RE = re.compile(
    r"(?i)("
    r"[\$€£₹]\s?\d|"                      # $60, €99
    r"\b\d[\d,]*\s*(-|–|—|to)\s*\d|"      # 60-99, 0–4
    r"\b(under|over|less than|more than|at least|up to|below|above)\s+[\$€£₹]?\d|"
    r"\b\d[\d,]*\s*(or (more|less|fewer|older|younger)|\+)|"
    r"\b\d+\s*(mm|cm|kg|lb|oz|ml|l|hours?|hrs?|mins?|minutes?|days?|weeks?|months?|years?)\b"
    r")"
)
# An explicit uncertainty option is itself an escape hatch.
# Real-world entities a respondent might have to add to, which is the only case
# where an "Other (please specify)" is genuinely needed.
_ENTITY_OPTION_RE = re.compile(
    r"(?i)(shop|store|retailer|website|site|app|pharmacy|chemist|salon|"
    r"market|supermarket|clinic|brand|channel|platform|social|search|"
    r"friend|family|doctor|dermatolog|influencer|magazine|tv|advert|"
    r"online|in-store|mall|boutique|counter|catalogue|email|newsletter)"
)

_DONT_KNOW_RE = re.compile(
    r"(?i)^\s*(i\s*(do\s*n[o']t|don't)\s*know|not\s+sure|unsure|"
    r"no\s+preference|prefer\s+not\s+to\s+say|don'?t\s+know)\b"
)
_LIKERT_NEG_END = re.compile(
    r"(?i)^(strongly\s+disagree|very\s+dissatisfied|very\s+poor|"
    r"not\s+at\s+all|never|extremely\s+unlikely|completely\s+disagree|"
    r"very\s+unlikely)$"
)
_LIKERT_POS_END = re.compile(
    r"(?i)^(strongly\s+agree|very\s+satisfied|excellent|extremely|"
    r"always|completely\s+agree|very\s+likely|extremely\s+likely)$"
)
_LIKERT_NEUTRAL = re.compile(
    r"(?i)\b(neither|neutral|neither\s+agree\s+nor\s+disagree|"
    r"no\s+opinion|undecided|moderately)\b"
)


# change for b2c questionarie — A10: respondents read the stem, not a brief.
# Long stems and research vocabulary are the two things that made the earlier
# runs read as heavy, so both are policed mechanically — the prompt alone has
# not been enough to hold the register.
#
# A11: the objective is CLARITY, not brevity. The v10 ceiling of 15 words pushed
# generation toward stems so terse they were ambiguous ("How often do you buy
# them?"), which is a worse defect than a slightly long question. The window is
# now two-sided: too short is a failure in its own right.
_MAX_STEM_WORDS = 18
_MIN_STEM_WORDS = 6

# Pronouns standing in for the category. A stem built on these cannot be read
# without its section heading, so it fails as a standalone question.
_VAGUE_REFERENT_RE = re.compile(
    r"(?i)\b(them|it|they|this\s+product|these\s+products|the\s+product|"
    r"this\s+category|the\s+category|this\s+item|the\s+item)\b"
)

_JARGON_RE = re.compile(
    r"(?i)\b("
    r"attributes?|drivers?|criteria|consumption\s+occasions?|purchase\s+journey|"
    r"trade[-\s]?offs?|touchpoints?|value\s+proposition|decision\s+matrix|"
    r"consideration\s+set|utili[sz]ation|propensity|salien(?:ce|t)"
    r")\b"
)

_PADDING_RE = re.compile(
    r"(?i)("
    r"thinking\s+about|"
    # change for b2c questionarie — A14: "Which of THESE best describes" is the
    # same padded formula as "which of the FOLLOWING best describes" and was
    # slipping through. It appeared on 7 of 29 questions in one survey, which is
    # what makes an instrument read as formulaic rather than professional.
    r"which\s+(of\s+(the\s+following|these)\s+)?best\s+(describes|reflects|captures|fits)|"
    r"which\s+statement\s+best\s+(describes|reflects)|"
    r"which\s+of\s+these\s+(would\s+you\s+say|best)|"
    r"to\s+what\s+extent\s+do\s+you\s+(agree|feel)|"
    r"please\s+(indicate|select|specify)|"
    r"considering\s+your\s+(recent|most\s+recent|typical)|"
    # change for b2c questionarie -- CHECK (Section 30c): "how would you
    # describe what you expect to pay" wraps a direct numeric question
    # ("how much would you expect to pay") in a throat-clearing "describe"
    # frame -- confirmed live on a monetary question. Same fix class as the
    # other padded openers above: collapse to the direct question.
    r"how\s+would\s+you\s+describe\s+what\s+you\s+(expect|plan|intend)\s+to"
    r")"
)


def _stem_word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def _category_tokens(segment: str) -> set:
    """Content words of the category name, for the standalone-stem check."""
    # change for b2c questionarie — A11
    return {
        w for w in re.split(r"[^a-z]+", (segment or "").lower())
        if len(w) > 2 and w not in _STOP
    }


def _is_overcomplicated(text: str, segment: str = "") -> str:
    """Reason the stem fails the readability window, or ''.

    # change for b2c questionarie — A10, widened in A11

    Two-sided: a stem can fail by being bloated OR by being so terse the
    respondent cannot tell what is being asked.
    """
    t = text or ""
    n = _stem_word_count(t)
    if n > _MAX_STEM_WORDS:
        return f"stem is {n} words — cut it to 8-16"
    if n < _MIN_STEM_WORDS:
        return f"stem is only {n} words — too terse to be unambiguous, aim for 8-16"

    # Standalone check: the stem must name what it is asking about. A pronoun
    # referent only works while the section heading is visible above it.
    cat = _category_tokens(segment)
    if cat and _VAGUE_REFERENT_RE.search(t):
        if not (cat & _tokens(t)):
            return (
                "stem refers to the product only as 'it'/'them' — name the "
                "category so the question stands on its own"
            )
    m = _PADDING_RE.search(t)
    if m:
        return f"padded opener {m.group(0)!r} — ask the question directly"
    m = _JARGON_RE.search(t)
    if m:
        return f"research jargon {m.group(0)!r} in the stem — use everyday words"
    # A parenthetical example ("video quality (e.g., 4K, HDR)") carries commas
    # without adding a clause — strip it before judging sentence structure.
    bare = re.sub(r"\([^)]*\)", "", t)
    if bare.count(",") >= 2:
        return "multi-clause stem — one clause per question"
    return ""


# change for b2c questionarie — A12: an option that says the respondent is not
# in the category contradicts the audience definition, which qualifies everyone
# in the sample as a user. "Other" is fine; "None — I don't use it" is not.
# change for b2c questionarie — "None" is only a non-user claim when it denies
# CATEGORY USAGE. On an opinion or reason list ("what would make you switch",
# "what disappointed you") a None answer is a legitimate opinion — the person
# still buys the category, they just have no complaint. The old pattern only
# whitelisted "none of these matter / are important", so every other honest
# None-of-these got flagged as impossible data. That put the architect in a
# no-win position: rule 7 demands an escape hatch on categorical single_choice,
# and this rule then rejected the escape hatch it just added.
_NON_USER_OPTION_RE = re.compile(
    r"(?i)^\s*("
    r"none\s*$|"
    r"none\s+of\s+(these|the\s+above)\s*$|"
    r"i\s+(do\s*n[o']t|don't|never)\s+(use|buy|subscribe|own|watch|drink|wear|shop)|"
    r"not\s+applicable|n/?a\s*$|"
    r"i\s+(do\s*n[o']t|don't)\s+(have|purchase)\b"
    r")"
)
# A trailing predicate turns "None of these" into an opinion, not a usership
# denial: "none of these would make me switch", "none of these matter to me".
_NONE_OPINION_RE = re.compile(
    r"(?i)^\s*none\s+of\s+(these|the\s+above|them)\s+\S+"
)


def _is_non_user_option(label: str) -> bool:
    """True when an option asserts the respondent is outside the category."""
    text = str(label or "").strip()
    if _NONE_OPINION_RE.match(text):
        return False
    return bool(_NON_USER_OPTION_RE.match(text))


def _nps_from_answers(answers: list) -> tuple[float, float, float] | None:
    """(promoters, passives, detractors) share from a 0-10 recommend item.

    # change for b2c questionarie — A12
    """
    buckets = {"p": 0.0, "s": 0.0, "d": 0.0}
    seen = 0
    for a in answers or []:
        label = str(a.get("label") or "")
        m = re.match(r"\s*(\d{1,2})\b", label)
        if not m:
            continue
        score = int(m.group(1))
        if score > 10:
            continue
        seen += 1
        pct = float(a.get("percentage") or 0)
        if score >= 9:
            buckets["p"] += pct
        elif score >= 7:
            buckets["s"] += pct
        else:
            buckets["d"] += pct
    if seen < 8:  # not a 0-10 scale
        return None
    return buckets["p"], buckets["s"], buckets["d"]


def _satisfied_share(answers: list) -> float | None:
    """Share choosing a satisfied point on a satisfaction scale."""
    total = sat = 0.0
    hits = 0
    for a in answers or []:
        label = str(a.get("label") or "").lower()
        if "satisf" not in label:
            continue
        hits += 1
        pct = float(a.get("percentage") or 0)
        total += pct
        if "dissatisf" not in label and "neither" not in label and "nor" not in label:
            sat += pct
    if hits < 3 or total <= 0:
        return None
    return sat * 100.0 / total


# change for b2c questionarie — A12: two questions can ask the same thing in
# entirely different words ("main reason you KEEP" vs "biggest reason you would
# STAY"). Token overlap cannot see that — both pairs score under 0.2 Jaccard —
# so redundancy is detected by INTENT instead. Each pattern names one thing a
# consumer survey can ask; two questions matching the same intent, in the same
# direction, are the same question twice.
_INTENT_PATTERNS = (
    ("retention_reason", r"(?i)\b(reason|why)\b.{0,40}\b(keep|stay|remain|continue|retain)\b"),
    # Stems, not exact words: "cancelling" must match the same intent as "cancel".
    ("churn_trigger", r"(?i)\b(cancel\w*|quit\w*|stop using|leav\w*|switch away|give up|drop\w*)\b"),
    ("discovery_source", r"(?i)\b(hear about|find out|discover|learn about|come across|first (see|hear))\b"),
    ("research_source", r"(?i)\b(compare|research|look up|check reviews|read reviews|evaluate)\b"),
    ("satisfaction_overall", r"(?i)\boverall\b.{0,30}\bsatisf"),
    ("recommend_intent", r"(?i)\b(recommend|tell a friend|refer)\b"),
    ("purchase_frequency", r"(?i)\bhow often\b.{0,30}\b(buy|purchase|order|replace)\b"),
    ("usage_frequency", r"(?i)\bhow often\b.{0,30}\b(use|watch|wear|drink|run|listen)\b"),
    ("spend_level", r"(?i)\b(how much|what).{0,25}\b(spend|pay|cost)\b"),
    ("switch_trigger", r"(?i)\b(switch|change) (to|brands|providers|services)\b"),
    ("must_have", r"(?i)\b(must[- ]have|essential|non[- ]negotiable|deal[- ]?breaker)\b"),
    ("purchase_channel", r"(?i)\bwhere do you (usually |normally |typically )?(buy|shop|purchase)\b"),
)


def _intent_signature(text: str) -> str:
    """The thing this question asks about, independent of its wording."""
    t = text or ""
    for name, pattern in _INTENT_PATTERNS:
        if re.search(pattern, t):
            return name
    return ""


def _looks_double_barreled(text: str) -> bool:
    return bool(_DOUBLE_BARREL_RE.search(text or ""))


def _looks_leading(text: str) -> bool:
    return bool(_LEADING_RE.search(text or ""))


def _has_escape_option(options: list) -> bool:
    return any(_ESCAPE_RE.search(str(o or "")) for o in (options or []))


def _needs_escape_hatch(qtype: str, options: list) -> bool:
    """Categorical single_choice lists need Other/None; complete scales do not."""
    if qtype != "single_choice":
        return False
    opts = [str(o or "") for o in (options or [])]
    if len(opts) < 3:
        return False
    # change for b2c questionarie — a numeric/currency band set is exhaustive by
    # construction ("Under $60 … $150 or more" covers every possible answer), so
    # demanding an 'Other' on it is wrong. _SCALE_COMPLETE_RE could never catch
    # these: it lists "\$" between \b word boundaries, and there is no word
    # boundary before a dollar sign, so every price band silently missed.
    if sum(1 for o in opts if _NUMERIC_BAND_RE.search(o)) >= max(2, len(opts) // 2):
        return False
    # An explicit "I don't know" already gives the respondent a way out.
    if any(_DONT_KNOW_RE.match(o.strip()) for o in opts):
        return False
    # change for b2c questionarie — a list that already spans the full range of
    # possible answers needs no "Other". "Morning only / Evening only / Both /
    # Only when..." covers every case; so does a directional ladder ("spend
    # less ... spend more") and an explicit closing negative ("Nothing would
    # make me switch"). Demanding an escape option on these produced 6 false
    # flags on one survey and pushed the repair loop into churn.
    joined = " | ".join(o.lower() for o in opts)
    if re.search(r"(?i)(nothing would|none would|no change|neither)", joined):
        return False
    if sum(1 for o in opts if re.match(r"(?i)^(only |both |i (do|only)|all of)", o.strip())) >= 2:
        return False
    if sum(1 for o in opts if re.search(r"(?i)(less|more|same|about the same)", o)) >= 3:
        return False
    # An "Other" only earns its place where the real world holds items the list
    # cannot enumerate - shops, brands, channels, sources, people. A list of
    # DESCRIBED BEHAVIOURS or REASONS ("I use one or two products", "To prevent
    # signs of ageing") is the writer's own taxonomy: it is meant to be
    # exhaustive, and bolting "Other" onto it invites a shrug instead of an
    # answer. Only demand the escape hatch when the options name real-world
    # entities the respondent might need to add to.
    if sum(1 for o in opts if _ENTITY_OPTION_RE.search(o)) < 2:
        return False
    # change for b2c questionarie — a count ladder that ends in an open top band
    # ("1 pair / 2 / 3 / 4 or more", "Less than once / 1 to 2 / 3 / 4 or more")
    # already covers every possible answer, so demanding an 'Other' on it is
    # wrong. The tell is the final option carrying "or more" / "or fewer" / "+".
    last = opts[-1].strip().lower()
    if re.search(r"(or\s+(more|less|fewer|older|younger|above|over)|\+)\s*$", last):
        return False
    # An NPS-style numeric ladder (0-10) is a scale, not a category list.
    numericish = sum(1 for o in opts if re.match(r"^\s*\d+\b", o.strip()))
    if numericish >= max(3, len(opts) - 2):
        return False
    # Likely a complete ordinal / demographic scale — skip.
    hit = sum(1 for o in opts if _SCALE_COMPLETE_RE.search(o))
    if hit >= max(2, len(opts) // 2):
        return False
    return True


def _likert_balanced(qtype: str, options: list) -> tuple[bool, str]:
    """Return (ok, reason). Checks count, neutral midpoint, bipolar ends."""
    opts = [str(o or "").strip() for o in (options or [])]
    expected = 5 if qtype == "likert_5" else 7 if qtype == "likert_7" else None
    if expected is None:
        return True, ""
    if len(opts) != expected:
        return False, f"{qtype} must have exactly {expected} options (has {len(opts)})"
    mid = opts[len(opts) // 2]
    if not _LIKERT_NEUTRAL.search(mid):
        # Soft: allow midpoint labels that are clearly middle of a named continuum
        if not re.search(r"(?i)\b(neither|neutral|moderate|fair|average|somewhat)\b", mid):
            return False, f"{qtype} midpoint not neutral: {mid!r}"
    first, last = opts[0], opts[-1]
    ends_ok = (
        (_LIKERT_NEG_END.match(first) and _LIKERT_POS_END.match(last))
        or (_LIKERT_POS_END.match(first) and _LIKERT_NEG_END.match(last))
    )
    # If ends don't match known templates, still require opposite polarity words.
    if not ends_ok:
        first_l, last_l = first.lower(), last.lower()
        polarity_flip = (
            ("disagree" in first_l and "agree" in last_l)
            or ("agree" in first_l and "disagree" in last_l)
            or ("dissatisf" in first_l and "satisf" in last_l)
            or ("satisf" in first_l and "dissatisf" in last_l)
            or ("unlikely" in first_l and "likely" in last_l)
            or ("likely" in first_l and "unlikely" in last_l)
            or ("poor" in first_l and ("excellent" in last_l or "good" in last_l))
            or (("excellent" in first_l or "good" in first_l) and "poor" in last_l)
        )
        if not polarity_flip:
            return False, f"{qtype} ends not a balanced bipolar scale: {first!r} … {last!r}"
    return True, ""


def _likert_direction(options: list) -> str:
    """Return 'neg_to_pos', 'pos_to_neg', or ''."""
    opts = [str(o or "").strip() for o in (options or [])]
    if len(opts) < 3:
        return ""
    first, last = opts[0].lower(), opts[-1].lower()
    if (
        ("disagree" in first and "agree" in last)
        or ("dissatisf" in first and "satisf" in last)
        or ("unlikely" in first and "likely" in last)
        or ("poor" in first and ("excellent" in last or "good" in last))
        or (_LIKERT_NEG_END.match(opts[0]) and _LIKERT_POS_END.match(opts[-1]))
    ):
        return "neg_to_pos"
    if (
        ("agree" in first and "disagree" in last)
        or ("satisf" in first and "dissatisf" in last)
        or ("likely" in first and "unlikely" in last)
        or (("excellent" in first or "good" in first) and "poor" in last)
        or (_LIKERT_POS_END.match(opts[0]) and _LIKERT_NEG_END.match(opts[-1]))
    ):
        return "pos_to_neg"
    return ""


class _JudgeResult(BaseModel):
    duplicates: list[list[str]] = []
    issues: list[QuestionIssue] = []


JUDGE_PROMPT = """\
You are an exacting survey-data auditor for a B2C survey on {segment}. Today is
{today}. You are given answered questions and the evidence index. Do NOT rewrite
— only flag. Check:
1. TRUE DUPLICATES AND NEAR-SIMILAR (HARD RULE): flag two questions when they
   collect the SAME or HIGHLY SIMILAR information — near-identical intent,
   paraphrase, overlapping constructs, or largely overlapping options. Prefer
   to flag when one adds little new insight. Theme-sharing alone is OK only when
   the constructs clearly differ (e.g. purchase frequency vs occasion of use).
   When unsure whether they are too similar, FLAG them. Cluster ids (first id =
   keep). fix_target=architect.
2. SUBJECT DRIFT: flag any question whose TEXT or OPTIONS are about a DIFFERENT
   product than "{segment}" (e.g. plant-based / oat / almond / soy / coconut milk
   when the target is organic DAIRY milk). The question must ask about the target
   segment itself. fix_target=architect.
3. QUESTION QUALITY / HYGIENE (fix_target=architect):
   - Double-barreled: asks two things at once ("price and service").
   - Leading/loaded: steers the respondent ("Wouldn't you agree…").
   - MECE gaps/overlaps: option sets that overlap, leave gaps, have overlapping
     numeric ranges, or recycle generic Brand/Price/Quality templates across
     many questions. Flag missing Other/None on categorical single_choice when
     the set is not an inherently complete scale.
   - Unbalanced likert: asymmetric ends or missing neutral midpoint.
   - Age / income: flag ANY question asking age, age group, how old, household
     income, salary, or earnings.
   - Brand / company names: flag ANY question asking which brand/company/
     manufacturer the respondent buys/uses/prefers, or listing brand/company
     names as options.
4. IRRELEVANT GEOGRAPHY SCREENER: flag ANY question asking where the respondent
   lives/resides/is based (region/country of residence). Region is a UI filter,
   not a survey question. fix_target=architect.
5. READ THE WHOLE THING AS ONE SURVEY, THE WAY A RESPONDENT WOULD. Before
   judging any single item, read every question in order and ask yourself:
   "Does this read like a real consumer survey a research firm would field, or
   like a list a machine generated?" Then flag what breaks that impression:
   - BROKEN ENGLISH. Read each stem ALOUD. Flag any that is missing a word or
     does not parse: "Which must a smartwatch do for you to buy it?" is missing
     "of these"; "What was the most important when you chose?" is missing a
     noun. A missing word is the single most visible defect in a deliverable.
   - THE SAME QUESTION ASKED TWICE IN DIFFERENT WORDS. "What matters most?",
     "What was most important?", "Which must it do?", "How do you judge
     quality?" and "What disappoints you?" are ONE question. Keep the best one
     and flag the rest, even when the wording shares few words.
   - QUESTIONS THAT TELL NOBODY ANYTHING ABOUT THE PRODUCT: who in the
     household decides, what time of day they use it, specs the owner would
     have to look up. The answer must say something about how the product is
     chosen, paid for, used, judged or replaced.
   - STIFF OR RESEARCH-Y LANGUAGE where a shopper's words belong ("For which
     purposes is the device utilised?" instead of "What do you use it for?").
   Flag each with fix_target=architect and quote the offending words.
EVIDENCE RULE (never violate): only flag what you can point at. Every issue you
raise MUST quote the exact words from the question or option that are wrong. If
you cannot quote the offending text, do not raise the issue. Reporting nothing
for a clean question is the correct answer — an empty issue list is a good
result, not a failure to find something.

Do NOT comment on data provenance, sourcing, grounding or citations. This survey
carries no per-option source data by design; there is nothing there to audit and
speculating about it produces false findings.

Reference question ids specifically.

Example:
- Two near-identical questions across two sections →
  duplicate cluster (fix_target=architect).
- "Which of the following best describes your age group?" → hygiene age ban
  (fix_target=architect).
- "Which brand of {segment} do you buy most often?" → hygiene brand-name ban
  (fix_target=architect).

Answered questions (JSON):
{questions_json}
"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _exact_text_duplicates(answered: list) -> list:
    """Cluster ids that share the exact same normalized question text."""
    buckets: dict[str, list] = {}
    for q in answered:
        key = _norm(q.get("text") or "")
        if not key:
            continue
        buckets.setdefault(key, []).append(q.get("id"))
    return [ids for ids in buckets.values() if len(ids) > 1]


def _near_exact_duplicates(answered: list, segment: str = "") -> list:
    """Near-identical pairs (fuzzy token / option overlap) — always uncapped.

    # change for b2c questionarie — Phase A3
    """
    ids = [q.get("id") for q in answered if q.get("id")]
    parent = {qid: qid for qid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(answered)):
        for j in range(i + 1, len(answered)):
            qi, qj = answered[i], answered[j]
            aid, bid = qi.get("id"), qj.get("id")
            if not aid or not bid:
                continue
            if is_near_exact(
                qi.get("text") or "",
                qj.get("text") or "",
                qi.get("options") or [],
                qj.get("options") or [],
                segment=segment,
            ):
                union(aid, bid)

    geo_ids = [
        q.get("id") for q in answered
        if q.get("id") and _is_geo_residence_question(q.get("text") or "")
    ]
    for extra in geo_ids[1:]:
        union(geo_ids[0], extra)

    clusters: dict = {}
    for qid in parent:
        clusters.setdefault(find(qid), []).append(qid)
    return [cids for cids in clusters.values() if len(cids) > 1]


def _heuristic_duplicates(answered: list, segment: str = "") -> list:
    """Loose similarity clusters (capped later). Skips near-exact pairs.

    # change for b2c questionarie — LOOSE_JACCARD only; near-exact handled separately
    """
    seg = _sim_tokens(segment)
    toks = [(q.get("id"), _sim_tokens(q.get("text", "")) - seg) for q in answered]
    parent = {qid: qid for qid, _ in toks if qid}
    by_id = {q.get("id"): q for q in answered}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            id_a, a = toks[i]
            id_b, b = toks[j]
            if not id_a or not id_b or not a or not b:
                continue
            qa, qb = by_id.get(id_a) or {}, by_id.get(id_b) or {}
            if is_near_exact(
                qa.get("text") or "", qb.get("text") or "",
                qa.get("options") or [], qb.get("options") or [],
                segment=segment,
            ):
                continue
            shared = len(a & b)
            if shared >= _MIN_SHARED_TOKENS and shared / len(a | b) >= _DEDUP_JACCARD:
                union(id_a, id_b)

    clusters = {}
    for qid, _ in toks:
        if qid in parent:
            clusters.setdefault(find(qid), []).append(qid)
    return [cids for cids in clusters.values() if len(cids) > 1]


# change for b2c questionarie — corroboration bar for judge-only duplicates.
_JUDGE_DUP_TEXT = 0.30
_JUDGE_DUP_OPTIONS = 0.20


def _corroborated_clusters(
    clusters: list, answered: list, segment: str,
) -> list:
    """Keep only judge duplicate clusters that something measurable supports.

    A pair stands if it is near-exact, shares wording above a modest bar, shares
    option vocabulary, or occupies the SAME storyline beat — beats being the
    strongest structural signal that two questions cover one moment.
    """
    from src.nodes.assembler import _option_token_overlap

    by_id = {q.get("id"): q for q in answered}
    kept: list = []
    for cluster in clusters:
        members = [qid for qid in cluster if qid in by_id]
        if len(members) < 2:
            continue
        head = by_id[members[0]]
        survivors = [members[0]]
        for qid in members[1:]:
            q = by_id[qid]
            a, b = head.get("text") or "", q.get("text") or ""
            oa, ob = head.get("options") or [], q.get("options") or []
            same_beat = bool(head.get("beat_id")) and head.get("beat_id") == q.get("beat_id")
            if (
                is_near_exact(a, b, oa, ob, segment=segment)
                or token_set_jaccard(a, b, segment) >= _JUDGE_DUP_TEXT
                or _option_token_overlap(oa, ob, segment) >= _JUDGE_DUP_OPTIONS
                or same_beat
            ):
                survivors.append(qid)
        if len(survivors) >= 2:
            kept.append(survivors)
    return kept


def _llm_judge(answered: list, index: dict, segment: str, today: str) -> _JudgeResult:
    """Judgment pass for semantic dedup + grounding/label nuance. Empty on failure."""
    import json

    compact = [{
        "id": q.get("id"), "tab": q.get("tab"), "text": q.get("text"),
        "type": q.get("type"),
        # change for b2c questionarie — the grounding fields are always null,
        # so sending them invited the judge to invent provenance faults.
        "options": q.get("options") or [],
        "answers": [{
            "label": a.get("label"), "percentage": a.get("percentage"),
        } for a in (q.get("answers") or [])],
    } for q in answered]
    prompt = JUDGE_PROMPT.format(
        segment=segment, today=today,
        questions_json=json.dumps(compact, ensure_ascii=False),
    )
    try:
        llm = get_structured_llm(_JudgeResult, temperature=0, max_tokens=8000)
        return llm.invoke(prompt)
    except Exception:  # noqa: BLE001 — degrade to mechanical + heuristic dedup only
        return _JudgeResult(duplicates=[], issues=[])


def _merge_clusters(a: list, b: list) -> list:
    """Union-merge two lists of id-clusters into disjoint clusters."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for cluster in list(a) + list(b):
        for x in cluster[1:]:
            union(cluster[0], x)
    groups = {}
    for cluster in list(a) + list(b):
        for x in cluster:
            groups.setdefault(find(x), set()).add(x)
    return [sorted(ids) for ids in groups.values() if len(ids) > 1]


def validator_critic(state: SurveyState) -> dict:
    """Audit the answered survey, score it, compute grounded_pct (Node 6)."""
    answered = state.get("answered_questions") or []
    counts_by_tab = state.get("counts_by_tab") or {}
    index = state.get("evidence_index") or {}
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    today = current_date_context()["iso"]

    # change for b2c questionarie — the currency this file's market uses, so
    # every money option can be checked against it.
    from src.standard_sections import currency_for as _currency_for
    _expected_currency = _currency_for(
        state.get("country") or state.get("geography_label") or state.get("region") or ""
    )

    _seen_templates = {}  # change for b2c questionarie -- Issue 4/9d
    _seen_micro_topics = {}  # change for b2c questionarie -- packaging etc.
    _seen_option_sets = {}  # change for b2c questionarie -- category-list overlap
    _seen_price_bands = {}  # change for b2c questionarie -- CHECK (Section 30b)
    _declared_segments = {
        sg.get('name') for sg in
        ((state.get('survey_blueprint') or {}).get('segments') or [])
        if sg.get('name')
    }
    per_q = {}  # id -> {"issues": [...], "targets": set()}

    def add(qid, msg, target):
        slot = per_q.setdefault(qid, {"issues": [], "targets": set()})
        slot["issues"].append(msg)
        slot["targets"].add(target)

    misattribution = False
    auto_fail = False
    hygiene_fail = False  # change for b2c questionarie
    currency_fail = False  # change for b2c questionarie -- CHECK (Section 26f)
    # change for b2c questionarie -- CHECK (Section 27g): same root cause as
    # currency_fail above. The detector already catches this defect every
    # time (confirmed by direct test); the addendum's own re-check on a
    # fresh generation still shipped one 100%-sum question, which means the
    # best-pass restore is (again) crowning a pass by score alone while this
    # specific defect was flagged but never regenerated before the revision
    # cap hit. Tracked the same way currency_fail is, so a clean pass can
    # never be outranked by score alone.
    multiselect_sum_fail = False
    # change for b2c questionarie -- CHECK (Section 30, escalated from
    # 28c): a construct-collision duplicate (e.g. fit_difficulty: "how
    # confident are you finding X that fits" vs "how often do you struggle
    # finding X that fits") shipped THREE consecutive generations despite
    # this exact detector flagging it every time -- the self-diagnosis was
    # never wired into the zero-tolerance best-pass gate the way currency
    # and multiselect-sum defects already are. "Known duplicate identified
    # by internal validation" is now a hard blocker on being crowned best,
    # exactly like those two defect classes.
    known_duplicate_fail = False
    # change for b2c questionarie -- CHECK (user directive, 2026-09-08):
    # set when any of the 16 required question types is missing, or the
    # advocacy/NPS question is not an 11-point 0-10 scale. Feeds the
    # zero-tolerance gate below so an incomplete pass cannot be crowned best.
    required_slot_fail = False
    total_opts = grounded = direct = adjacent = 0
    likert_dirs = []  # track scale direction consistency across survey

    # change for b2c questionarie -- CHECK (Section 26i): collect every
    # multi-select question's answer-percentage total FIRST, so the
    # suspicious-100%-sum detector below can compare one question's total
    # against its siblings in the same survey.
    _multiselect_sums = {}
    for q in answered:
        if q.get("type") == "multiple_choice":
            qsum = sum(float(a.get("percentage") or 0.0) for a in (q.get("answers") or []))
            _multiselect_sums[q.get("id")] = qsum

    for q in answered:
        qid = q.get("id")
        qtype = q.get("type")
        answers = q.get("answers") or []
        options = q.get("options") or []
        text = q.get("text") or ""
        # change for b2c questionarie — Phase A9: Screening and Profiling are
        # templated methodological sections. Age/income live in Profiling BY
        # DESIGN (cross-tabs only), and a screener must name the category
        # concretely. Exempt them from the instrument bans; every other section
        # is still policed exactly as before.
        standard = q.get("question_layer") in ("screening", "profiling")

        # D. Open-ended / structural auto-fail.
        if qtype not in VALID_TYPES or not options:
            add(qid, "open-ended or missing options (auto-fail)", "architect")
            auto_fail = True

        # Residence geography is a UI filter — never a survey instrument.
        if _is_geo_residence_question(text):
            add(
                qid,
                "irrelevant geography screener (geo-residence) — region is selected in the UI, not asked here",
                "architect",
            )

        # change for b2c questionarie — ban age/income and brand/company names
        if not standard and _is_age_or_income_question(text):
            add(
                qid,
                "hygiene: banned age/income demographic question — put in targetCustomer, not the instrument",
                "architect",
            )
            hygiene_fail = True
        # change for b2c questionarie — A16
        if not standard and _duplicates_profiling(text):
            add(
                qid,
                "relevance: Profiling already asks this at the end of the "
                "survey for cross-tabs. Asking it again here spends a slot on "
                "a fact the study already has — replace it with a question "
                "about what the respondent DOES in this category",
                "architect",
            )
            hygiene_fail = True
        if not standard and _is_brand_name_question(text, options):
            add(
                qid,
                "hygiene: banned brand/company-name question — no manufacturer names in stem or options",
                "architect",
            )
            hygiene_fail = True
        if not standard:
            named_example = _named_example_issue(options)
            if named_example:
                add(qid, f"hygiene: this question {named_example}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — A10/A11: keep the stems readable.
        if not standard:
            why = _is_overcomplicated(text, segment)
            if why:
                add(qid, f"hygiene: question not clearly worded — {why}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — mechanical hygiene (A2.4)
        if _looks_double_barreled(text):
            add(qid, "hygiene: double-barreled question (asks two things at once)", "architect")
            hygiene_fail = True
        if _looks_leading(text):
            add(qid, "hygiene: leading/loaded wording", "architect")
            hygiene_fail = True
        if _is_broken_grid_question(text, options, qtype):
            add(
                qid,
                "hygiene: broken matrix/grid — stem says 'each of the following' "
                "but the options cannot answer it (scale with nothing to rate, or "
                "an attribute list asked as a single pick); write one likert per factor",
                "architect",
            )
            hygiene_fail = True
        # change for b2c questionarie — A12: the audience definition qualifies
        # every respondent as a category user, so a "None / I don't use it"
        # option produces answers that cannot exist in the fielded sample.
        if not standard:
            for opt in options:
                if _is_non_user_option(opt):
                    add(
                        qid,
                        f"logic: option {str(opt)[:40]!r} says the respondent is "
                        "not a category user, which the audience definition "
                        "already rules out — replace it with a real answer",
                        "architect",
                    )
                    hygiene_fail = True
                    break

        if _needs_escape_hatch(qtype, options) and not _has_escape_option(options):
            add(
                qid,
                "hygiene: single_choice missing Other/None escape hatch (MECE)",
                "architect",
            )
            hygiene_fail = True
        if qtype in ("likert_5", "likert_7"):
            ok, reason = _likert_balanced(qtype, options)
            if not ok:
                add(qid, f"hygiene: unbalanced likert — {reason}", "architect")
                hygiene_fail = True
            direction = _likert_direction(options)
            if direction:
                likert_dirs.append((qid, direction))
        else:
            # change for b2c questionarie — most ordinal scales in practice ship
            # as single_choice, not likert_*, so the polarity check above never
            # saw them and satisfaction/likelihood scales drifted apart within
            # one survey. Track those too, using the option labels rather than
            # the declared type.
            direction = _scale_polarity(options)
            if direction:
                likert_dirs.append((qid, direction))

        # change for b2c questionarie -- CHECK (Section 27a): only the FIRST
        # category_entry question is "the opening" this rule is about -- a
        # beat can hold more than one question, and a later one in the same
        # beat legitimately explores the activity/occasion once the product
        # has already been named by the first.
        if (
            not standard
            and q.get("beat_canonical") == "category_entry"
            and (q.get("funnel_position") or 1) == 1
        ):
            anchoring = _category_entry_anchoring_issue(text, segment)
            if anchoring:
                add(qid, f"flow: this question {anchoring}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — a question tagged to a beat whose
        # subject it does not match gets sorted into the wrong place, and the
        # survey reads as if it jumped topic mid-section.
        if not standard:
            mismatch = _beat_subject_mismatch(text, q.get("beat_canonical") or "")
            if mismatch:
                add(
                    qid,
                    f"flow: {mismatch} — retag it to the beat it actually "
                    "answers, or rewrite it to serve this beat",
                    "architect",
                )
                hygiene_fail = True

        # change for b2c questionarie — plain, active, product-focused wording.
        if not standard:
            plain = _plain_language_issue(text) or _readability_issue(text)
            if plain:
                add(qid, f"language: this question {plain}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — Issue 9a: mixed answer dimensions.
        if not standard:
            mixed = _mixed_dimension_issue(options)
            if mixed:
                add(qid, f"hygiene: this question {mixed}", "architect")
                hygiene_fail = True
            optout = _missing_optout_issue(text, qtype, options)
            if optout:
                add(qid, f"hygiene: this question {optout}", "architect")
                hygiene_fail = True
            audience = _audience_contradiction_issue(text, options)
            if audience:
                add(qid, f"logic: this question {audience}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — Issue 9c: subject-verb agreement.
        if not standard:
            agree = _agreement_issue(text)
            if agree:
                add(qid, f"language: this question {agree}", "architect")
                hygiene_fail = True
            reg = _register_issue(text, segment)
            if reg:
                add(qid, f"language: this question {reg}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — Issue 8: segment name drift.
        note = q.get('distribution_note') or ''
        if standard:
            # Profiling/screener notes get a stricter, zero-tolerance check:
            # no persona attribution at all, not even a declared one.
            leak = _profiling_persona_leak_issue(note, _declared_segments)
            if leak:
                add(qid, f"consistency: {leak}", "architect")
                hygiene_fail = True
        else:
            drift = _segment_drift_issue(note, _declared_segments)
            if drift:
                add(qid, f"consistency: {drift}", "architect")
                hygiene_fail = True
        causal = _causal_claim_issue(note)
        if causal:
            add(qid, f"consistency: this note {causal}", "architect")
            hygiene_fail = True

        # change for b2c questionarie — Issue 4/9d: same template, different type.
        collide = _template_type_collision(qid, text, qtype, _seen_templates)
        if collide:
            add(qid, f"hygiene: this question {collide}", "architect")
            hygiene_fail = True

        if not standard:
            micro = _micro_topic_issue(qid, text, _seen_micro_topics)
            if micro:
                add(qid, f"redundancy: this question {micro}", "architect")
                hygiene_fail = True
            narrow = _narrow_subcategory_issue(text, segment)
            if narrow:
                add(qid, f"hygiene: this question {narrow}", "architect")
                hygiene_fail = True
            off_category = _off_category_activity_issue(text, options, segment)
            if off_category:
                add(qid, f"hygiene: this question {off_category}", "architect")
                hygiene_fail = True
            # change for b2c questionarie -- CHECK (Section 26d/26e): this used
            # to gate on qtype == "multiple_choice", so a single_choice / (the
            # now-removed) ranking / multiple_choice trio asking the same
            # underlying construct with the same option set never got
            # compared against each other at all -- confirmed on the Running
            # Shoes survey, where "what matters most in a shoe" appeared as a
            # single_choice pick, a ranking, AND a multi-select must-have list
            # with near-identical options, none of which triggered this check
            # because two of the three legs were never single_choice-gated
            # in. Overlap is meaningful regardless of response format, so
            # this must run for every question type.
            opt_overlap = _option_overlap_issue(qid, options, _seen_option_sets)
            if opt_overlap:
                add(qid, f"redundancy: this question {opt_overlap}", "architect")
                hygiene_fail = True

        # change for b2c questionarie -- CHECK (Section 30b): two questions
        # pricing the same product with mismatched bands can't be reconciled
        # against each other in analysis, even if each is individually a
        # legitimate distinct question.
        if not standard and _PRICE_POINT_STEM_RE.search(text):
            band_mismatch = _price_band_mismatch_issue(qid, options, _seen_price_bands)
            if band_mismatch:
                add(qid, f"consistency: this question {band_mismatch}", "architect")
                hygiene_fail = True

        # change for b2c questionarie -- CHECK (Section 29a): a recurring
        # spend question needs a stated time unit to be answerable
        # consistently across respondents.
        if not standard:
            spend_period = _spend_missing_period_issue(text)
            if spend_period:
                add(qid, f"hygiene: this question {spend_period}", "architect")
                hygiene_fail = True

        # change for b2c questionarie — Issue 1 / CHECK 18 (Section 26f): wrong
        # currency for the market. This must hold "without exception" -- a
        # currency-wrong pass previously could still score high enough
        # elsewhere to be crowned the snapshotted best-scoring pass and ship
        # (confirmed: the Running Shoes China survey shipped a USD price-band
        # question this exact check flags). currency_fail excludes the WHOLE
        # pass from best-pass eligibility below, not just this one question.
        cur_issue = _currency_issue(options, _expected_currency)
        if cur_issue:
            add(qid, f"consistency: this question {cur_issue}", "architect")
            hygiene_fail = True
            currency_fail = True

        # change for b2c questionarie -- CHECK (Section 30, currency): core
        # layer is cached across every country, so any hardcoded currency
        # there is wrong regardless of what _expected_currency says for THIS
        # specific job (it may coincidentally match if this job happens to
        # be the one that populated the cache) -- check unconditionally.
        if q.get("question_layer") == "core":
            core_cur_issue = _core_layer_currency_issue(options)
            if core_cur_issue:
                add(qid, f"consistency: this question {core_cur_issue}", "architect")
                hygiene_fail = True
                currency_fail = True

        # change for b2c questionarie — numeric recall over a look-back window.
        if not standard and _is_numeric_recall_question(text, options):
            add(
                qid,
                "hygiene: asks the respondent to total occasions/spend over a "
                "look-back window — that is an estimate, not a recallable fact; "
                "ask relative frequency (Always or almost always / Often / "
                "Sometimes / Rarely / Never) or a current standing state instead",
                "architect",
            )
            hygiene_fail = True

        # A + B per option.
        total = 0.0
        vals = []
        for a in answers:
            total_opts += 1
            p = a.get("percentage")
            if p is None:
                add(qid, f"option '{a.get('label')}' has null percentage", "simulator")
                p = 0.0
            elif p < 0 or p > 100:
                add(qid, f"option '{a.get('label')}' out of range: {p}", "simulator")
            total += float(p or 0)
            vals.append(float(p or 0))

            gi = a.get("grounded_in")
            sm = a.get("source_market")
            gl = a.get("grounding_label") or ""
            conf = a.get("confidence") or "low"
            if gi:
                grounded += 1
                if gl == "":
                    direct += 1
                else:
                    adjacent += 1
                if not sm:
                    add(qid, f"grounded option '{a.get('label')}' missing source_market", "simulator")
                else:
                    is_direct = _norm(sm) == _norm(segment)
                    if not is_direct and gl == "":
                        add(qid, f"MISATTRIBUTION: adjacent figure ({sm}) on '{a.get('label')}' has empty grounding_label", "simulator")
                        misattribution = True
                    elif is_direct and gl != "":
                        add(qid, f"direct-match option '{a.get('label')}' should have empty label, has {gl!r}", "simulator")
            else:  # persona-simulated / inferred — count medium+ confidence as panel-quality
                if conf in ("high", "medium"):
                    grounded += 1
                    direct += 1
                if gl:
                    add(qid, f"simulated option '{a.get('label')}' carries a grounding_label", "simulator")
                if sm:
                    add(qid, f"simulated option '{a.get('label')}' carries source_market", "simulator")

        # A. Sum semantics.
        total = round(total, 1)
        if qtype in _SUM_TO_100_TYPES:
            if abs(total - 100.0) > 1.0:
                add(qid, f"{qtype} sums to {total}, expected 100", "simulator")
            if len(vals) >= 3 and len({round(v, 1) for v in vals}) == 1:
                add(qid, "flat/uniform distribution (low quality)", "simulator")
            if (qtype == "single_choice" and len(vals) >= 3
                    and max(vals) >= 99.0
                    and sum(1 for v in vals if v <= 0.5) >= len(vals) - 1):
                add(qid, "degenerate 100/0/0 single_choice distribution (low quality)", "simulator")
        elif qtype == "multiple_choice" and len(vals) >= 3:
            # change for b2c questionarie -- CHECK (Section 26i): a fixed
            # 99-101 band flagged this correctly in testing, but the China
            # survey run still shipped two 100.0%-sum multi-select questions,
            # most likely because the best-scoring-pass restore kept an
            # earlier pass where this was flagged but not yet fixed (see the
            # "detection != fix" lesson from the 'friend or colleague' bug).
            # Comparing against sibling multi-select totals in the SAME
            # survey makes the signal sharper: a question near 100% next to
            # others that clearly exceed 100% is the tell that it was
            # generated as a constrained share, not independent probabilities.
            suspicious = _suspicious_multiselect_sum_issue(qid, total, _multiselect_sums)
            if suspicious:
                add(qid, f"hygiene: this question {suspicious}", "simulator")
                multiselect_sum_fail = True

    # change for b2c questionarie — A14: nested bands cannot be a single pick.
    # "$5 or more / $10 or more / $15 or more" — anyone for whom $5 is too
    # expensive is also above $10, so several options are simultaneously true
    # and the respondent has no defensible answer. Same for "at least $2 off /
    # at least $4 off". Real bands are disjoint ranges.
    _NESTED_BAND_RE = re.compile(
        r"(?i)^\s*(?:at\s+least\s+)?[$£€]?\s*\d[\d,.]*\s*(or\s+more|or\s+above|\+)\b"
        r"|^\s*at\s+least\s+[$£€]?\s*\d"
    )
    for q in answered:
        if q.get("question_layer") in ("screening", "profiling"):
            continue
        if q.get("type") != "single_choice":
            continue
        options = q.get("options") or []
        nested = sum(1 for o in options if _NESTED_BAND_RE.match(str(o)))
        if nested >= 3:
            add(
                q.get("id"),
                "logic: the options are nested thresholds (\"$10 or more\", "
                "\"at least $4 off\") — more than one is true for the same "
                "respondent, so a single pick has no defensible answer. Use "
                "disjoint bands (\"$10-$14\", \"$15-$19\") or ask for the "
                "single cut-off point directly",
                "architect",
            )
            hygiene_fail = True

    # change for b2c questionarie — A14: one construction used over and over is
    # what makes a questionnaire read as machine-produced. A professional
    # instrument varies its phrasing even when the underlying ask is similar.
    openers: dict = {}
    for q in answered:
        if q.get("question_layer") in ("screening", "profiling"):
            continue
        words = [w for w in re.split(r"\s+", (q.get("text") or "").strip()) if w]
        if len(words) < 3:
            continue
        opener = " ".join(words[:3]).lower().rstrip(",")
        openers.setdefault(opener, []).append(q.get("id"))
    for opener, ids in openers.items():
        if len(ids) >= 3:
            for qid in ids[2:]:
                add(
                    qid,
                    f"phrasing: {len(ids)} questions open with {opener!r}. Vary "
                    "the construction — repeating one formula makes the "
                    "instrument read as generated rather than written",
                    "architect",
                )
                hygiene_fail = True

    # change for b2c questionarie — A13: two questions asking for the same
    # quantity must offer the same bands, or their answers cannot be compared
    # or cross-tabulated. Frequency and spend are where this bites.
    _BAND_KINDS = (
        ("frequency", re.compile(r"(?i)\bhow often\b")),
        ("spend", re.compile(r"(?i)\b(how much|what).{0,25}\b(spend|pay|cost)\b")),
        ("count", re.compile(r"(?i)\bhow many\b")),
    )
    bands_by_kind: dict = {}
    for q in answered:
        if q.get("question_layer") in ("screening", "profiling"):
            continue
        for kind, pattern in _BAND_KINDS:
            if pattern.search(q.get("text") or ""):
                sig = tuple(_norm(str(o)) for o in (q.get("options") or []))
                bands_by_kind.setdefault(kind, []).append((q.get("id"), sig))
                break
    for kind, entries in bands_by_kind.items():
        if len(entries) < 2:
            continue
        first_id, first_sig = entries[0]
        for qid, sig in entries[1:]:
            if sig and first_sig and sig != first_sig:
                add(
                    qid,
                    f"consistency: this asks for a {kind} using different bands "
                    f"from {first_id}. Two questions measuring the same quantity "
                    "must share one band set, or the answers cannot be compared",
                    "architect",
                )
                hygiene_fail = True

    # change for b2c questionarie — A13: a question whose stem asks for a rating
    # must offer a rating scale. Without one the respondent has nothing to
    # answer with and the chart shows a share where a score was intended.
    _RATING_STEM = re.compile(
        r"(?i)\b(how satisfied|how important|how likely|how would you rate|"
        r"rate how|to what extent)\b"
    )
    for q in answered:
        if q.get("question_layer") in ("screening", "profiling"):
            continue
        text, options = q.get("text") or "", q.get("options") or []
        if not _RATING_STEM.search(text) or not options:
            continue
        scale_hits = sum(
            1 for o in options
            if any(h in " ".join(str(o).lower().split()) for h in _SCALE_OPTION_HINTS)
            or re.match(r"^\s*\d{1,2}\b", str(o))
        )
        if scale_hits < max(3, len(options) - 1):
            add(
                q.get("id"),
                "format: the stem asks for a rating but the options are not a "
                "rating scale — give it a labelled scale, or rewrite the stem "
                "as the categorical question the options actually answer",
                "architect",
            )
            hygiene_fail = True

    # change for b2c questionarie — A12: the category's defining issues must
    # actually be asked about. A streaming study with nothing on account sharing
    # or the ad tier, or a shoe study with nothing on replacement cycle, reads
    # as generic however well written the questions are. The planner declares
    # the topics per category; this checks the instrument delivers them.
    blueprint = state.get("survey_blueprint") or {}
    topics = [t for t in (blueprint.get("must_cover_topics") or []) if t]
    if topics:
        haystack = " ".join(
            f"{q.get('text') or ''} {' '.join(str(o) for o in (q.get('options') or []))}"
            for q in answered
            if q.get("question_layer") not in ("screening", "profiling")
        ).lower()
        missing = []
        for topic in topics:
            words = [w for w in re.split(r"[^a-z0-9]+", topic.lower()) if len(w) > 2]
            if not words:
                continue
            # Covered when the topic's distinctive words appear somewhere in the
            # instrument — a stem or an option both count as asking about it.
            if not all(w in haystack for w in words):
                missing.append(topic)
        if missing:
            first = next(
                (q.get("id") for q in answered
                 if q.get("question_layer") not in ("screening", "profiling")),
                None,
            )
            add(
                first,
                "coverage: this category cannot credibly omit "
                + ", ".join(repr(t) for t in missing[:4])
                + " — no question or option mentions it. Replace a weaker "
                "question with one that covers it",
                "architect",
            )
            hygiene_fail = True

    # change for b2c questionarie — A12: same intent asked twice. Reported to
    # the architect rather than deleted, so the slot is rewritten into a topic
    # the survey is missing instead of the questionnaire simply getting shorter.
    by_intent: dict = {}
    for q in answered:
        if q.get("question_layer") in ("screening", "profiling"):
            continue
        sig = _intent_signature(q.get("text") or "")
        if sig:
            by_intent.setdefault(sig, []).append(q)
    for sig, group in by_intent.items():
        if len(group) < 2:
            continue
        first = group[0]
        for dup in group[1:]:
            add(
                dup.get("id"),
                f"redundancy: this asks the same thing as "
                f"{first.get('id')} ({sig.replace('_', ' ')}) in different "
                f"words — {(first.get('text') or '')[:60]!r}. Replace it with a "
                "topic the survey does not yet cover",
                "architect",
            )
            hygiene_fail = True

    # change for b2c questionarie — A12: the instrument must be able to measure
    # the axis the study says it segments on. A price-sensitivity lens with no
    # spend, willingness-to-pay or price-increase question cannot assign anyone
    # to a price-driven segment — the segmentation would be unfielded opinion.
    lens = (state.get("survey_blueprint") or {}).get("segmentation_lens") or ""
    if re.search(r"(?i)\bprice|spend|value orientation|willingness to pay", lens):
        spend_re = re.compile(
            r"(?i)(how much|what.{0,20}(spend|pay|cost)|per month|monthly "
            r"(spend|cost|bill)|price increase|willing to pay|budget)"
        )
        has_spend = any(
            spend_re.search(q.get("text") or "")
            for q in answered
            if q.get("question_layer") not in ("screening", "profiling")
        )
        if not has_spend:
            first = next(
                (q.get("id") for q in answered
                 if q.get("question_layer") not in ("screening", "profiling")),
                None,
            )
            add(
                first,
                "coverage: the study segments on price sensitivity but no "
                "question asks about monthly spend, willingness to pay, or "
                "reaction to a price increase — add one, or the price-driven "
                "segments cannot be assigned",
                "architect",
            )
            hygiene_fail = True

    # change for b2c questionarie — A12: NPS vs satisfaction must not contradict.
    # A survey publishing a deeply negative NPS alongside a majority-satisfied
    # reading is describing two different populations. An analyst computes the
    # NPS before publishing it; this does that mechanically.
    nps_qid = nps_value = None
    sat_share = None
    for q in answered:
        parts = _nps_from_answers(q.get("answers") or [])
        if parts and nps_qid is None:
            promoters, _passives, detractors = parts
            nps_qid, nps_value = q.get("id"), promoters - detractors
            continue
        if sat_share is None and q.get("type") in ("likert_5", "likert_7"):
            share = _satisfied_share(q.get("answers") or [])
            if share is not None:
                sat_share = share

    if nps_value is not None and sat_share is not None:
        # A net-negative NPS means detractors outnumber promoters. That is not
        # compatible with a clear satisfied majority in the same sample.
        # Thresholds are judgement, not arithmetic: a detractor-heavy NPS beside
        # a satisfied majority is not strictly impossible (people can be content
        # yet not recommend a commodity service), but at this spread it is
        # implausible enough that it should be re-estimated rather than shipped.
        if nps_value < -20 and sat_share > 50:
            add(
                nps_qid,
                f"logic: recommendation and satisfaction contradict — this "
                f"question yields NPS {nps_value:+.0f} (detractor-heavy) while "
                f"{sat_share:.0f}% report being satisfied. Both cannot describe "
                "the same sample; re-estimate the recommendation distribution",
                "simulator",
            )
            hygiene_fail = True
        elif nps_value > 40 and sat_share < 45:
            add(
                nps_qid,
                f"logic: NPS {nps_value:+.0f} is strongly positive while only "
                f"{sat_share:.0f}% report satisfaction — re-estimate",
                "simulator",
            )
            hygiene_fail = True

    # change for b2c questionarie — consistent likert polarity across the survey
    # change for b2c questionarie -- CHECK (Section 27e, 30): construct
    # groups (module-level CONSTRUCTS / find_construct_duplicates) span the
    # whole survey, not just one tab -- "which would you least give up in a
    # shoe" (Purchase Journey) and "give up one thing to get a lower price"
    # (Brand/Product Experience) are the same sacrifice/trade-off construct
    # filed in different sections, and a respondent answers the same
    # construct identically regardless of which section it was filed under.
    by_construct = find_construct_duplicates(answered)
    for name, qs in by_construct.items():
        first = qs[0].get("text") or ""
        for dup in qs[1:]:
            add(
                dup.get("id"),
                f"redundancy: measures the same thing as "
                f"{qs[0].get('id')} ({name.replace('_', ' ')}) — "
                f"\"{first[:60]}\". Two questions a respondent would answer "
                "the same way are one question; drop or replace this one",
                "architect",
            )
        hygiene_fail = True
        known_duplicate_fail = True

    # change for b2c questionarie — CONTIGUITY. A topic must occupy one
    # unbroken run of questions. Krosnick: question-order context effects are
    # "almost always confined to contiguous items", so a topic split across
    # non-adjacent positions both reads as a jump and measures differently in
    # each place. The beat is the topic unit — questions are already sorted by
    # beat, so a beat appearing, stopping and resuming means two beats were
    # interleaved and the section reads as a random walk.
    by_tab: dict[str, list[dict]] = {}
    for q in answered:
        if (q.get("question_layer") or "") in ("screening", "profiling"):
            continue
        by_tab.setdefault(q.get("tab") or "", []).append(q)

    for tab, qs in by_tab.items():
        ordered_beats = [
            (q.get("beat_id") or q.get("beat_canonical") or "") for q in qs
        ]
        seen_runs: dict[str, int] = {}
        prev = None
        for beat in ordered_beats:
            if beat and beat != prev:
                seen_runs[beat] = seen_runs.get(beat, 0) + 1
            prev = beat
        for beat, runs in seen_runs.items():
            if runs > 1:
                offenders = [
                    q for q in qs
                    if (q.get("beat_id") or q.get("beat_canonical")) == beat
                ]
                for q in offenders[1:]:
                    add(
                        q.get("id"),
                        f"flow: topic '{beat}' is split across non-adjacent "
                        f"questions in '{tab}' — all questions on one topic must "
                        "sit together, or the survey reads as if it jumps back "
                        "and forth",
                        "architect",
                    )
                hygiene_fail = True

    if likert_dirs:
        dirs = {d for _, d in likert_dirs}
        if len(dirs) > 1:
            majority = max(dirs, key=lambda d: sum(1 for _, x in likert_dirs if x == d))
            for qid, d in likert_dirs:
                if d != majority:
                    add(
                        qid,
                        f"hygiene: likert direction {d} inconsistent with survey majority {majority}",
                        "architect",
                    )
                    hygiene_fail = True

    # Anti-fabrication. Within a question, one source grounding many options; and
    # across questions, one source grounding >3 questions with an IDENTICAL
    # distribution — both are fabrication laundered through labeling.
    ref_sigs = {}
    for q in answered:
        ans = q.get("answers") or []
        grefs = [a.get("grounded_in") for a in ans if a.get("grounded_in")]
        if len(grefs) >= 2 and len(set(grefs)) == 1:
            add(q.get("id"), "fabrication: one source grounds multiple options", "simulator")
        sig = tuple(round(float(a.get("percentage") or 0), 1) for a in ans)
        for ref in set(grefs):
            ref_sigs.setdefault(ref, {}).setdefault(sig, []).append(q.get("id"))
    for ref, sigs in ref_sigs.items():
        for sig, qids in sigs.items():
            if len(qids) > 3:
                for qid in qids:
                    add(qid, f"fabrication: source {ref} grounds {len(qids)} questions with identical distribution", "simulator")

    # C. Cross-tab dedup: near-exact (uncapped) + soft heuristic/LLM (capped).
    # change for b2c questionarie — Phase A3 split
    judge = _llm_judge(answered, index, segment, today)
    hard_dups = _merge_clusters(
        _exact_text_duplicates(answered),
        _near_exact_duplicates(answered, segment),
    )
    # change for b2c questionarie — a judge-only duplicate claim must be
    # corroborated before it can cost a question. On one measured run the judge
    # flagged 6 "duplicates" in a set with ZERO measurable similarity, and the
    # unconditional drop removed all six. Mechanical duplicates still drop
    # unconditionally; a judge claim needs shared wording, shared options, or
    # the same storyline beat to stand.
    judge_dups = _corroborated_clusters(
        list(judge.duplicates or []), answered, segment,
    )
    soft_dups = _merge_clusters(
        _heuristic_duplicates(answered, segment),
        judge_dups,
    )
    # Remove soft clusters that are already covered as hard near-exact.
    hard_ids = {qid for cluster in hard_dups for qid in cluster}
    soft_only = []
    for cluster in soft_dups:
        remaining = [qid for qid in cluster if qid not in hard_ids]
        # If cluster overlaps a hard pair, only keep soft members not in hard set
        # when ≥2 remain as their own soft group.
        if len(remaining) >= 2 and not all(qid in hard_ids for qid in cluster):
            soft_only.append(remaining)
        elif not any(qid in hard_ids for qid in cluster) and len(cluster) >= 2:
            soft_only.append(cluster)

    # Soft semantic clusters are treated as HARD uniqueness failures (no cap).
    # change for b2c questionarie — similar questions must never ship.
    duplicates = _merge_clusters(hard_dups, soft_only)
    uniqueness_fail = bool(hard_dups) or bool(soft_only)
    hard_dup_fail = uniqueness_fail
    for cluster in duplicates:
        for dup in cluster[1:]:
            kind = "near-exact duplicate" if any(dup in h for h in hard_dups) else "similar/duplicate"
            add(dup, f"{kind} of {cluster[0]}", "architect")
    _VALID_TARGETS = {"architect", "simulator", "estimator", "personas"}
    for qi in (judge.issues or []):
        for msg in qi.issues:
            target = qi.fix_target if qi.fix_target in _VALID_TARGETS else "simulator"
            if target == "estimator":
                target = "simulator"
            add(qi.id, msg, target)

    # D. Coverage vs plan — quota mode must hit MIN per tab / MIN total.
    # change for b2c questionarie
    feedback = []
    quota_fail = False
    blueprint = state.get("survey_blueprint")

    # change for b2c questionarie -- CHECK (user directive, 2026-09-08): the
    # 16 required question types are now GATED, not merely suggested in the
    # architect prompt. A pass missing any required slot is marked dirty
    # (required_slot_fail) so the best-pass restore can never crown it, and
    # each gap is fed back to the architect BY NAME so the regeneration
    # knows exactly which question to write. Confirmed necessary: across
    # several live runs the price question went missing entirely, a section
    # shipped 3 questions instead of 4, and the NPS question shipped as a
    # 5-option single_select instead of an 11-point 0-10 scale -- with
    # nothing in code to stop any of it from being published.
    _canon_to_market = {}
    for _s in ((blueprint or {}).get("sections") or []):
        _c = (_s.get("canonical") or "").strip()
        _sid = (_s.get("section_id") or "").strip()
        if _c and _sid:
            _canon_to_market[_c] = _sid
    for _canon, _gaps in find_missing_required_slots(answered, _canon_to_market).items():
        _labels = "; ".join(label for _key, label in _gaps)
        _n = len(REQUIRED_SLOTS.get(_canon) or ())
        feedback.append(
            f"coverage: section '{_canon}' is missing required question "
            f"type(s): {_labels}. All {_n} required types for this section "
            "must be present — write the missing one(s)."
        )
        required_slot_fail = True
        hygiene_fail = True

    # change for b2c questionarie -- CHECK (user directive, 2026-09-09): the
    # slot list is CLOSED, so an extra question on an unlisted theme is a
    # defect even when every required theme is already present.
    for _canon, _extras in find_offtheme_questions(answered, _canon_to_market).items():
        for _stem, _note in _extras:
            feedback.append(
                f"coverage: off-theme question in section '{_canon}' "
                f"({_note}): \"{_stem[:70]}\". The slot list for each section "
                "is closed — remove it and use the slot for its required "
                "theme instead."
            )
        required_slot_fail = True
        hygiene_fail = True

    _nps_issue = find_nps_shape_issue(answered)
    if _nps_issue:
        feedback.append(f"coverage: {_nps_issue}")
        required_slot_fail = True
        hygiene_fail = True
    for tab in narrative.section_ids(blueprint):
        c = counts_by_tab.get(tab)
        if c is None:
            c = sum(1 for q in answered if q.get("tab") == tab)
        # change for b2c questionarie -- CHECK (audit, 2026-09-09): this used
        # the FLAT QUESTIONS_PER_TAB_MIN (the smallest section's size) for
        # every section, so Purchase Journey could ship 6 of its required 7
        # -- or Product Experience 5 of 6 -- and pass silently. Each section
        # now has its own floor, which IS its exact framework count.
        _canon_floor = narrative.section_canonical(blueprint, tab)
        _floor = QUESTIONS_PER_SECTION.get(_canon_floor, QUESTIONS_PER_TAB_MIN)
        if c < _floor:
            feedback.append(
                f"tab '{tab}' under floor: {c}/{_floor} "
                "(the framework fixes this section's count)"
            )
            if GENERATION_MODE == "quota":
                quota_fail = True
        # change for b2c questionarie — no "under target" note. Telling the
        # architect a section is short of a number is exactly what produced
        # padding: it filled the gap with rephrasings of a question it had
        # already asked. Anything at or above the floor is an acceptable
        # section, and a 7-question section of real questions is a better
        # deliverable than an 8-question one carrying a duplicate.
    if len(answered) < _MIN_TOTAL_QUESTIONS:
        feedback.append(
            f"total questions {len(answered)} below floor {_MIN_TOTAL_QUESTIONS} "
            f"(target {TOTAL_TARGET})"
        )
        if GENERATION_MODE == "quota":
            quota_fail = True
    if quota_fail:
        feedback.insert(
            0,
            f"QUOTA: survey short of {_MIN_TOTAL_QUESTIONS}–{TOTAL_TARGET} questions "
            "— architect must refill dropped/missing items",
        )

    # change for b2c questionarie — Phase A8: report how much of the planned
    # storyline the survey actually covers. Order itself is enforced in code
    # (narrative.order_questions), so gaps are reported, not failed — a revision
    # here would churn without fixing anything the architect controls.
    story_gaps = []
    untagged_total = 0
    for tab in narrative.section_ids(blueprint):
        beats = narrative.beats_for_tab(blueprint, tab)
        cov = narrative.coverage(
            [q for q in answered if q.get("tab") == tab], beats,
        )
        untagged_total += cov["untagged"]
        if cov["missing"]:
            story_gaps.append(f"{tab}: missing {', '.join(cov['missing'][:4])}")
    if story_gaps:
        feedback.append("story coverage gaps — " + "; ".join(story_gaps))
    if untagged_total:
        feedback.append(
            f"story: {untagged_total} question(s) carry no beat_id and will sort "
            "to the end of their section"
        )

    # E. grounded_pct reinterpreted as panel-confidence coverage under simulation.
    gp = round(grounded / total_opts * 100, 1) if total_opts else 0.0
    dgp = round(direct / total_opts * 100, 1) if total_opts else 0.0
    agp = round(adjacent / total_opts * 100, 1) if total_opts else 0.0

    def _resolve_target(targets: set) -> str:
        if "architect" in targets:
            return "architect"
        if "personas" in targets:
            return "personas"
        return "simulator"

    per_question = [
        {"id": qid,
         "issues": info["issues"],
         "fix_target": _resolve_target(info["targets"])}
        for qid, info in per_q.items()
    ]

    # Score: fraction of clean questions, with severe-integrity caps.
    total_q = len(answered) or 1
    clean = sum(1 for q in answered if q.get("id") not in per_q)
    score = clean / total_q
    if misattribution:
        score = min(score, 0.5)
    if auto_fail:
        score = min(score, 0.4)
    score = round(score, 3)
    has_geo_screener = any(
        _is_geo_residence_question(q.get("text") or "") for q in answered
    )
    passed = (
        score >= PASS_SCORE
        and not auto_fail
        and not misattribution
        and not quota_fail
        and not has_geo_screener
        and not hygiene_fail  # change for b2c questionarie
        and not hard_dup_fail  # change for b2c questionarie — near-exact must revise
    )

    if misattribution:
        feedback.insert(0, "INTEGRITY: misattribution detected (adjacent figure without grounding_label) — top-priority fix")
    if has_geo_screener:
        feedback.insert(0, "IRRELEVANT: geography-of-residence screener(s) present — remove and refill")
    if hygiene_fail:
        feedback.insert(
            0,
            "HYGIENE: double-barreled / leading / MECE / unbalanced-likert issues — architect must revise",
        )
    if hard_dup_fail:
        feedback.insert(
            0,
            "DEDUP: duplicate or highly similar questions present — architect must drop extras and refill with distinct items",
        )
    feedback.append(f"grounded_pct={gp}% (direct={dgp}%, adjacent={agp}%)")
    feedback.append(
        f"dedup: hard={len(hard_dups)} clusters, soft_as_hard={len(soft_only)} "
        f"(thresholds near={NEAR_EXACT_JACCARD}, loose={LOOSE_JACCARD})"
    )

    critique = Critique(
        score=score, passed=passed, feedback=feedback,
        per_question=[QuestionIssue(**qi) for qi in per_question],
        duplicates=duplicates,
        hard_duplicates=_merge_clusters(hard_dups, soft_only),
        grounded_pct=gp, direct_grounded_pct=dgp, adjacent_grounded_pct=agp,
        counts_by_tab=counts_by_tab,
    ).model_dump()

    result = {
        "critique": critique,
        "grounded_pct": gp,
        "direct_grounded_pct": dgp,
        "adjacent_grounded_pct": agp,
    }

    # change for b2c questionarie — remember the best pass.
    # Revisions regenerate whole sections rather than patching flagged items, so
    # a later pass can be much worse (observed 0.742 -> 0.031). Snapshot every
    # improvement; the assembler restores it if the loop ends on a worse pass.
    #
    # CHECK 18 / Section 26f, CHECK / Section 27g, CHECK / Section 30
    # (escalated 28c): score alone let a currency-wrong, multi-select-sum-
    # wrong, or KNOWN-DUPLICATE pass get snapshotted as "best" over an
    # earlier pass that was clean on that specific defect but scored lower
    # on unrelated issues. The known-duplicate case is the most damning of
    # the three: the SAME construct-collision pair (fit-finding confidence
    # vs fit-finding frequency) was correctly self-diagnosed by this exact
    # detector on three consecutive generations and shipped anyway every
    # time -- a self-diagnosis with no enforced fix is equivalent to having
    # none. A pass with ANY zero-tolerance defect (currency, multi-select-
    # sum independence, or a known construct-collision duplicate) may only
    # replace the previous best if the previous best ALSO had one of these;
    # a pass clean on all three always outranks one dirty on any, regardless
    # of score, so this exception-free rule cannot be scored away. If every
    # pass this run has one of these defects, the highest-scoring one is
    # still kept (never ship nothing) -- that residual case is a generation
    # failure to fix upstream, not something the snapshot step can invent a
    # fix for.
    previous_best = state.get("best_score")
    previous_zero_tolerance_clean = state.get("best_zero_tolerance_clean")
    this_zero_tolerance_clean = not (
        currency_fail or multiselect_sum_fail or known_duplicate_fail
        or required_slot_fail
    )
    if previous_best is None:
        is_better = True
    elif this_zero_tolerance_clean != previous_zero_tolerance_clean:
        is_better = this_zero_tolerance_clean  # clean always beats dirty, any score
    else:
        is_better = score > previous_best  # same cleanliness — higher score wins
    if is_better:
        result["best_score"] = score
        result["best_answered"] = answered
        result["best_critique"] = critique
        result["best_counts_by_tab"] = counts_by_tab
        result["best_zero_tolerance_clean"] = this_zero_tolerance_clean

    # Increment the revision counter on failure so the graph's revise loop
    # terminates at the cap (single source of truth for the counter).
    if not passed:
        result["revision_count"] = (state.get("revision_count") or 0) + 1
    return result


if __name__ == "__main__":
    SEG = "Organic Milk"
    answered = [
        {  # (d) clean grounded single_choice summing to 100
            "id": "cp1", "tab": "consumer_profile", "text": "How often do you buy organic milk?",
            "type": "single_choice", "options": ["Weekly", "Every 2 weeks", "Monthly", "Rarely"],
            "chart_type": "bar", "serves_topic": "purchase frequency", "evidence_aligned": True,
            "sample_size": 2500, "distribution_note": "", "is_grounded": True, "question_confidence": "high",
            "answers": [
                {"label": "Weekly", "percentage": 41.0, "grounded_in": "https://example.gov/dairy", "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
                {"label": "Every 2 weeks", "percentage": 33.0, "grounded_in": "https://example.gov/dairy", "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
                {"label": "Monthly", "percentage": 18.0, "grounded_in": "https://example.gov/dairy", "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
                {"label": "Rarely", "percentage": 8.0, "grounded_in": "https://example.gov/dairy", "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
            ],
        },
        {  # (a) multiple_choice WRONGLY summed to 100
            "id": "bb5", "tab": "buying_behavior", "text": "Which factors influence your purchase? (Select all)",
            "type": "multiple_choice", "options": ["Quality", "Price", "Brand", "Reviews"],
            "chart_type": "horizontal_bar", "serves_topic": None, "evidence_aligned": False,
            "sample_size": 2500, "distribution_note": "", "is_grounded": False, "question_confidence": "low",
            "answers": [
                {"label": "Quality", "percentage": 30.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Price", "percentage": 30.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Brand", "percentage": 25.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Reviews", "percentage": 15.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
            ],
        },
        {  # (b) adjacent-grounded option MISSING its label (misattribution)
            "id": "bb6", "tab": "buying_behavior", "text": "How important is quality when choosing?",
            "type": "single_choice", "options": ["Very important", "Somewhat", "Not important"],
            "chart_type": "bar", "serves_topic": "purchase factors", "evidence_aligned": True,
            "sample_size": 2500, "distribution_note": "", "is_grounded": True, "question_confidence": "medium",
            "answers": [
                {"label": "Very important", "percentage": 78.0, "grounded_in": "report:9589", "source_market": "Sheep Milk", "grounding_label": "", "confidence": "medium"},
                {"label": "Somewhat", "percentage": 17.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Not important", "percentage": 5.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
            ],
        },
        {  # (c) duplicate of cp1 across a different tab
            "id": "bb1", "tab": "buying_behavior", "text": "In which region do you live?",
            "type": "single_choice", "options": ["North America", "Europe", "Asia Pacific", "Rest of World"],
            "chart_type": "donut", "serves_topic": None, "evidence_aligned": False,
            "sample_size": 2500, "distribution_note": "", "is_grounded": False, "question_confidence": "low",
            "answers": [
                {"label": "North America", "percentage": 40.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Europe", "percentage": 30.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Asia Pacific", "percentage": 20.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
                {"label": "Rest of World", "percentage": 10.0, "grounded_in": None, "source_market": None, "grounding_label": "", "confidence": "low"},
            ],
        },
    ]
    state = {
        "answered_questions": answered,
        "counts_by_tab": {"consumer_profile": 1, "buying_behavior": 3,
                          "preferences_expectations": 0, "satisfaction_future_intent": 0},
        "normalized_segment": SEG, "evidence_index": {}, "revision_count": 0,
    }
    out = validator_critic(state)
    c = out["critique"]
    print(f"score={c['score']} passed={c['passed']}")
    print(f"grounded_pct={c['grounded_pct']}% direct={c['direct_grounded_pct']}% adjacent={c['adjacent_grounded_pct']}%")
    print(f"duplicates={c['duplicates']}")
    print("per_question:")
    for qi in c["per_question"]:
        print(f"  [{qi['id']}] fix_target={qi['fix_target']}")
        for issue in qi["issues"]:
            print(f"      - {issue}")
    print("feedback:")
    for f in c["feedback"]:
        print(f"  - {f}")
