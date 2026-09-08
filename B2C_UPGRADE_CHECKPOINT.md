# B2C Agent Upgrade — Checkpoint 2 (2026-08-10)

Focus per your direction: **question & answer quality only**. Country work deferred.

---

## Key correction to Checkpoint 1

Checkpoint 1 said agree/disagree scales were the #1 defect, based on survey-methodology
literature. **I audited the actual output and that was wrong** — only **4 of 1,331**
questions use agree/disagree. The architect prompt already blocks most textbook defects.

Measured baseline across 1,331 questions / 43 published files:

| Check | Result |
|---|---|
| agree/disagree Likert | 4 (negligible) |
| multi-select missing None/Other | **0** (already handled) |
| stems > 25 words | 24 |
| **numeric recall over a window** | **76 → 52 confirmed** ⚠️ |
| **files with mixed scale polarity** | **10 of 43** ⚠️ |

Lesson worth keeping: audit real output before trusting a generic defect list.

---

## DONE — code changes (2 files)

### 1. `src/nodes/validator_critic.py`

**(a) New detector `_is_numeric_recall_question(text, options)`**
Catches "In the last 30 days, how many times did you…" / "how much did you spend in
total" — totals nobody can actually recall, so answers are guesses and the resulting
percentages are noise.

Deliberately narrow to avoid revision churn:
- needs BOTH a counting stem AND an explicit look-back window
- "how many services do you **currently** pay for" → passes (standing stock question)
- "**how often** … in the last 30 days" → passes when options are a frequency scale
  (that's legitimate and common), flagged only when options are numeric bands

Tightening this cut false positives from 76 → 52 flags; the 24 legitimate
frequency-scale questions now pass.

**(b) New detector `_scale_polarity(options)` + wired into the existing loop**
Root cause found: a direction-consistency check already existed, but only ran for
`qtype in ("likert_5","likert_7")`. **Most ordinal scales ship as `single_choice`**, so
they bypassed it — which is why satisfaction ran negative-first while likelihood ran
positive-first inside the same survey. Now all ordinal scales are tracked by their
option labels regardless of declared type.

Both feed the existing `hygiene_fail` → revision loop, so the pipeline self-repairs
rather than needing hand-patched files.

### 2. `src/nodes/question_architect.py`

- Rewrote rule 4 to ban look-back totals, with GOOD/BAD examples and the test:
  *"could the respondent answer confidently without doing arithmetic or consulting a receipt?"*
- Added a **survey-wide polarity rule** — every ordinal scale runs negative → positive,
  whatever the question type.
- Added a **verbatim scale library** (frequency / satisfaction / likelihood / importance /
  purchase intent) so wording stops drifting between sections.
- Added guidance preferring construct-specific scales over agree/disagree.

### Verification
- 10/10 recall-detector unit tests pass (incl. the tricky "how often" cases)
- 5/5 polarity tests pass, no false positives on categorical/price lists
- both files compile

---

---

## DONE — instrument length cut 31 → 22 (matching published practice)

Your call that "we don't need this many questions" is backed by the reference sources.

**Evidence gathered from the real instruments:**
- **Pew ATP W127 "Data Privacy"** (a full topical study) = 30 stems, 24 ASK ALL; filters
  + a split ballot mean one respondent answers **~22–26**. Their broadband *tracker* is 8.
- **Qualtrics**: drop-off climbs sharply past **12 min desktop / 9 min mobile**.
- **SurveyMonkey** (100k surveys): steepest incremental drop-off is in the first 15 Qs,
  and dwell time falls **30s → 19s by Q26** — the tail of a long survey is answered with
  measurably less thought.
- **Sawtooth**: segmentation separation *degrades* as basis variables grow —
  avg F-stat **67 at 20 vars → 50 at 40 → 32 at 80**. So cutting length does not cost
  cluster quality, it improves it. This was the decisive finding.
- Professional norm: **4–6 sections x 3–5 questions**; Deloitte reports in 5 sections.

**Changes:**

| | Before | After |
|---|---|---|
| Per section | 5 / **8** / 10 | 3 / **4** / 5 |
| Basis questions | 32 | **16** |
| + profiling | 6 | 6 |
| Total screens | ~38 (median shipped 31) | **22** |
| Est. desktop LOI | ~18 min | **~11 min** |

1. `src/question_plan.py` — quota cut to 3/4/5 per section, with the benchmark evidence
   recorded in the comment. All downstream nodes read from these constants, so the change
   propagates cleanly (verified: assembler, global_selector, question_architect,
   survey_simulator, validator_critic all consume `QUESTIONS_PER_TAB_TARGET`).
2. `src/nodes/story_planner.py` — **beats cut 3-8 → 3-4 per section.** This was a real
   conflict: 8 beats cannot fit into 4 questions, so beats would have competed for slots.
3. `src/nodes/question_architect.py` — reframed as a short instrument: every slot must
   earn its place, explicit instruction not to pad toward the count with softer variants.

Env-overridable if you want to tune: `QUESTIONS_PER_TAB_TARGET=5` gives 20 basis + 6 = 26.

---

---

## DONE — environment is live (2026-08-10)

`.env` written at `b2c-survey-agent-main/.env`, sourced from the CI BB project `.env`.

Only **one** variable is strictly required by `src/config.py`: `NEON_CONNECTION_STRING`.
Everything else defaults or degrades gracefully.

| Key | Source | Status |
|---|---|---|
| `NEON_CONNECTION_STRING` | CI BB `postgres_sql` | ✅ verified — `SELECT 1` OK |
| `DEEPSEEK_API_KEY` / `MODEL` | CI BB | ✅ verified — live call returned `OK` |
| `EXA_API_KEY` | **not in CI BB .env** | ⚠️ unset → search degrades to DuckDuckGo |
| `REDDIT_*` / `YOUTUBE_*` | **not in CI BB .env** | ⚠️ disabled → no consumer-voice enrichment |
| `ANTHROPIC_API_KEY` | commented out in CI BB | not used (provider = deepseek) |
| `REDIS_URL` | queue mode only | not needed for `run.py` |

**Verified the DB is the right one** — the agent reads `cmi_reports` (9,565 rows),
`cmi_reportsummery` (2,231) and `cmi_report_dynamic` (4,620); all present, and
`find_reports("running shoes")` returns real hits.

### Dedicated venv — do not use the shared one

`requirements.txt` pins `pydantic==2.10.4` / `psycopg2-binary==2.9.10`, which have no
wheels for **Python 3.14** and fail to compile (Rust/maturin error). Installing them into
the shared `c:\CI BB\.venv` would also have **downgraded the working B2B setup**.

So the B2C agent now has its own venv with 3.14-compatible versions:

```
cd "c:\CI BB\b2c-survey-agent-main"
.venv\Scripts\python.exe run.py "global running shoes market"
```

Both `.env` and `.venv/` are already in `.gitignore` — the key will not be committed.

---

---

## DONE — first live run (running shoes / North America, ~13 min)

`RUN_REGIONS="North America" .venv\Scripts\python.exe run.py "global running shoes market"`

Pipeline ran end to end. Output: `output/global-running-shoes-market_north-america.json`

**The length change worked:**

| | Before | After |
|---|---|---|
| Total questions | 31 (median) | **20** |
| Per section | 7/7/8/4 + 6 | **3/4/4/3 + 6 profiling** |
| Beats per section | 3–8 | **4/4/4/4** |

**Both of my earlier fixes verified on fresh output:**
- numeric-recall violations: **0** (this was the top defect — "how many miles in total")
- scale polarity: **consistent** (`neg_first` only; no mixed-direction survey)

**Known non-issue:** the run ends with `FileNotFoundError: No regional survey files under
..\cmi-platform-ai\frontend-v2\...`. That is only the Global blend (A6) step, which needs
all 5 regions present. Expected with `RUN_REGIONS=North America`; the region survey itself
generated fine.

### Two real validator bugs found and fixed

The validator scored the survey 0.4 (needs 0.85) and burned all 3 revisions. Investigating
showed most flags were **validator false positives, not question defects**:

1. **Escape-hatch rule vs price bands.** `_SCALE_COMPLETE_RE` listed `\$` between `\b` word
   boundaries — but there is no word boundary before `$`, so **every currency band silently
   failed the check** and price questions were told to add an "Other". Numeric/currency/
   measurement bands are exhaustive by construction. Added `_NUMERIC_BAND_RE`, plus an
   exemption when the list already offers "I don't know".
2. **"None of these" contradiction.** Rule 7 requires an escape hatch on categorical
   single_choice; the non-user rule then rejected the escape hatch it just demanded. The old
   regex only whitelisted "none of these *matter/are important*", so an honest opinion answer
   like "None of these would make me switch" was flagged as impossible data. Added
   `_NONE_OPINION_RE` — a trailing predicate makes it an opinion, not a usership denial.

Re-scored the same survey: **0.4 → 0.55**, flags 12 → 9. Of the 9 remaining, 5 are on
profiling questions (age/income/gender/NPS) that carry `question_layer='profiling'` and
**are exempt in the real pipeline** — they only appeared because my offline harness lost
that field. Effective score ≈ **0.75**.

9/9 unit tests pass on both fixes.

---

## NOT DONE — resume here

1. **Re-run with the validator fixes in place** — the run above used the pre-fix validator,
   so it wasted revisions on false positives. A fresh run should score materially higher and
   spend its revisions on the 4 genuine issues below.
2. **4 genuine question defects the validator correctly caught** (these are real, and the
   revision loop should now fix them):
   - `q9` stem is 20 words — "When you pick up a running shoe in a store, which one sign most
     tells you it is well made?"
   - `q11` vs `q10` redundancy — sustainability trade-off overlaps the width/drop question
   - `q12` "How likely are you to buy another pair…" — stem asks for a rating, options aren't
     a rating scale
   - `q13` uses a bare "None of these" (still correctly flagged; needs a predicate)
3. **Compare against a real internet B2C survey** — your task 3, still pending.
4. Section names don't match the CMI mandated 4-section framework
   (Consumer Profile & Usage / Purchase Journey & Drivers / Brand Experience &
   Satisfaction / Unmet Needs, Switching & Intent) — see the RAR analysis. Not yet fixed.
5. Cosmetic, known: mojibake in published JSON (`Running Shoes � consumer survey`), and
   the "2,500 synthetic respondents" wording in `howToReadTheNumbers` conflicts with the
   methodology-wording rule.
6. Deferred by your call: country-level output (blocked in 3 places; DB has no country
   data — `cmi_countries` isn't in Neon and the 5455 mirror was down).
