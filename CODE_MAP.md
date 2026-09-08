# Code map — b2c_survey

A read of the actual code (not the README's description of it). Covers how state
flows, what each prompt does, where quality rules are enforced, and which parts
of the codebase are no longer wired in.

---

## 1. Two graphs, three entry points

`src/graph.py` compiles **three** graphs, but only two are used in production:

| Graph | Entry → exit | Used by |
| --- | --- | --- |
| `build_classify_graph()` | `scope_classify` → END \| `reject_stop` → END | `classify_segment_async` |
| `build_region_graph()` | `evidence_harvester` → … → `assembler` → END | `run_region_pipeline` |
| `build_graph()` | full single-shot (classify + pipeline) | nothing — legacy/debug |

The 8-node architecture in the README is really **1 node + 6 nodes + a non-graph
fan-in**:

```
scope_classify  (classify graph, runs ONCE)
    │
    ├─ gate fails ─→ reject_stop → output/<slug>_rejected.json, no region files
    │
    └─ gate passes → for each of 5 continents, run the region graph:
         evidence_harvester → evidence_synthesizer → persona_generator
           → story_planner → question_architect → survey_simulator → validator_critic
                                        │
                        ┌───────────────┴───────────────┐
                    passed / cap reached          needs revision
                        │                              │
                     assembler                 architect | personas | simulator
                        │
                   final_survey → output/*.json + public/surveys/<slug>/<region>.json

    then (outside LangGraph):
    global_selector.publish_global_selection() → global.json + index.json
```

**Entry points** (`src/tasks.py`):

- `run.py "<segment>"` → `run_one_async` → classify, then the 5 regions
  **sequentially** in-process, then A6. No Redis needed.
- `enqueue.py` → RQ job `classify_and_fanout` → classify, then
  `_enqueue_regional_jobs` enqueues 5 `run_one_region` jobs + one
  `run_global_selection` with `depends_on=[the 5]`.

Both paths converge on the same functions, so behavior is identical apart from
concurrency and the `_maybe_stagger_start()` sleep (RQ only, driven by
`job.meta["stagger_index"]`).

---

## 2. State flow

`src/state.py` is one flat `TypedDict` (`SurveyState`, `total=False`). Only
`evidence_pool` has a reducer (`operator.add`); everything else is
last-write-wins. Nodes return partial dicts that LangGraph merges.

Field ownership, in write order:

| Written by | Keys |
| --- | --- |
| `scope_classify` | `normalized_segment`, `segment_type`, `classification_confidence`, `is_hybrid`, `region`, `regions_to_simulate`, `classification_reason`, `audience_profile`, `gate_passed`, `gate_decision`, `status`, `rejection_reason` |
| `evidence_harvester` | `evidence_pool` (+=), `evidence_empty` |
| `evidence_synthesizer` | `market_knowledge_profile`, `grounding_thin` |
| `persona_generator` | `persona_catalog` |
| `story_planner` | `survey_blueprint` |
| `question_architect` | `questions`, `counts_by_tab`, `questionnaire_id`, `questionnaire_cache_hit`, `questions_by_region` |
| `survey_simulator` | `answered_questions`, `answered_by_region`, `grounded_question_count`, `simulation_confidence_avg` |
| `validator_critic` | `critique`, `grounded_pct`, `direct_grounded_pct`, `adjacent_grounded_pct`, `revision_count` (only on failure) |
| `assembler` | `final_survey` |

**Never written by anything:** `evidence_index`. See §7.

The region graph starts *after* `scope_classify`, so `run_region_pipeline`
hand-seeds the classification fields into the initial state, plus
`classification_preseeded: True`. That flag exists so the passthrough branch at
the top of `scope_classify` can short-circuit if the full graph is ever used —
in the region graph the node isn't present at all.

---

## 3. Node by node

### `scope_classify` — the B2C gate

One structured LLM call (`ScopeResult`, temperature 0). The prompt asks for
normalization, `b2c|b2b|other`, a confidence, an `is_hybrid` flag, and a
consumer `audience_profile`, with three worked examples (organic milk → b2c,
hydraulic seals → b2b, insulin → hybrid).

`decide_b2c_gate()` is pure logic and independently testable
(`python -m src.nodes.scope_classify` runs four cases with no LLM):

```
b2c                            → accept   (accepted_b2c)
is_hybrid and not strict       → accept   (accepted_hybrid_proxy)
conf in (medium, low), lenient → accept   (accepted_ambiguous)
strict and not b2c             → reject   (rejected_strict)
(b2b|other) and conf == high   → reject   (rejected_pure_b2b)
otherwise                      → accept   (accepted_lenient_fallback)
```

Default `B2C_GATE_STRICT=false`, so **the only hard stop is high-confidence pure
b2b/other with `is_hybrid=false`**. `audience_profile` produced here is the sole
input to `targetCustomer` in the final JSON — nothing downstream enriches it.

### `evidence_harvester` — two sources, concurrent, failure-tolerant

`asyncio.gather(..., return_exceptions=True)` over:

- **SOURCE A — CMI Postgres** (`_harvest_database`, sync, in `to_thread`).
  `db.find_reports` tokenizes the segment (stopwords like "global"/"market"
  dropped, plurals stemmed, a small synonym map maps `sneakers → footwear/
  athletic/shoe`), then scores reports by how many distinct tokens ILIKE-match
  `newsSubject|keyword|keywords|segmentation`. Takes 25 candidates, reorders so
  reports that actually have summary/dynamic rows come first
  (`reports_with_data`), mines the top 8. Per-report exceptions are swallowed;
  the connection is closed in `finally`.
- **SOURCE B — web** (`web_search.search_web`), 7 fixed queries built in
  `_build_web_queries`, each tagged with an `angle`.

Routing lives in `web_search._dispatch_query`:

| Angle | Chain |
| --- | --- |
| `complaints`, `needs`, `customer_language`, `reviews` (voice) | YouTube + Reddit + Twitter; Exa if <3 findings; DDG if <2 |
| everything else | Exa; DDG if <2 findings |

Facts are unified by `evidence_utils.build_fact` — every fact from either source
has the same shape (`fact/value/is_numeric/topic/tab_hint/origin/authority/
source_ref/title/retrieved_at`). Authority is derived from origin
(`stat`/`cmi_database` = high, `exa`/`web` = medium, `reddit`/`youtube`/
`twitter` = low). `build_voice_fact` force-nulls `value`/`is_numeric` so social
sources can never carry a figure. Pool is deduped and capped at 120, numeric
first. Region is stamped on every fact.

Both sources failing → `evidence_empty: True`, and the pipeline continues.

### `evidence_synthesizer` — the structural/voice firewall

Splits the pool into **structural** vs **voice** (`_is_voice`: origin in
`VOICE_ONLY_ORIGINS`, or topic contains complaint/customer/voice/review), sorts
structural numeric-first, and passes them to the LLM as two separate JSON blocks
with hard rules:

1. Structural bands (`market_maturity`, `avg_income_band`, `price_sensitivity`,
   `growth_outlook`, `technology_adoption`, `environmental_awareness`,
   `brand_loyalty`, `consumer_sentiment`, `government_incentives`) may **only**
   come from structural evidence.
2. `top_pain_points` / `consumer_language` / `feature_requests` should prefer
   voice evidence — **language only, never numbers**.
3. `regional_notes` for all 6 regions, no percentages.

This is the architectural claim the whole product rests on: Reddit shapes
*wording*, reports shape *bands*, and neither is pasted as an answer percentage.

Cached on disk keyed by `(schema, segment, region, first 80 facts)`. On LLM
failure or empty evidence it falls back to `_fallback_mkp` (generic bands,
`evidence_confidence="low"`) and sets `grounding_thin=True`.

### `persona_generator` — archetypes + weight packs

Two prompt types:

- `PERSONA_PROMPT` → 12 archetypes, region-native, told to absorb
  `consumer_language`/`top_pain_points` into `pain_points` with real wording and
  **not** to invent population percentages.
- `WEIGHT_PROMPT` → one call *per region in `regions_to_simulate`* returning
  `persona_id → percent`, seeded with example regional patterns (Europe = eco,
  APAC = value/mobile, LatAm/MEA = price-sensitive, NA = premium).

`_normalize_weights` rescales to sum 100.0 and absorbs drift into the largest
persona. Any failure → equal weights. `<10` personas → `_fallback_personas`
(12 hardcoded templates). Cached per `(schema, segment, region, mkp notes,
confidence)`.

For a single-continent job, only that region's pack is generated and no Global
blend is computed.

### `question_architect` — the biggest prompt, and where most rules live

`PROMPT_TEMPLATE` (~150 lines) is the quality contract. Sections:

- **Tone** — Nielsen/McKinsey consulting voice, client-ready.
- **Funnel** — broad→narrow, easy→hard, safe→sensitive; model assigns
  `funnel_position`.
- **Hygiene rules** — one idea per question, neutral wording, MECE options,
  balanced symmetric Likert with neutral midpoint, no broken matrix stems
  ("each of the following" + scale-only options), concrete timeframes, hard
  uniqueness.
- **Banned content** — age/income, brand/company names (stem *and* options),
  geography of residence.
- **Subject rule** — text and options must be about the exact target product;
  adjacent-market evidence may inform topics but never change what's asked.
- **Chart variety** — must spread across bar/horizontal_bar/radar/donut/stacked;
  `type` (answer format) and `chart_type` (visualization) are distinct fields.
- **Personalization** — lift specific option labels from consumer voice
  ("Battery dies too fast"), not generic Brand/Price/Quality stacks.

Generation is chunked (`QUESTION_CHUNK = 8`) with up to `MAX_TAB_ATTEMPTS = 6`
retries per tab, because DeepSeek's ~8k output cap truncates large calls. Each
chunk's prompt is fed the running list of already-accepted question texts
(last 60) as an explicit "do not paraphrase these" block.

**Post-generation code filters** (`_generate_tab`) — the model is not trusted:

```python
too_similar_to_any(text, running, segment, threshold=LOOSE_JACCARD)  # 0.58
_is_geo_residence_question(text)
_is_age_or_income_question(text)
_is_brand_name_question(text, options)
```

Each rejection is recorded in `notes`. Note `if added == 0: break` — a chunk
that's entirely rejected ends the tab early and produces a shortfall.

**Core + module split** (`REGION_QUESTION_MODE=core_plus_module`, the default):

With TARGET=8, MIN=5 → `MODULE_PER_TAB = 2`, `CORE_PER_TAB = 6`.

- CORE is generated once with `region="Global"` and cached under key
  `(schema, segment, "core", mode)` — **no region or persona in the key**, so
  all five region jobs share it. Whichever region runs first (North America)
  shapes the core instrument with *its* MKP and personas.
- MODULE is generated per region with `seed_avoid` = all core question texts,
  and a prompt addendum demanding region-native channels/norms/pains.
- Merged 6+2 per tab, ids renumbered `cp1..cp8`, `bb1..`, `pe1..`, `sf1..`.

On a **revision** (`is_revision=True`) the core/module path is skipped entirely
and `_build_tabs` refills with `layer="full"`, so post-revision region files
carry a mix of `core`/`module`/`full` layers.

Kept questions on revision are filtered by `drop_ids` (duplicate clusters +
`fix_target == "architect"` issues matching a keyword list) and re-filtered
through the same three ban predicates.

### `survey_simulator` — where percentages are born

Batches of 8 questions, one async LLM call per batch, `asyncio.gather` across
batches and across regions. The prompt frames it as simulating a panel of
`SAMPLE_SIZE = 2500` over the **weighted persona mix**, with:

1. one estimate per question id, percentages for every option label as given;
2. `single_choice`/`ranking`/`likert_*` sum to 100 (±1); `multiple_choice`
   options independent (do NOT sum);
3. distributions must be **non-uniform**;
4. never cite Reddit as a numeric source, never invent URLs;
5. `distribution_note` naming which personas drove the mix;
6. a `confidence` (high/medium/low).

`_enforce_sums` then does the math in code: clamp to [0,100], and for
sum-to-100 types renormalize and apply **largest-remainder rounding to exactly
100.00 at 2 decimals**. Failure of a whole batch → `_fallback` (a decaying
1−0.06n ramp, confidence `low`).

Revision handling: only questions flagged `simulator`/`estimator` (or listed in
`critique.revise_questions`) are re-simulated; prior answers are merged back in.

Output: `answered_questions` (the primary region), `answered_by_region` (slug →
[{id, values}]), and `data_by_region` attached to each answered question.

### `validator_critic` — audit only, never rewrites

Mechanical checks in code, one LLM judgment pass on top.

**Per question (code):**

| Check | Flag | Target |
| --- | --- | --- |
| type not in VALID_TYPES, or no options | `auto_fail` | architect |
| geo-residence stem | `has_geo_screener` | architect |
| age/income stem | `hygiene_fail` | architect |
| brand/company stem or ≥3 brand-ish options | `hygiene_fail` | architect |
| double-barreled (regex) | `hygiene_fail` | architect |
| leading/loaded ("Wouldn't you agree…") | `hygiene_fail` | architect |
| broken grid (matrix stem + scale-only options) | `hygiene_fail` | architect |
| single_choice missing Other/None where the set isn't a complete scale | `hygiene_fail` | architect |
| Likert wrong length / non-neutral midpoint / non-bipolar ends | `hygiene_fail` | architect |
| Likert polarity inconsistent with survey majority | `hygiene_fail` | architect |
| null / out-of-range percentage | — | simulator |
| sum-to-100 type off by >1 | — | simulator |
| flat uniform distribution (≥3 identical values) | — | simulator |
| degenerate 100/0/0 single_choice | — | simulator |
| multiple_choice summing to ~100 | — | simulator |

**Dedup** is split three ways and then collapsed:

- `_exact_text_duplicates` — identical normalized text.
- `_near_exact_duplicates` — `is_near_exact` (Jaccard ≥ 0.78 on
  segment-stripped tokens, or ≥ 0.70 text + ≥ 0.60 option overlap), union-find
  clustered. Also unions all geo-residence questions together.
- `_heuristic_duplicates` — loose Jaccard ≥ 0.58, skipping pairs already
  near-exact — merged with the LLM judge's `duplicates`.

Soft clusters are then **treated as hard failures** (`uniqueness_fail`), which
is how the README's "no duplicates or highly similar questions" rule is
enforced. Thresholds live in `src/question_similarity.py`.

**Score and pass:**

```python
score = clean_questions / total_questions      # capped 0.5 on misattribution, 0.4 on auto_fail
passed = score >= 0.85 and not (auto_fail or misattribution or quota_fail
                                or has_geo_screener or hygiene_fail or hard_dup_fail)
```

Quota check: per tab below `QUESTIONS_PER_TAB_MIN` (5) → `quota_fail` in quota
mode; below TARGET (8) but ≥ MIN → soft note only.

On failure it increments `revision_count` — this node is the single source of
truth for that counter — and `graph.route_after_validator` picks the next node
by inspecting `fix_target` values and scanning the feedback blob for keywords
(`quota`, `under target`, `irrelevant`, `geo-residence`, `geography`). Priority
is architect > personas > simulator. `MAX_REVISIONS = 3` then forces assembler.

### `assembler` — pure merge, no LLM

1. `_drop_duplicates` — banned screeners (geo/age/income/brand) are **always**
   dropped; hard duplicate extras drop down to MIN even if that breaks quota;
   soft duplicates only drop while the tab stays at/above TARGET.
2. Group by tab in `CATEGORY_PLAN` order, sort by `funnel_position` with
   evidence-aligned as tie-break, renumber `q1..qN` globally.
3. `_diversify_section_charts` — reassigns each question the **least-used
   eligible** chart kind so a section spreads across all 8 frontend `ChartKind`
   values. This overrides whatever the architect chose.
4. `_to_frontend_question` — builds the frontend contract
   (`id/qNum/question/options/chart/data/insight/distributionNote/N/…`).
   `insight` is always `"Top response: <label> (<pct>%)"`; the persona rationale
   goes in `distributionNote`. `N` is a display-only 300–500 multiple of 10,
   deterministically seeded per `segment:region:question-id`; the real
   `sample_size` stays 2500.
5. `dataByRegion`: for a single-region job it publishes **only** that region's
   slug. For global/multi-region jobs it prefers real simulated values, and only
   falls back to `_REGION_BIAS` keyword-multiplier + seeded-noise approximation
   when nothing was simulated — it explicitly refuses to fabricate continents a
   job never ran.
6. `_build_target_customer` — turns `audience_profile` into the `targetCustomer`
   block emitted **first** in the JSON, with a note stating age/income/brand
   items are research context only, never asked.
7. Metadata: totals, counts, grounded percentages, `revision_count`,
   `grounding_thin`, `warnings`, `chart_kinds_used`.

### `global_selector` — A6, selection not generation

Runs outside LangGraph, reading the *published* region files from
`public/surveys/<slug>/`.

- Flatten all regional frontend questions with section + region provenance.
- `_cluster_near_exact` groups the same question across regions (same
  `section_id` only).
- `_score_cluster` = `coverage×0.30 + layer_bonus + grounded(0.20) +
  hygiene×0.15`, where `layer_bonus` is core 0.35 / full 0.18 / module 0.12.
- `_pick_clusters` in `prefer_core` mode fills CORE_PER_TAB from core clusters
  first, then modules, then anything, rejecting any candidate near-exact to an
  already-selected one.
- `_blend_data` = population-weighted mean using `REGION_BLEND_WEIGHTS`
  (NA .22 / EU .22 / APAC .32 / LatAm .12 / MEA .12), union of labels across
  regions, renormalized to 100 when the total lands in [90, 110].
- Writes `global.json`, rewrites `index.json` with the ready regions in
  canonical order, and refreshes the legacy flat `<slug>.json`.

---

## 4. Where each product rule is actually enforced

| Rule | Prompt | Code |
| --- | --- | --- |
| 5–10 Q/tab | architect `count` per chunk | `question_plan` clamps; validator `quota_fail` at <MIN; assembler warning at <TARGET |
| Segment fit | architect SUBJECT RULE | validator LLM judge check 4 (no mechanical check) |
| Section fit | `_TAB_GUIDANCE` per tab | none |
| **Story order** | planner arc + architect STORYLINE RULES | `narrative.order_questions` (architect, merge, assembler) + A6 `canonical_rank` |
| Uniqueness | architect ABSOLUTE UNIQUENESS + avoid-list | `question_similarity` + 3 dedup passes + `hard_dup_fail` + unconditional `_drop_duplicates` + `_enforce_absolute_uniqueness` + A6 cross-section `_accept` |
| No age/income | architect BANNED CONTENT | `_AGE_INCOME_RE` in architect, validator, **and** assembler |
| No brand/company | architect BANNED CONTENT | `_BRAND_COMPANY_RE` ×3, plus the ≥3-brandish-options heuristic |
| No geo-residence | architect REGION RULE | `_GEO_RESIDENCE_RE` ×3 (assembler's variant is slightly narrower) |
| targetCustomer first | — | `build_survey_json` key order + `build_region_payload` |
| Display N 300–500 ×10 | — | `_display_n` |
| sample_size 2500 | — | `SAMPLE_SIZE` constant |
| Percentages 2dp, sum 100.00 | simulator rule 2 | `_enforce_sums` largest-remainder |
| Likert balance/polarity | architect HYGIENE | `_likert_balanced`, `_likert_direction` |
| Chart variety | architect chart guidance | `_diversify_section_charts` (overrides the model) |

The three ban regexes are **duplicated verbatim** in `question_architect.py`,
`validator_critic.py`, and (two of them) `assembler.py`. Editing one without the
others is the most likely way to introduce an inconsistency.

---

## 5. Caching

`src/cache_store.py` — plain disk JSON under `.cache/survey_agent/<ns>/<sha>.json`,
never raises, atomic via tmp+replace.

| Namespace | Key parts | Effect |
| --- | --- | --- |
| `mkp` | schema, segment, region, first 80 facts | region-scoped |
| `personas` | schema, segment, region, mkp notes, confidence | region-scoped |
| `questionnaire` (per-region) | schema, segment, region slug, mode, pain points, persona ids | region-scoped |
| `questionnaire` (core) | schema, segment, `"core"`, mode | **shared across all regions** |

Cache is bypassed on revisions. The per-region questionnaire cache also requires
`≥ MIN_TOTAL_QUESTIONS` before it's accepted as a hit.

---

## 6. Reliability layer

- **LLM** (`src/llm.py`): DeepSeek gets a custom structured path — the Pydantic
  JSON schema is appended to the prompt and `repair_json_text` strips fences,
  takes the outermost `{...}`, and peels up to 5 trailing braces (DeepSeek emits
  extras). Other providers use LangChain `with_structured_output`. Both are
  wrapped by `_call_provider_sync/async`: acquire a token, retry up to 5 times
  on 429/timeout with jittered exponential backoff capped at 8s.
- **Rate limit** (`src/ratelimit.py`): Redis token bucket, refill+take in one
  atomic Lua script so all workers share it. **Fails open** — if Redis is down
  it sleeps `1/rate` locally and returns, which is what makes `run.py` work
  without Redis.
- **Retries**: RQ `Retry(max=2, interval=[30,120])` on region jobs,
  `[15,60]` on the global fan-in.
- **Stagger**: `job.meta["stagger_index"] × STAGGER_SECONDS`, capped at
  `STAGGER_CAP_SECONDS`.
- **Writes**: `atomic_write_text` (tmp + fsync + `os.replace`) everywhere
  published, so a reader never sees partial JSON.
- **Publish guard**: `_frontend_parent_ready()` — if the sibling frontend
  checkout isn't there, publish is a no-op returning `None` rather than an error.

---

## 7. Dead and half-dead code

Worth knowing before changing anything in this area.

**Fully unused modules** — imported by nothing but themselves:

- `src/nodes/answer_estimator.py` (24 KB) — the pre-persona-simulation node.
  Superseded by `survey_simulator`.
- `src/nodes/evidence_indexer.py` (14 KB) — would have produced `evidence_index`.

**`evidence_index` is always `{}`.** `question_architect` and `validator_critic`
both read `state.get("evidence_index") or {}`, but no node in either graph
writes it. Consequences:

- In `_tab_evidence_from_mkp`, the `grounded_facts` / `by_tab` /
  `context_facts` branches never execute. The architect's `figures_json` is
  built **only** from `mkp.top_purchase_drivers`, and `voice_json` only from
  MKP `top_pain_points` / `consumer_language` / `feature_requests`.
- The raw `evidence_pool` therefore never reaches the architect directly — it
  only reaches it compressed through the MKP.

**The grounding/attribution system is inert.** `survey_simulator._build_answered`
hardcodes every answer's `grounded_in: None`, `source_market: None`,
`grounding_label: ""`. Nothing else constructs answers. So in `validator_critic`:

- the `if gi:` branch never runs → `misattribution` is never `True`, `adjacent`
  is always 0, `adjacent_grounded_pct` is always 0.0;
- `grounded` / `direct` are incremented purely by `confidence in (high, medium)`,
  so **`grounded_pct` is "% of options the simulator was ≥medium confident
  about"**, not an evidence-grounding rate (the docstring says as much);
- the anti-fabrication checks (`grefs`) never fire;
- judge prompt checks 2 (label integrity) and 3 (mis-mapped grounding) ask about
  fields that are always null;
- `assembler`'s `_HIGH_ADJACENT_PCT` warning can never trigger.

The module docstrings and the `Critique` fields still describe this as the
highest-priority failure mode, which is the single most misleading thing in the
codebase.

**Other stale bits:**

- `build_graph()` — compiled but never called.
- `survey_io.publish_survey_bundle()` — never called.
- `_display_n`'s docstring says "25k–30k"; the code produces 300–500.
- `route_after_validator` handles a `"estimator"` target, but
  `validator_critic._resolve_target` can only ever emit
  architect/personas/simulator.
- The per-region legacy flat file `<slug>.json` is overwritten by each of the
  five region jobs in turn, then again by A6 — only the last write survives.

---

## 8. Things to know before running it

- `src/config.py` raises at **import time** if `NEON_CONNECTION_STRING` is
  missing or the active provider's key is unset. Any `from src...` import fails
  until `.env` exists.
- `FRONTEND_SURVEYS_DIR` defaults to `../cmi-platform-ai/frontend-v2/apps/web/
  public/surveys` — a sibling checkout that isn't in this repo. Without it,
  region files are never published and **A6 will raise `FileNotFoundError`**,
  because `global_selector` reads its inputs from the published directory, not
  from `output/`.
- `RUN_REGIONS` filters which continents run *and* which A6 blends, but
  `run_one_async` / `_enqueue_regional_jobs` both iterate the full
  `GEOGRAPHIC_REGIONS`, not `RUN_GEOGRAPHIC_REGIONS` — the env var only
  narrows the A6 load step.
- `GENERATION_MODE` is a hardcoded `"quota"` in `question_plan.py`; "honest"
  mode is reachable only by editing the file.
- Tests: `tests/test_b2c_gate_fanout.py` is the only pytest-shaped file. Everything
  under `scripts/test_*.py` is a standalone phase-gate script run with `python`.
  Most need the repo root on `PYTHONPATH`; `scripts/test_a8_storyline.py`
  bootstraps its own `sys.path` and runs with no LLM or network.
- Two `scripts/test_*.py` files fail for reasons unrelated to any code change:
  `test_a6_global.py` asserts `QUESTIONS_PER_TAB_TARGET >= 10` (stale — the
  default is now 8) and aborts before its real select/blend tests run;
  `test_c_slug_audit.py` needs the sibling frontend checkout.

---

## 9. Phase A8 — the storyline (question order)

Added after the initial read. Order used to be emergent; now it is planned.

**`src/narrative.py`** — the ordering contract.

- `CANONICAL_BEATS` — the standard arc per section (5–7 beats each). Serves three
  jobs: the shared vocabulary, the cross-region anchor, and the fallback when the
  planner LLM fails.
- `allocate_questions` / `chunk_allocations` — spread a tab's quota across beats;
  earlier beats win the remainder, so a short tab still opens correctly. Chunks
  never straddle the arc out of order.
- `order_questions` — **the single ordering function.** Sort key is
  `(beat order, within-beat hint, generation order)`; stamps a dense
  `narrative_order` / `funnel_position` and backfills `beat_title` /
  `beat_canonical`.
- `remap_questions_to_beats` — moves a question onto a *different* blueprint's
  beats via the canonical anchor, then title tokens, then nearest canonical rank.

**`src/nodes/story_planner.py`** — one LLM call between `persona_generator` and
`question_architect`. Produces 3–8 ordered beats per section, adapted to the
category and the region, each declaring the canonical beat it maps to. Cached per
`(segment, region)`. Failure → canonical arc.

**Per-region storylines.** Each region plans its own arc (chosen deliberately —
local shopping reality reshapes the story). The `canonical` tag on every beat is
what keeps this from wrecking A6: it lets five independently-planned regions be
aligned on the same narrative moment.

**Ordering is enforced at four points**, so no single failure can scramble it:

1. `_generate_tab` — questions are generated per beat with explicit quotas, then
   `order_questions` runs before the tab is returned. A wrong `beat_id` is
   repaired onto the chunk's beats (`_nearest_beat_id`).
2. Core+module merge — CORE is written against the canonical arc, remapped onto
   the region's storyline, then merged **by beat**. A regional module question
   about an early beat now sits early instead of being appended last.
3. `assembler` — sorts by `narrative_order` and re-densifies to 1..N after any
   dedup drop.
4. `global_selector` — clusters carry `beat_canonical`; selection takes at most
   one cluster per beat first (so global covers the whole arc), and the final
   sort is by `narrative.canonical_rank`.

**Absolute uniqueness** now has no escape hatch:

- `_drop_duplicates` drops every flagged extra unconditionally — quota no longer
  protects a soft duplicate (it previously did, whenever dropping one would take
  a tab below target).
- `_enforce_absolute_uniqueness` is a final survey-wide sweep across all four
  tabs, keeping the first of any similar group. It catches exact text, near-exact,
  loose-Jaccard, and "same options + same type + overlapping wording" repeats —
  with a deliberate guard so two different Likert items sharing one scale are not
  falsely merged.
- A6 `_pick_clusters` accumulates a `taken` list across sections, so the global
  file cannot repeat a question in a different tab.

**Bugs fixed along the way:**

- `token_set_jaccard` returned `0.0` whenever segment-stripping emptied a token
  set, so short on-topic paraphrases ("How often do you buy organic milk?" vs
  "How frequently do you purchase organic milk?") scored as maximally
  *dissimilar* and shipped as duplicates. Now falls back to unstripped tokens.
- `question_layer` was dropped by `survey_simulator` (`AnsweredQuestion` had no
  such field), so every published question was `"full"` and A6's core bonus
  (0.35 vs 0.18) never applied. Now carried through.
