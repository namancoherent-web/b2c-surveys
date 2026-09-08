# Survey Agent

An agentic market-research **survey generator** built on
[LangGraph](https://github.com/langchain-ai/langgraph). Given a market segment,
it produces a **consulting-grade B2C questionnaire**: about **5–10 questions per
section** (default **8** → ~**32** total across 4 tabs), highly specific to the
segment, with synthetic answer-% distributions **grounded in retrieved evidence
where possible** (CMI database + web).

## Quality bar (product rules)

| Rule | Behavior |
| --- | --- |
| Count | **5–10** Q/tab (`QUESTIONS_PER_TAB_MIN/MAX/TARGET`, default 5 / 10 / **8**) |
| Segment fit | Every stem + options about the **exact** target product/category |
| Section fit | Each tab stays in remit (profile / behavior / preferences / satisfaction) |
| **Story order** | Questions run in a planned narrative order — each fills a **storyline beat**, broad → specific → sensitive. Order is decided in code, never by model output |
| Uniqueness | **Absolute** — no question repeats anywhere in the survey, across sections, even reworded. Quota never protects a duplicate |
| Banned items | **Age / income**; **brand or company names** (stem or options); geo-of-residence |
| Target customer | `targetCustomer` object at the **start** of the JSON (who is interviewed) |
| Display `N` | Per question, integer **300–500**, **multiple of 10** |
| Backend sample | Rich `sample_size` stays **2500** (simulator panel size) |
| Percentages | **2 decimals**; single_choice / ranking / likert sum to **100.00** |

`insight` on each frontend question is always `"Top response: …"`. Persona/%
rationale lives in `distributionNote`.

## The 9-node architecture (persona simulation)

1. **scope_classify** — normalize segment, B2C/B2B/other + confidence; B2C gate (hard-stop high-confidence pure B2B). Runs once before fan-out.
2. **evidence_harvester** — CMI DB + Exa/YouTube/Reddit/DDG (Jina page fetch).
3. **evidence_synthesizer** — compress into a Market Knowledge Profile (MKP).
4. **persona_generator** — 10–20 archetypes + regional weight packs.
5. **story_planner** — plan the survey's **narrative arc**: ordered storyline beats per tab, adapted to the category and region (Phase A8).
6. **question_architect** — ~8×4 questions generated *into* beats; region jobs use core+module (`REGION_QUESTION_MODE`).
7. **survey_simulator** — persona-weighted panel simulation (N=2500), per region.
8. **validator_critic** — math + hygiene + uniqueness + story coverage; revise simulator/architect/personas.
9. **assembler** — `targetCustomer` + frontend JSON (`N`, `distributionNote`), published in story order.

### Storyline (Phase A8)

Sections are fixed (profile → behavior → preferences → satisfaction). The
ordered **beats** inside each section are planned per segment *and* per region,
so a consumable, a durable and a subscription each get the arc that fits them.

- `src/narrative.py` — canonical arc (the shared beat vocabulary + fallback),
  allocation, and the single ordering function.
- Every question carries `beat_id`; publish order is
  `(beat order, position within beat)` — computed in code, so a chunked or
  retried generation can never scramble the sequence.
- Each beat declares a `canonical` id. Because regions plan independently, this
  is the anchor that lets A6 line the same narrative moment up across all five
  regional files and keeps the shared CORE instrument in its right slot.
- The storyline ships in `metadata.storyline`, keyed by frontend section id.

Regression suite: `python scripts/test_a8_storyline.py` (no LLM, no network).

Percentages are **not** pasted from reports. Reports shape MKP + persona mixes;
YouTube/Reddit/Twitter shape language/pains only (never numeric bands).

**Facebook** is not implemented — public FB search is not feasible without a
partnership.

### Region jobs (Phase A4b/A5) + B2C gate

Input is **segment only**. Flow:

1. `classify_and_fanout` / `run.py` runs `scope_classify` **once**
2. High-confidence pure B2B/other → `reject_stop` (writes `output/<slug>_rejected.json`; no region files)
3. B2C (and lenient hybrids/ambiguous) → fan out the fixed **5 continents**, each starting at `evidence_harvester` (no re-classify)
4. A6 global blend after all five

```bash
python run.py "Organic Milk"              # classify → 5 regions + global.json
python run.py "industrial hydraulic seals"  # rejected at gate
```

Hybrids (B2B market with clear consumer proxy) still **proceed** with a consumer-proxy audience (`B2C_GATE_STRICT=false` default). Set `B2C_GATE_STRICT=true` for stricter rejection.

With `REGION_QUESTION_MODE=core_plus_module` (default): shared CORE (~6/tab) +
regional MODULE (~2/tab). Facts and personas are region-scoped; `question_layer`
marks `core` vs `module`.

### Global selection (Phase A6)

After all region files exist, `run_global_selection` **selects** (no new
generation) and **blends** percentages:

- Pool regional questions → near-exact dedup → score by coverage / layer /
  grounding / hygiene → pick ~target per tab (5–10 band)
- Global `%` = population-weighted mean of the same question’s regional sims
- Writes `global.json` and finalizes `index.json`

RQ: `enqueue.py` enqueues one `classify_and_fanout` job per segment; that worker
dynamically enqueues the 5 region jobs + a fan-in job with `depends_on`.

### Published survey layout (Phase A7)

Each run writes to the frontend `public/surveys/` directory:

```
public/surveys/<slug>.json              # legacy (survey-preview) — kept for now
public/surveys/<slug>/
  index.json                            # {segment, slug, regions, generatedAt}
  global.json                           # A6 select+blend (no new generation)
  north-america.json
  europe.json
  asia-pacific.json
  latin-america.json
  middle-east-africa.json
```

Per-region files put **`targetCustomer` first**, then `segment` / `slug` /
`region` / `frontend` / `metadata`. `frontend.sections` already have region-
projected `data` (no `dataByRegion` in published files).

### Canonical region slugs (Part C)

Agent `PUBLISH_REGION_SLUGS` and frontend `geographies.ts` `REGIONS` /
`SURVEY_REGION_SLUGS` must stay byte-identical:

`global` · `north-america` · `europe` · `asia-pacific` · `latin-america` · `middle-east-africa`

Audit: `python scripts/test_c_slug_audit.py`

### The 4 tabs

| Tab | Holds |
| --- | --- |
| `consumer_profile` | Category relationship — usage context, occasions, frequency, need-state (**not** age/income/residence). |
| `buying_behavior` | How they buy — triggers, channels, category spend, decision drivers (**not** brand/company name lists). |
| `preferences_expectations` | What they want — attributes, formats, claims, trade-offs (no named competitors). |
| `satisfaction_future_intent` | Outcomes — satisfaction, friction, switching, recommend / future use. |

## Project layout

```
survey_agent/
├── run.py                      # CLI: classify → 5 regions + global (no Docker)
├── enqueue.py                  # RQ: classify_and_fanout per segment
├── requirements.txt
├── .env.example
├── AGENT_ARCHITECTURE.md
├── scripts/
│   └── export_survey_pdf.py    # JSON → PDF (targetCustomer + Qs + charts)
├── src/
│   ├── config.py
│   ├── web_search.py           # orchestrates Exa + Reddit + DDG by angle
│   ├── diagnose_sources.py     # pre-flight source health check
│   ├── question_plan.py        # tab counts, regions, SAMPLE_SIZE
│   ├── question_similarity.py  # near-exact / loose dedup thresholds
│   ├── fetchers/
│   │   └── jina_reader.py      # clean page fetch (r.jina.ai)
│   ├── searchers/
│   │   ├── exa_search.py       # semantic search (Exa free tier)
│   │   ├── reddit_search.py    # consumer voice (rdt-cli)
│   │   └── ddg_pool.py         # DuckDuckGo fallback pool
│   └── nodes/
└── output/
```

## Setup

From the `survey_agent/` directory:

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
```

Fill in `.env`:

| Variable | Required | Notes |
| --- | --- | --- |
| `NEON_CONNECTION_STRING` | Yes | CMI Postgres |
| `DEEPSEEK_API_KEY` | Yes (default provider) | Native DeepSeek API — [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| `LLM_PROVIDER` | Optional | `deepseek` (default) \| `openrouter` \| `anthropic` |
| `EXA_API_KEY` | Recommended | Free at [exa.ai](https://exa.ai) — primary semantic search |
| `REDDIT_CLI_COOKIE_PATH` | Optional | Path to exported rdt-cli cookies |
| `REDDIT_ENABLED` | Optional | `true` by default |
| `QUESTIONS_PER_TAB_MIN` | Optional | Default `5` |
| `QUESTIONS_PER_TAB_MAX` | Optional | Default `10` |
| `QUESTIONS_PER_TAB_TARGET` | Optional | Default `8` |
| `RUN_REGIONS` | Optional | Empty = all 5 continents in A6 blend; e.g. `Europe,North America` to limit |

### Exa API key (free tier)

1. Sign up at [https://exa.ai](https://exa.ai)
2. Create an API key
3. Set `EXA_API_KEY=exa_...` in `.env`

### Reddit / rdt-cli (consumer voice)

Reddit is the best source for real complaint language on B2C segments, but it is
**fragile** — cookie auth can break or accounts can be rate-limited.

#### Windows (recommended — manual cookie paste)

On Windows, `rdt login` often fails with **"No Reddit cookies found"** because
reading Edge/Chrome cookies requires **Administrator** privileges or hits Chrome
encryption limits. Use manual setup instead:

1. Log into https://www.reddit.com in **Edge**
2. Press **F12** → **Application** → **Cookies** → `https://www.reddit.com`
3. Click **`reddit_session`** and copy its **Value**
4. Run:

```powershell
.venv\Scripts\python.exe scripts\setup_reddit_cookies.py
```

5. Add to `.env`:

```
REDDIT_CLI_COOKIE_PATH=./secrets/reddit-cookies.json
REDDIT_ENABLED=true
```

#### Alternative: rdt login (macOS/Linux, or Windows as Admin)

```bash
pip install rdt-cli
rdt login          # may need "Run as Administrator" on Windows + Edge
rdt status
```

If Reddit stops working, set `REDDIT_ENABLED=false` — the pipeline continues
with Exa + DuckDuckGo only.

### Pre-flight diagnostics

```bash
python -m src.diagnose_sources
```

Checks Jina Reader, Exa, Reddit, DDG, and the merged orchestrator.

## Run without Docker (all 5 regions)

Recommended for a single segment on your laptop. No Redis / RQ required.

```powershell
cd survey_agent
.\.venv\Scripts\Activate.ps1
# Leave RUN_REGIONS empty in .env for all 5 continents in the global blend
.\.venv\Scripts\python.exe run.py "global avocado oil market"
```

This **classifies once**, then runs **North America → Europe → Asia Pacific →
Latin America → Middle East & Africa** sequentially, then **A6 global**.

Outputs:

- `output/global-<slug>_<region>.json` (and timestamped copies on re-run)
- `output/global-<slug>_global.json`
- Published under `cmi-platform-ai/frontend-v2/apps/web/public/surveys/<slug>/`

Preview (with frontend running):

```text
http://localhost:3000/survey/<slug>/global
http://localhost:3000/survey/<slug>/asia-pacific
```

## PDF export

```powershell
# All region JSONs under output/final_avacado_oil/ → PDF beside each file
.\.venv\Scripts\python.exe scripts\export_survey_pdf.py --final-avocado

# Or any single file / sneakers / avocado helpers
.\.venv\Scripts\python.exe scripts\export_survey_pdf.py --sneakers
.\.venv\Scripts\python.exe scripts\export_survey_pdf.py --avocado
.\.venv\Scripts\python.exe scripts\export_survey_pdf.py output\some-file.json
```

PDF layout: title → **Target customer** (all `targetCustomer` keys except
`note`) → questions in `qNum` order with options, chart, `distributionNote`,
and per-question `N`.

## Web search source routing

| Query angle | Primary | Secondary | Fallback |
| --- | --- | --- | --- |
| trends, pricing, competitors, market_share | Exa | — | DDG |
| complaints, needs, customer_language, reviews | Reddit | Exa | DDG |
| default | Exa | — | DDG |

All page fetches use **Jina Reader** (`r.jina.ai`) with requests+BS fallback.

## Parallel batch (RQ + Docker)

Run multiple segments in parallel without changing the LangGraph pipeline. Worker
replicas control **how many segments run at once**; rate limits (Phase 3+) control
**API calls per second**.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows)
- `.env` filled in (same keys as single-run CLI)
- Optional: set `FRONTEND_SURVEYS_HOST_DIR` in `.env` to the absolute host path of
  `cmi-platform-ai/frontend-v2/apps/web/public/surveys` (compose mounts it into workers)

### 1. Install queue deps (host, for `enqueue.py`)

```powershell
pip install -r requirements.txt
```

### 2. Start Redis + worker + dashboard

```powershell
docker compose up --build
# Keep 5 workers in parallel:
docker compose up -d --build --scale worker=5
```

- Dashboard: http://localhost:9181
- Redis on host: **`localhost:6380`** (mapped from container 6379; avoids clash with other stacks)

Always pass `--scale worker=5` if you want five workers — a plain `docker compose up`
scales back to **1** worker and removes extras.

### 3. Enqueue jobs (from host)

```powershell
$env:REDIS_URL = "redis://localhost:6380"
.\.venv\Scripts\python.exe enqueue.py segments-avocado-oil.json
# or: enqueue.py segments-one.json / segments.json
```

Jobs call `classify_and_fanout` (classify once → enqueue 5 regions + global).
CLI alternative (no queue): `python run.py "<segment>"`.

### 4. Monitor

- **RQ Dashboard** — job status, failures, retries: http://localhost:9181
- **Worker logs** — `docker compose logs -f worker`
- **Preview** — `pnpm dev:web` in `cmi-platform-ai/frontend-v2`, then
  `http://localhost:3000/survey/<slug>/global`

### Phase 2 pass criteria

1. `python enqueue.py segments-one.json` prints `enqueued ... -> <job-id>`
2. Dashboard shows job → finished
3. Region + global JSON appear under `output/`
4. Survey JSON copied to the frontend `public/surveys/` mount
5. CLI still works: `python run.py "global skincare market"`

### Layer 2 — LLM rate limit (Phase 3)

Workers share a Redis token-bucket (`src/ratelimit.py`) so parallel DeepSeek calls
do not stampede into 429s. Set in `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_RATE_PER_SEC` | `5` | Steady LLM tokens/sec across **all** workers |
| `LLM_BURST` | `10` | Short burst capacity |

Rebuild workers after code changes:

```powershell
docker compose up -d --build --scale worker=2
```

If 429s reappear under load, **lower `LLM_RATE_PER_SEC`**, not worker count.

### Stagger + web rate limit + VPN rotate (Phase 4)

| Dial | Env | Default | Effect |
| --- | --- | --- | --- |
| Start desync | `STAGGER_SECONDS` | `30` | RQ job `i` sleeps `min(i * 30, STAGGER_CAP_SECONDS)` before pipeline |
| Web throttle | `WEB_RATE_PER_SEC` / `WEB_BURST` | `1` / `3` | Shared Redis bucket on DDG + Jina across all workers |
| IP rotate | `VPN_ROTATE_*` | on | On DDG/Jina block → `POST host.docker.internal:8765/rotate` |

**Host VPN tool** (sibling checkout):

```powershell
cd C:\Users\Ambika Soni\Downloads\coherent_b2c_survey_question_2\automation-tool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

In the UI set mode to **On scraper block**. Leave it running while workers fetch.

**Egress gotcha (§6):** rotating the host IP only helps if container traffic exits through Proton.

```powershell
docker compose exec worker python scripts/verify_vpn_egress.py
```

- IP changes → good.
- IP unchanged → document/fix networking (route Docker through Proton, or run fetch on host). Set `VPN_ROTATE_ENABLED=false` to disable without crashes.

### Job retries (Phase 5)

Whole-job retries are set in `enqueue.py`:

```python
retry=Retry(max=2, interval=[30, 120])  # up to 2 retries after first failure
```

A failing segment retries, then lands in RQ’s **Failed** registry; other jobs keep running.

Smoke test (workers must include latest `src/tasks.py`):

```powershell
docker compose up -d --build --scale worker=2
.\.venv\Scripts\python.exe scripts\test_rq_retry.py
```

Pass: script prints `PASS: failing job retried then failed; sibling job succeeded.`
Also visible at http://localhost:9181 (Failed + Finished).

### Scale to 5 workers (Phase 7)

```powershell
docker compose up -d --build --scale worker=5
$env:REDIS_URL = "redis://localhost:6380"
.\.venv\Scripts\python.exe enqueue.py segments.json
```

Monitor:

- Dashboard: http://localhost:9181  
- `docker compose logs -f worker`  
- `.\.venv\Scripts\python.exe scripts\batch_status.py` — `grounded_pct` per output file  

**Pass:** all 5 jobs finish; `grounded_pct` stays near your single-run baseline (not collapsed).  
If DeepSeek 429s return, lower `LLM_RATE_PER_SEC` in `.env` and recreate workers — **do not** lower worker count first.

Two dials:

| Dial | Control |
| --- | --- |
| Segments in parallel | `--scale worker=N` |
| API calls / sec | `LLM_RATE_PER_SEC` / `WEB_RATE_PER_SEC` |
# b2c-surveys
