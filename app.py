"""B2C survey generator — operator UI.

# change for b2c questionarie

Run:
    .venv\\Scripts\\streamlit.exe run app.py

Deliberately thin: it collects the market and the geographies, shells out to the
same `run.py` the CLI uses, and lists what landed in `output/`. Generation logic
lives in the pipeline, not here, so the UI cannot drift from what the CLI does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import geography  # noqa: E402
from src.cmi_match import match_domain, section_labels  # noqa: E402

OUTPUT = ROOT / "output"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

st.set_page_config(page_title="CMI B2C Survey Generator", page_icon="📋",
                   layout="wide")

st.title("CMI B2C Survey Generator")
st.caption(
    "Consumer segmentation surveys — CMI survey categories, "
    "region and country coverage."
)

# --------------------------------------------------------------------------
# 1. Market
# --------------------------------------------------------------------------
st.subheader("1 · Market")
market = st.text_input(
    "Market segment",
    value="global running shoes market",
    help="e.g. 'global running shoes market', 'organic milk', 'smartwatches'",
)

if market.strip():
    match = match_domain(market)
    conf = match.get("confidence")
    badge = {"subsector": "🟢", "domain": "🟡", "default": "⚪"}.get(conf, "⚪")
    cols = st.columns([1, 1, 2])
    cols[0].metric("CMI domain", match.get("domain") or "—")
    cols[1].metric("Sub-sector", match.get("sub_sector") or "not matched")
    cols[2].markdown(
        f"{badge} **Framework match: {conf}**  \n"
        + ("Matched a named CMI sub-sector." if conf == "subsector"
           else "Matched the domain only." if conf == "domain"
           else "No taxonomy match — using the Consumer Goods framework.")
    )
    with st.expander("Survey categories this market will use", expanded=True):
        for i, (_cid, label) in enumerate(section_labels(match), start=1):
            st.write(f"**{i}.** {label}")

st.divider()

# --------------------------------------------------------------------------
# 2. Geography
# --------------------------------------------------------------------------
st.subheader("2 · Geography")
st.caption(
    "Surveys are generated per region, plus the top 2 countries in each "
    "selected region."
)

regions = st.multiselect(
    "Regions",
    options=geography.ALL_REGIONS,
    default=["North America"],
)

include_countries = st.checkbox(
    "Also generate the top 2 countries per region", value=True,
)

selected_countries: dict[str, list[str]] = {}
if regions:
    st.write("**Countries**")
    cols = st.columns(min(len(regions), 3))
    for i, region in enumerate(regions):
        with cols[i % len(cols)]:
            st.markdown(f"*{region}*")
            default = geography.countries_for(region, top_only=True)
            picked = st.multiselect(
                f"countries_{region}",
                options=geography.countries_for(region, top_only=False),
                default=default if include_countries else [],
                label_visibility="collapsed",
                key=f"c_{region}",
            )
            selected_countries[region] = picked

n_files = len(regions) + sum(len(v) for v in selected_countries.values())
if regions:
    st.info(
        f"**{n_files} survey file(s)** will be generated "
        f"({len(regions)} region + "
        f"{sum(len(v) for v in selected_countries.values())} country). "
        f"Roughly {n_files * 5}–{n_files * 7} minutes."
    )

st.divider()

# --------------------------------------------------------------------------
# 3. Run
# --------------------------------------------------------------------------
st.subheader("3 · Generate")

col_a, col_b = st.columns([1, 3])
fresh = col_a.checkbox(
    "Ignore cache", value=True,
    help="Clears the questionnaire and storyline caches first. Required after "
         "any prompt or rule change — the cache key does not include prompt "
         "text, so edits are otherwise silently ignored.",
)

if col_b.button("Generate surveys", type="primary", disabled=not (market and regions)):
    if fresh:
        for ns in ("questionnaire", "storyline"):
            d = ROOT / ".cache" / "survey_agent" / ns
            if d.is_dir():
                for f in d.iterdir():
                    f.unlink(missing_ok=True)
        st.write("Cleared questionnaire + storyline cache.")

    env = dict(os.environ)
    env["RUN_REGIONS"] = ",".join(regions)
    env["PYTHONIOENCODING"] = "utf-8"

    log = st.empty()
    with st.spinner(f"Generating — {market!r} across {len(regions)} region(s)…"):
        proc = subprocess.run(
            [str(PY), "run.py", market],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
    log.code(tail or "(no output)", language="text")

    # The Global blend step needs all five regions on disk and raises when only
    # a subset was run. The region survey itself is already written by then.
    if proc.returncode == 0 or "No regional survey files" in (proc.stdout or ""):
        st.success("Generation finished.")
    else:
        st.error(f"Run exited {proc.returncode}. See the log above.")

st.divider()

# --------------------------------------------------------------------------
# 4. Results
# --------------------------------------------------------------------------
st.subheader("4 · Files")

if OUTPUT.is_dir():
    files = sorted(
        [p for p in OUTPUT.glob("*.json") if not p.name.startswith("_")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:15]
    if not files:
        st.write("Nothing generated yet.")

    for p in files:
        pdf = p.with_suffix(".pdf")
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))["survey"]["metadata"]
            n_q = meta.get("total_questions", "?")
            segs = [s.get("name") for s in (meta.get("segments") or [])]
        except Exception:  # noqa: BLE001
            n_q, segs = "—", []

        with st.expander(f"{p.stem}  ·  {n_q} questions", expanded=False):
            if segs:
                st.caption("Segments: " + " · ".join(str(s) for s in segs))
            c = st.columns([1, 1, 2])
            c[0].download_button("Download JSON", p.read_bytes(), p.name,
                                 key=f"j_{p.name}", mime="application/json")
            if pdf.is_file():
                c[1].download_button("Download PDF", pdf.read_bytes(), pdf.name,
                                     key=f"p_{p.name}", mime="application/pdf")
            elif c[1].button("Make PDF", key=f"m_{p.name}"):
                with st.spinner("Rendering…"):
                    subprocess.run(
                        [str(PY), "scripts/export_survey_pdf.py", str(p)],
                        cwd=str(ROOT), capture_output=True, text=True,
                    )
                st.rerun()

            # Read the questions straight out of the file — the point of the UI
            # is reviewing them here rather than opening a PDF viewer.
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                for sec in doc.get("frontend", {}).get("sections", []):
                    qs = sec.get("questions") or []
                    if not qs:
                        continue
                    st.markdown(f"**{sec.get('title') or sec.get('id')}**")
                    for q in qs:
                        text = q.get("question") or q.get("text") or ""
                        st.write(f"- {text}")
                        opts = [
                            (o.get("label") if isinstance(o, dict) else o)
                            for o in (q.get("options") or [])
                        ]
                        if opts:
                            st.caption("   " + "  ·  ".join(str(o) for o in opts[:8]))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not read questions: {exc}")
