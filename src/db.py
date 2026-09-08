"""Neon Postgres connectivity for the CMI database.

Connections use a ``connect_timeout`` because Neon free-tier endpoints suspend
when idle and cold-start on the first request, which can take several seconds.
Callers must always close the connection (use try/finally).
"""

import os
import re
import sys
import time

import psycopg2
from psycopg2 import OperationalError, sql

from src.config import NEON_CONNECTION_STRING
from src.evidence_utils import extract_value
from src.html_utils import clean_html

# Generous enough to absorb a Neon cold-start, short enough to fail clearly.
CONNECT_TIMEOUT_SECONDS = 15
_DB_CONNECT_RETRIES = int(os.getenv("DB_CONNECT_RETRIES", "3"))
_DB_CONNECT_BACKOFF = float(os.getenv("DB_CONNECT_BACKOFF_SEC", "2"))

# ---------------------------------------------------------------------------
# CMI schema names — CONFIRMED via information_schema (2026-06-30).
#
# These reflect the real columns, not assumptions. If the schema changes, update
# them here. Mixed-case identifiers (e.g. "newsId") are quoted via psycopg2.sql.
# ---------------------------------------------------------------------------
REPORTS_TABLE = "cmi_reports"
REPORTS_ID_COL = "newsId"           # primary key (joined as nid/rid elsewhere)
REPORTS_TITLE_COL = "newsSubject"   # report title
REPORTS_ACTIVE_COL = "IsActive"     # 1 = live report
# Text columns matched against the segment (no plain text category column exists).
REPORTS_MATCH_COLS = ("newsSubject", "keyword", "keywords", "segmentation")

# Generic words stripped before token matching, so e.g. "global organic milk
# market" matches on "organic"/"milk" rather than the whole phrase.
_SEGMENT_STOPWORDS = frozenset({
    "global", "worldwide", "international", "market", "markets", "industry",
    "sector", "report", "size", "share", "growth", "trends", "trend",
    "analysis", "forecast", "outlook", "the", "and", "for", "of", "by",
    "product", "products", "service", "services", "solution", "solutions",
    "type", "types",
})


# Product-category synonyms so niche consumer terms still hit CMI report titles
# (e.g. "sneakers" rarely appears; "footwear" / "athletic" do).
_TOKEN_SYNONYMS = {
    "sneaker": ("footwear", "athletic", "shoe", "shoes"),
    "sneakers": ("footwear", "athletic", "shoe", "shoes"),
    "trainer": ("footwear", "athletic", "shoe"),
    "trainers": ("footwear", "athletic", "shoe"),
    "shoe": ("footwear",),
    "shoes": ("footwear", "shoe"),
    "apparel": ("clothing", "wear"),
    "garment": ("apparel", "clothing"),
    "smartphone": ("mobile", "phone", "handset"),
    "phone": ("smartphone", "mobile"),
    "yogurt": ("yoghurt", "dairy"),
    "yoghurt": ("yogurt", "dairy"),
}


def _singularize(token: str) -> str | None:
    """Cheap English plural stem for DB recall (sneakers → sneaker)."""
    if len(token) <= 4:
        return None
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("zes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return None


def _tokenize_segment(segment: str) -> list:
    """Split a segment into significant lowercase tokens (stopwords dropped).

    Also expands plurals and a small synonym map so consumer slang (e.g.
    \"sneakers\") still matches industrial report titles (\"footwear\").
    """
    words = re.findall(r"[a-z0-9]+", (segment or "").lower())
    base = [w for w in words if len(w) > 2 and w not in _SEGMENT_STOPWORDS]
    out: list[str] = []
    seen: set[str] = set()
    for tok in base:
        candidates = [tok]
        stem = _singularize(tok)
        if stem and stem not in _SEGMENT_STOPWORDS:
            candidates.append(stem)
        for c in list(candidates):
            candidates.extend(_TOKEN_SYNONYMS.get(c, ()))
        for c in candidates:
            if len(c) > 2 and c not in _SEGMENT_STOPWORDS and c not in seen:
                seen.add(c)
                out.append(c)
    return out

SUMMARY_TABLE = "cmi_reportsummery"
SUMMARY_JOIN_KEY = "nid"             # -> cmi_reports.newsId
# Named free-text section columns worth mining, as (column, friendly_label).
# Scope/sources/boilerplate columns are intentionally excluded.
SUMMARY_CONTENT_COLUMNS = (
    ("market_size_trends", "market size & trends"),
    ("market_concentration_comp_landscape", "competitive landscape"),
    ("key_analyst_takeaways", "key analyst takeaways"),
    ("regional_insights", "regional insights"),
    ("key_developments", "key developments"),
    ("report_segmentation", "report segmentation"),
    ("market_dynamics", "market dynamics"),
    ("key_stakeholders", "key stakeholders"),
    ("segmental_insights", "segmental insights"),
    ("current_events_impact", "current events impact"),
    ("end_user_feedback", "end-user feedback"),
    ("ai_role_market", "AI role in market"),
    ("market_trends", "market trends"),
)
# insight{N}_header / insight{N}_description pairs (N = 1..5).
SUMMARY_INSIGHT_RANGE = range(1, 6)
SUMMARY_INSIGHT_HEADER = "insight{n}_header"
SUMMARY_INSIGHT_DESC = "insight{n}_description"

DYNAMIC_TABLE = "cmi_report_dynamic"
DYNAMIC_JOIN_KEY = "rid"             # -> cmi_reports.newsId
# Wide format: field_{N} / disc_{N} pairs (N = 1..20).
DYNAMIC_PAIR_RANGE = range(1, 21)
DYNAMIC_FIELD_PREFIX = "field_"
DYNAMIC_DESC_PREFIX = "disc_"


def get_connection():
    """Open a short-lived connection to Neon (close after each harvest use).

    Retries on connection failures so a VPN IP rotate mid-run can recover
    instead of killing the job. Always close the returned connection in a
    ``finally`` block.
    """
    last: BaseException | None = None
    for attempt in range(_DB_CONNECT_RETRIES):
        try:
            return psycopg2.connect(
                NEON_CONNECTION_STRING,
                connect_timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except (OperationalError, OSError) as exc:
            last = exc
            if attempt == _DB_CONNECT_RETRIES - 1:
                break
            time.sleep(_DB_CONNECT_BACKOFF * (attempt + 1))
    assert last is not None
    raise last


def test_connection() -> bool:
    """Run ``SELECT 1`` and print the outcome. Returns True on success."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        print("[db] Connection OK — SELECT 1 succeeded.")
        return True
    except Exception as exc:  # noqa: BLE001 — surface any failure to the operator
        print(f"[db] Connection FAILED: {exc}")
        return False
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Evidence query helpers (Node 2 — evidence_harvester).
#
# All queries are PARAMETERIZED. The segment is never string-formatted into SQL;
# dynamic *identifiers* (column/table names) are composed only from the trusted
# configurable constants above via psycopg2.sql, never from user input.
# ---------------------------------------------------------------------------


def find_reports(segment: str, conn, limit: int = 25) -> list:
    """Token-match reports to a segment, ranked by how many tokens hit.

    The segment is split into significant tokens (stopwords dropped). A report
    matches if ANY token appears in newsSubject/keyword/keywords/segmentation;
    results are ranked by the number of distinct tokens matched (more = better),
    then by recency (newsId desc). This raises recall over full-phrase matching
    so a segment like "organic milk market" also surfaces related reports.

    Returns a list of ``(id, title, category)`` tuples (up to ``limit``). There
    is no plain-text category column, so ``category`` is the report's ``keyword``.
    """
    tokens = _tokenize_segment(segment)
    if not tokens:
        tokens = [segment.strip()] if segment and segment.strip() else []
    if not tokens:
        return []

    # One scored CASE per token: 1 if any match column ILIKEs the token.
    case_parts = []
    params = []
    for tok in tokens:
        ors = sql.SQL(" OR ").join(
            sql.SQL("{col} ILIKE %s").format(col=sql.Identifier(c))
            for c in REPORTS_MATCH_COLS
        )
        case_parts.append(
            sql.SQL("(CASE WHEN ({ors}) THEN 1 ELSE 0 END)").format(ors=ors)
        )
        params.extend([f"%{tok}%"] * len(REPORTS_MATCH_COLS))
    score_expr = sql.SQL(" + ").join(case_parts)

    query = sql.SQL(
        "SELECT id, title, kw FROM ("
        " SELECT {id} AS id, {title} AS title, {kw} AS kw, {score} AS score"
        " FROM {table} WHERE {active} = 1"
        ") t WHERE t.score > 0 "
        "ORDER BY t.score DESC, t.id DESC LIMIT %s"
    ).format(
        id=sql.Identifier(REPORTS_ID_COL),
        title=sql.Identifier(REPORTS_TITLE_COL),
        kw=sql.Identifier("keyword"),
        score=score_expr,
        table=sql.Identifier(REPORTS_TABLE),
        active=sql.Identifier(REPORTS_ACTIVE_COL),
    )
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def reports_with_data(report_ids, conn) -> set:
    """Return the subset of ``report_ids`` that have ≥1 row in the summary or
    dynamic enrichment tables — i.e. reports that can actually yield facts."""
    ids = list(report_ids)
    if not ids:
        return set()
    found = set()
    with conn.cursor() as cur:
        for table, key in (
            (SUMMARY_TABLE, SUMMARY_JOIN_KEY),
            (DYNAMIC_TABLE, DYNAMIC_JOIN_KEY),
        ):
            query = sql.SQL(
                "SELECT DISTINCT {key} FROM {table} WHERE {key} = ANY(%s)"
            ).format(key=sql.Identifier(key), table=sql.Identifier(table))
            cur.execute(query, (ids,))
            found.update(row[0] for row in cur.fetchall())
    return found


def get_report_summary_facts(report_id, conn) -> list:
    """Mine a report's cmi_reportsummery row into facts.

    Pulls the named free-text section columns (SUMMARY_CONTENT_COLUMNS) plus the
    five insight header/description pairs, cleans the HTML, and extracts a
    numeric value where present.

    Returns dicts: ``{"section", "fact", "value", "is_numeric"}``.
    """
    # Build the ordered column list: content columns first, then insight pairs.
    select_specs = []  # (kind, label) parallel to selected columns
    select_cols = []
    for col, label in SUMMARY_CONTENT_COLUMNS:
        select_cols.append(sql.Identifier(col))
        select_specs.append(("content", label))
    for n in SUMMARY_INSIGHT_RANGE:
        select_cols.append(sql.Identifier(SUMMARY_INSIGHT_HEADER.format(n=n)))
        select_cols.append(sql.Identifier(SUMMARY_INSIGHT_DESC.format(n=n)))
        select_specs.append(("insight", f"insight {n}"))

    query = sql.SQL("SELECT {cols} FROM {table} WHERE {key} = %s").format(
        cols=sql.SQL(", ").join(select_cols),
        table=sql.Identifier(SUMMARY_TABLE),
        key=sql.Identifier(SUMMARY_JOIN_KEY),
    )
    with conn.cursor() as cur:
        cur.execute(query, (report_id,))
        row = cur.fetchone()
    if not row:
        return []

    facts = []
    col_i = 0
    for kind, label in select_specs:
        if kind == "content":
            content = clean_html(row[col_i] or "")
            section = label
            col_i += 1
        else:  # insight header/description pair
            header = clean_html(row[col_i] or "")
            content = clean_html(row[col_i + 1] or "")
            section = header or label
            col_i += 2
        if not content:
            continue
        value = extract_value(content)
        facts.append({
            "section": section,
            "fact": content,
            "value": value,
            "is_numeric": value is not None,
        })
    return facts


def get_report_dynamic_facts(report_id, conn) -> list:
    """Pull a report's cmi_report_dynamic field_{N}/disc_{N} pairs as facts.

    Wide format: one row per report with up to 20 field/description pairs. Cleans
    the HTML and extracts a numeric value where present.

    Returns dicts: ``{"field", "fact", "value", "is_numeric"}``.
    """
    select_cols = []
    for n in DYNAMIC_PAIR_RANGE:
        select_cols.append(sql.Identifier(f"{DYNAMIC_FIELD_PREFIX}{n}"))
        select_cols.append(sql.Identifier(f"{DYNAMIC_DESC_PREFIX}{n}"))
    query = sql.SQL("SELECT {cols} FROM {table} WHERE {key} = %s").format(
        cols=sql.SQL(", ").join(select_cols),
        table=sql.Identifier(DYNAMIC_TABLE),
        key=sql.Identifier(DYNAMIC_JOIN_KEY),
    )
    with conn.cursor() as cur:
        cur.execute(query, (report_id,))
        row = cur.fetchone()
    if not row:
        return []

    facts = []
    for i in range(0, len(row), 2):
        field = clean_html(row[i] or "")
        description = clean_html(row[i + 1] or "")
        if not description:
            continue
        value = extract_value(description)
        facts.append({
            "field": field or "detail",
            "fact": description,
            "value": value,
            "is_numeric": value is not None,
        })
    return facts


if __name__ == "__main__":
    sys.exit(0 if test_connection() else 1)
