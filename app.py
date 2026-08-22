"""
app.py

Streamlit UI for the Voice of Customer Copilot pipeline.

This is an ORCHESTRATOR, not a reimplementation. It calls the three
existing, already-tested pipeline scripts (classify_reviews.py,
generate_citations.py, generate_recommendation.py) as subprocesses,
in the same sequence and with the same CSV-in/CSV-out contract they
already use from the command line. Nothing about the pipeline logic,
guardrails, or scoring changes here -- this file only adds a way to
trigger and watch that existing pipeline run, and to display its
existing output files.

Run from inside the project folder so relative paths resolve correctly:

    cd ~/Desktop/voc-copilot
    pip3 install streamlit --break-system-packages
    streamlit run app.py

Configure an LLM backend in the sidebar -- Gemini, OpenAI, a local Ollama
model (no API key needed), or any other OpenAI-compatible endpoint. All
keys entered are used only for this session and never written to disk.
"""

import os
import csv
import json
import datetime
import subprocess
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_FILE = os.path.join(PROJECT_DIR, "urban_company_reviews.csv")
# CLASSIFIED_FILE, THEME_REPORT_FILE, RECOMMENDATION_FILE, and which scripts
# to run are set further down based on the selected mode (Urban Company v1
# vs any-company v2), so that a validated run and an unvalidated run never
# read or write each other's files.

REQUIRED_REVIEW_COLUMNS = {"review_id", "review_text", "rating", "date", "thumbs_up_count"}

st.set_page_config(page_title="Voice of Customer Copilot", layout="wide")

# Colors that depend on whether the active theme (light, dark, or "auto"
# resolved from the browser) is dark or light -- used both for the CSS
# below and for the matplotlib donut charts, so cards/charts never render
# with hardcoded colors that blend into the opposite theme.
_theme_type = st.context.theme.type or "dark"
if _theme_type == "dark":
    CARD_BG = "#262730"       # Streamlit's own dark-theme secondary background
    CHART_TEXT_COLOR = "#e6e6ea"
else:
    CARD_BG = "#f0f2f6"       # Streamlit's own light-theme secondary background
    CHART_TEXT_COLOR = "#31333f"
VOLUME_BAR_COLOR = "#3b82f6"  # neutral blue -- the red default read as a warning

# The Raw Classifications data_editor's cell dropdown (theme / secondary
# theme / churn risk) renders as a floating overlay whose background color
# defaults to nearly the same dark shade as the page behind it, making the
# dropdown hard to visually separate from the app. These selectors target
# the dropdown's own overlay wrapper (.gdg-d19meir1.gdg-style) and its
# option list (react-select internals, matched by stable ARIA roles /
# emotion class-name suffixes rather than hashed class names) to give it a
# distinct background, border, and hover state.
st.markdown(
    """
    <style>
    .gdg-d19meir1.gdg-style {
        background-color: #2b2d3a !important;
        border: 1px solid rgba(250, 250, 250, 0.35) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6) !important;
    }
    [class*="-menu"], [role="listbox"] {
        background-color: #363848 !important;
        border: 1px solid rgba(250, 250, 250, 0.25) !important;
        border-radius: 6px !important;
    }
    [class*="-option"], [role="option"] {
        background-color: transparent !important;
    }
    [class*="-option"]:hover, [role="option"]:hover {
        background-color: rgba(255, 75, 75, 0.2) !important;
    }

    /* Trim the framework's default top/bottom page padding (96px / 160px)
       -- most of it is dead space once the fixed top toolbar (60px) is
       cleared, and it pushes the whole dashboard down before you've seen
       anything. */
    [data-testid="stMainBlockContainer"] {
        padding-top: 3.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Tighten the gap between side-by-side columns (the metric row and
       the chart row) from the 16px default. */
    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem !important;
    }

    /* Shrink the 5 metric cards (Total Reviews, Themes Identified, ...) --
       less padding around the number and a slightly smaller number, so
       the whole row takes noticeably less of the first screenful. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMetric"]) {
        padding: 0.5rem 0.9rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.82rem !important;
    }

    /* Shrink the three chart cards (Priority Score Distribution / Top
       Themes by Volume / Churn Risk Breakdown) the same way, and tighten
       the gap between their stacked internal rows -- Top Themes by Volume
       alone stacks 10 elements, and the default 16px gap between each one
       adds up fast. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] [data-testid="stImage"]),
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] [data-testid="stProgress"]) {
        padding: 0.5rem 0.9rem !important;
        gap: 0.35rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Card backgrounds -- a shade lighter (dark theme) or darker (light theme)
# than the page itself, so the metric cards and chart cards read as
# distinct surfaces instead of blending into the page background. Kept as
# its own small f-string block (rather than folding into the static CSS
# above) since it's the only part that depends on CARD_BG/VOLUME_BAR_COLOR.
st.markdown(
    f"""
    <style>
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMetric"]),
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] [data-testid="stImage"]),
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] [data-testid="stProgress"]) {{
        background-color: {CARD_BG} !important;
    }}
    div[data-testid="stProgressBarTrack"] > div {{
        background-color: {VOLUME_BAR_COLOR} !important;
    }}

    /* Secondary buttons (e.g. "Edit") default to a near-transparent
       background that all but disappears against the page -- give them the
       same solid card surface as everything else, so they read as an actual
       clickable button/CTA instead of flat text with a faint outline. */
    [data-testid="stBaseButton-secondary"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid rgba(250, 250, 250, 0.35) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_stage1_health(classified_file):
    """
    Sanity-checks Stage 1 output before letting Stage 2/3 run on it. A bad
    or expired API key doesn't crash classify_reviews.py -- the missing-ID
    guardrail catches every failed review and flags it
    MISSING_FROM_LLM_RESPONSE / INVALID_THEME_NEEDS_REVIEW rather than
    silently dropping it. That's correct guardrail behavior, but if it
    happens to nearly every review, the real problem is almost always an
    invalid/expired API key or a rate-limit issue -- not that the reviews
    were actually analyzed and happened to come out wrong. Catching that
    here stops Stage 2/3 from quietly running on empty/garbage data.
    """
    rows = load_csv(classified_file)
    if not rows:
        return False, "No reviews were classified at all."
    bad = sum(
        1 for r in rows
        if "MISSING_FROM_LLM_RESPONSE" in (r.get("theme_1") or "")
        or "INVALID_THEME_NEEDS_REVIEW" in (r.get("theme_1") or "")
    )
    frac_bad = bad / len(rows)
    if frac_bad >= 0.3:
        return False, (
            f"{bad} of {len(rows)} reviews ({frac_bad:.0%}) failed to classify. "
            f"This almost always means the Gemini API key was rejected or "
            f"rate-limited during this run -- not that the reviews were read "
            f"and analyzed incorrectly."
        )
    return True, None


def run_script_live(script_name, log_placeholder):
    """
    Runs a pipeline script as a subprocess from the project directory,
    streaming its stdout into the Streamlit UI line by line so the
    user can see real progress (batch numbers, rate-limit waits,
    guardrail triggers) rather than a silent spinner.
    """
    script_path = os.path.join(PROJECT_DIR, script_name)
    if not os.path.exists(script_path):
        st.error(f"Could not find {script_name} in {PROJECT_DIR}. "
                  f"Is the UI running from inside ~/Desktop/voc-copilot?")
        return False

    run_env = dict(os.environ)
    run_env["PYTHONUNBUFFERED"] = "1"  # BUGFIX: without this, Python buffers
                                         # print() output when stdout isn't a
                                         # real terminal (as it isn't here,
                                         # piped into Streamlit) -- progress
                                         # messages would sit in a buffer and
                                         # never appear until the process
                                         # finished, making a working run look
                                         # stuck.

    process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=run_env,
    )

    log_lines = []
    for line in process.stdout:
        log_lines.append(line.rstrip())
        # Keep the log box readable -- show the tail, not the whole history
        log_placeholder.code("\n".join(log_lines[-25:]), language="text")

    process.wait()
    return process.returncode == 0


def guardrail_summary_from_log(log_text):
    """Pulls out guardrail-relevant lines so they're visible as a distinct
    summary rather than buried in the raw console log."""
    keywords = ["GUARDRAIL", "INVALID_THEME", "MISSING_FROM_LLM_RESPONSE",
                "LOW_CONFIDENCE", "WATCH LIST", "WATCH_LIST"]
    return [line for line in log_text.splitlines() if any(k in line for k in keywords)]


# ---------------------------------------------------------------------------
# Dashboard metrics -- every number here is computed directly from the
# pipeline's own CSV outputs. Deltas ("vs last run") are real too: computed
# by comparing against a snapshot saved after the previous run for the same
# mode/company, stored in dashboard_history.json. No fabricated numbers.
# ---------------------------------------------------------------------------

HISTORY_FILE = os.path.join(PROJECT_DIR, "dashboard_history.json")

# Both taxonomies, used to constrain the editable dropdowns in Raw
# Classifications so a manual correction can't accidentally introduce an
# out-of-taxonomy value (the same guardrail principle as validate_theme()
# in the classify scripts, just applied to human edits too).
V1_TAXONOMY = [
    "punctuality_noshow", "service_quality", "over_priced", "refund_cancellation",
    "booking_app_ux", "safety_trust", "customer_support", "generic_praise",
    "competitor's_praise", "new_service_ask", "loyalty program ask",
    "useless review", "other_not_listed",
]
V2_TAXONOMY = [
    "pricing_value", "service_quality", "timeliness_reliability", "refund_cancellation",
    "app_booking_ux", "safety_trust", "customer_support", "generic_praise",
    "competitor_comparison", "feature_or_service_request", "no_signal", "other_not_listed",
]


def info_icon(tooltip_text):
    """Small hoverable info icon (native browser tooltip via title attr) for
    section headers that aren't st.metric (which has its own help= param)."""
    return f'<span title="{tooltip_text}" style="cursor:help; opacity:0.55; font-size:0.85em;"> ℹ️</span>'


CORRECTIONS_LOG_FILE = os.path.join(PROJECT_DIR, "corrections_log.csv")


def log_corrections(old_rows, new_rows, run_key):
    """
    Diffs old vs new classification rows (by review_id) and appends any
    actual field changes to a persistent, append-only log. This does NOT
    make the AI model learn or retrain automatically -- it's an honest
    record of human corrections that can be used later to expand the eval
    ground truth or inform the next prompt-iteration round. Returns the
    number of individual field changes logged.
    """
    old_by_id = {r["review_id"]: r for r in old_rows}
    changed = []
    for new_row in new_rows:
        rid = new_row.get("review_id", "")
        old_row = old_by_id.get(rid, {})
        for field in ["theme_1", "theme_2_optional", "is_churn_risk"]:
            old_val = (old_row.get(field) or "").strip()
            new_val = (new_row.get(field) or "").strip()
            if old_val != new_val:
                changed.append({
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "run_key": run_key,
                    "review_id": rid,
                    "field": field,
                    "old_value": old_val,
                    "new_value": new_val,
                })
    if changed:
        file_exists = os.path.exists(CORRECTIONS_LOG_FILE)
        with open(CORRECTIONS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "run_key", "review_id", "field", "old_value", "new_value"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(changed)
    return len(changed)


def prettify_theme(name):
    """theme_name_like_this -> Theme Name Like This, keeping apostrophes
    (e.g. competitor's) from being mangled by naive title-casing."""
    words = (name or "").replace("_", " ").split(" ")
    out = []
    for w in words:
        out.append((w[0].upper() + w[1:]) if w else w)
    return " ".join(out)


def score_bucket(value):
    if value >= 0.6:
        return "High"
    if value >= 0.3:
        return "Medium"
    return "Low"


# Categories that are never "actionable" in the sense of "a problem worth
# fixing" -- they're either praise (nothing to fix) or noise (no real
# signal). Covers naming variants across both the Urban Company taxonomy
# (v1) and the universal any-company taxonomy (v2).
NON_ACTIONABLE_THEMES = {"generic_praise", "no_signal", "useless review", "other_not_listed"}


def compute_dashboard_metrics(classified, theme_report, recommendation):
    total_reviews = len(classified)

    theme_score = {}
    ranked_scores = []
    for r in recommendation:
        try:
            score = float(r["priority_score"])
        except (KeyError, ValueError, TypeError):
            continue
        theme_score[r["theme"]] = score
        # Only RANKED (statistically robust) themes count toward the
        # headline "Top Priority Score" -- a Watch List theme's score can
        # look dramatic (e.g. 100% churn on 3 reviews) precisely because
        # it's NOT reliable enough to trust, so it shouldn't be the
        # number a PM sees as their top priority.
        if r.get("section") == "RANKED":
            ranked_scores.append(score)

    # Churn rate per theme, computed directly from the raw classifications
    # (not just the recommendation file) so every theme is covered, including
    # ones excluded from ranking.
    theme_totals, theme_churn = {}, {}
    for r in classified:
        t = (r.get("theme_1") or "").strip()
        if not t:
            continue
        theme_totals[t] = theme_totals.get(t, 0) + 1
        if (r.get("is_churn_risk") or "").strip().upper() == "Y":
            theme_churn[t] = theme_churn.get(t, 0) + 1
    theme_churn_rate = {t: theme_churn.get(t, 0) / n for t, n in theme_totals.items()}

    priority_buckets = {"High": 0, "Medium": 0, "Low": 0}
    churn_buckets = {"High": 0, "Medium": 0, "Low": 0}
    for r in classified:
        t = (r.get("theme_1") or "").strip()
        priority_buckets[score_bucket(theme_score.get(t, 0.0))] += 1
        churn_buckets[score_bucket(theme_churn_rate.get(t, 0.0))] += 1

    overall_churn_count = sum(
        1 for r in classified if (r.get("is_churn_risk") or "").strip().upper() == "Y"
    )
    overall_churn_rate = (overall_churn_count / total_reviews * 100) if total_reviews else 0.0

    theme_volume = sorted(
        ((r["theme"], int(r.get("review_count", 0))) for r in theme_report),
        key=lambda x: -x[1],
    )

    # "Actionable" = statistically robust (RANKED, not watch list or
    # excluded) AND represents an actual problem worth fixing -- praise and
    # no-signal categories are excluded, since "everyone loves us" or "no
    # real feedback" isn't something to act on. The full list (not just the
    # count) is kept so the UI can show what these insights actually are,
    # not just a bare number.
    actionable_themes = sorted(
        (
            {
                "theme": r["theme"],
                "review_count": int(r.get("review_count", 0)),
                "churn_rate": float(r.get("churn_rate", 0) or 0),
                "priority_score": float(r.get("priority_score", 0) or 0),
                "summary": r.get("summary", ""),
            }
            for r in recommendation
            if r.get("section") == "RANKED" and r.get("theme") not in NON_ACTIONABLE_THEMES
        ),
        key=lambda x: -x["priority_score"],
    )

    return {
        "total_reviews": total_reviews,
        "themes_identified": len(theme_report),
        "theme_set": sorted(theme_totals.keys()),
        "top_priority_score": round(max(ranked_scores), 4) if ranked_scores else 0.0,
        "overall_churn_rate": round(overall_churn_rate, 1),
        "churn_level": score_bucket(overall_churn_rate / 100),
        "actionable_insights": len(actionable_themes),
        "actionable_themes": actionable_themes,
        "priority_buckets": priority_buckets,
        "churn_buckets": churn_buckets,
        "theme_volume": theme_volume,
    }


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f)
    except OSError:
        pass


def compute_deltas(curr, prev):
    """Real deltas vs the previous run for this mode/company -- returns None
    if there's no previous run to compare against (first run)."""
    if not prev:
        return None

    def pct_change(c, p):
        if not p:
            return None
        return round((c - p) / p * 100, 1)

    return {
        "total_reviews_pct": pct_change(curr["total_reviews"], prev.get("total_reviews", 0)),
        "themes_new_count": len(set(curr["theme_set"]) - set(prev.get("theme_set", []))),
        "top_priority_score_pct": pct_change(curr["top_priority_score"], prev.get("top_priority_score", 0)),
        "churn_rate_pp": round(curr["overall_churn_rate"] - prev.get("overall_churn_rate", 0), 1),
        "actionable_insights_delta": curr["actionable_insights"] - prev.get("actionable_insights", 0),
    }


def render_dashboard(metrics, deltas):
    d = deltas or {}

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1.container(border=True):
        st.metric(
            "💬 Total Reviews", metrics["total_reviews"],
            delta=f"{d['total_reviews_pct']:+.0f}%" if d.get("total_reviews_pct") is not None else None,
            help="Count of reviews successfully classified in this run.",
        )
    with col2.container(border=True):
        st.metric(
            "🧩 Themes Identified", metrics["themes_identified"],
            delta=f"+{d['themes_new_count']} new" if d.get("themes_new_count") else None,
            help="Number of distinct categories with at least one review classified into them, out of the fixed taxonomy.",
        )
    with col3.container(border=True):
        st.metric(
            "🎯 Top Priority Score", f"{metrics['top_priority_score']:.3f}",
            delta=f"{d['top_priority_score_pct']:+.0f}%" if d.get("top_priority_score_pct") is not None else None,
            help="Highest priority_score among RANKED (statistically robust) themes only -- Watch List themes are excluded even if their raw score is higher, since their sample size is too small to trust. Formula: priority_score = (theme's review count ÷ total reviews) × 0.5 + churn_rate × 0.5.",
        )
    with col4.container(border=True):
        st.metric(
            "❤️ Avg Churn Risk", metrics["churn_level"],
            delta=f"{d['churn_rate_pp']:+.1f} pp" if d.get("churn_rate_pp") is not None else None,
            delta_color="inverse",  # rising churn is bad -> red on increase
            help="Share of ALL reviews flagged is_churn_risk=Y by Stage 1, bucketed: Low <30%, Medium 30-60%, High ≥60%.",
        )
    with col5.container(border=True):
        st.metric(
            "✅ Actionable Insights", metrics["actionable_insights"],
            delta=f"{d['actionable_insights_delta']:+d}" if d.get("actionable_insights_delta") is not None else None,
            help="Count of RANKED themes that represent an actual problem -- excludes generic praise, no-signal reviews, and uncategorized 'other'. See the list below to view them by name.",
        )
    if deltas is None:
        st.caption("First run for this company/mode -- deltas will show from the next run onward.")
    else:
        st.caption("Deltas shown are vs. this company/mode's previous run.")

    actionable = metrics["actionable_themes"]
    with st.expander(f"📋 View the {len(actionable)} actionable insight{'s' if len(actionable) != 1 else ''}", expanded=False):
        if not actionable:
            st.caption(
                "No theme currently meets the bar for an actionable insight -- either "
                "nothing has enough reviews yet to be statistically reliable, or the "
                "robust themes found so far are praise/no-signal rather than a "
                "specific problem to fix."
            )
        for t in actionable:
            st.markdown(f"**{prettify_theme(t['theme'])}**")
            st.caption(
                f"{t['review_count']} reviews · {t['churn_rate']*100:.0f}% churn risk · "
                f"priority score {t['priority_score']:.3f}"
            )
            if t["summary"]:
                st.write(t["summary"])
            st.divider()

    dcol1, dcol2, dcol3 = st.columns(3)

    bucket_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}

    def render_donut(container, title, buckets, total, tooltip):
        with container:
            with st.container(border=True):
                st.markdown(f"**{title}**{info_icon(tooltip)}", unsafe_allow_html=True)
                if total == 0:
                    st.caption("No data.")
                    return
                try:
                    import matplotlib.pyplot as plt
                    labels = ["High", "Medium", "Low"]
                    sizes = [buckets[l] for l in labels]
                    fig, ax = plt.subplots(figsize=(2.0, 2.0))
                    fig.patch.set_facecolor(CARD_BG)
                    ax.set_facecolor(CARD_BG)
                    ax.pie(
                        sizes,
                        colors=[bucket_colors[l] for l in labels],
                        startangle=90,
                        wedgeprops=dict(width=0.42, edgecolor=CARD_BG),
                    )
                    ax.text(0, 0, f"{total}\nTotal reviews", ha="center", va="center",
                            fontsize=9, color=CHART_TEXT_COLOR)
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                except ImportError:
                    st.caption("(install matplotlib for a chart: pip3 install matplotlib --break-system-packages)")
                for label in ["High", "Medium", "Low"]:
                    count = buckets[label]
                    pct = (count / total * 100) if total else 0
                    st.markdown(
                        f"<span style='color:{bucket_colors[label]}'>●</span> {label} &nbsp; {count} ({pct:.0f}%)",
                        unsafe_allow_html=True,
                    )

    render_donut(
        dcol1, "Priority Score Distribution", metrics["priority_buckets"], metrics["total_reviews"],
        tooltip="Every review inherits its own theme's priority_score, then gets bucketed: High &ge;0.6, Medium 0.3-0.6, Low &lt;0.3. Shows how much of your feedback sits in high- vs low-priority territory.",
    )

    with dcol2:
        with st.container(border=True):
            st.markdown(
                "**Top Themes by Volume**"
                + info_icon("Simple count of reviews per theme, ranked by raw frequency -- independent of churn rate or priority score. A high-volume theme isn't necessarily high-priority if its churn rate is low (e.g. generic praise)."),
                unsafe_allow_html=True,
            )
            top5 = metrics["theme_volume"][:5]
            total = metrics["total_reviews"]
            for theme, count in top5:
                pct = (count / total * 100) if total else 0
                st.caption(f"{prettify_theme(theme)} -- {count} ({pct:.0f}%)")
                st.progress(min(count / total, 1.0) if total else 0.0)

    render_donut(
        dcol3, "Churn Risk Breakdown", metrics["churn_buckets"], metrics["total_reviews"],
        tooltip="Every review inherits its own theme's churn rate (% of that theme's reviews flagged is_churn_risk=Y), then gets bucketed the same way as Priority Score. Not the same as an individual review's own Y/N flag.",
    )

    st.divider()


# ---------------------------------------------------------------------------
# Sidebar -- environment / input controls
# ---------------------------------------------------------------------------

st.sidebar.subheader("LLM Backend")
backend_choice = st.sidebar.selectbox(
    "Which LLM to use",
    ["Gemini", "OpenAI", "Ollama (local, no API key)", "Custom (OpenAI-compatible)"],
    key="llm_backend_choice",
    help="Switch anytime -- your pipeline scripts read whichever backend is "
         "configured here at the moment you click Analyze.",
)

# Reset provider-specific env vars on every render so switching backends
# never leaves a stale key/URL/model from a previous choice lying around.
os.environ.pop("LLM_API_KEY", None)
os.environ.pop("LLM_BASE_URL", None)
os.environ.pop("LLM_MODEL", None)

if backend_choice == "Gemini":
    os.environ["LLM_PROVIDER"] = "gemini"
    gemini_key = st.sidebar.text_input(
        "Gemini API key", type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        key="gemini_key_field",
        help="Get one at https://aistudio.google.com/apikey",
    )
    gemini_model = st.sidebar.text_input(
        "Model", value="gemini-3.5-flash-lite", key="gemini_model_field",
    )
    if gemini_key:
        os.environ["LLM_API_KEY"] = gemini_key
        os.environ["GEMINI_API_KEY"] = gemini_key  # kept for backward compatibility
    os.environ["LLM_MODEL"] = gemini_model or "gemini-3.5-flash-lite"

elif backend_choice == "OpenAI":
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    openai_key = st.sidebar.text_input(
        "OpenAI API key", type="password", key="openai_key_field",
    )
    openai_model = st.sidebar.text_input(
        "Model", value="gpt-4o-mini", key="openai_model_field",
    )
    if openai_key:
        os.environ["LLM_API_KEY"] = openai_key
    os.environ["LLM_MODEL"] = openai_model or "gpt-4o-mini"

elif backend_choice == "Ollama (local, no API key)":
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    ollama_url = st.sidebar.text_input(
        "Ollama URL", value="http://localhost:11434/v1", key="ollama_url_field",
    )
    ollama_model = st.sidebar.text_input(
        "Model", value="llama3.1", key="ollama_model_field",
        help="Must already be pulled locally -- run: ollama pull llama3.1",
    )
    os.environ["LLM_BASE_URL"] = ollama_url or "http://localhost:11434/v1"
    os.environ["LLM_MODEL"] = ollama_model or "llama3.1"
    st.sidebar.caption("No API key needed. Make sure Ollama is running locally (`ollama serve`).")

else:  # Custom (OpenAI-compatible)
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    custom_url = st.sidebar.text_input(
        "Base URL", placeholder="https://api.example.com/v1", key="custom_url_field",
        help="Groq, OpenRouter, LM Studio, vLLM, etc. -- any endpoint that speaks the OpenAI chat-completions API.",
    )
    custom_key = st.sidebar.text_input(
        "API key (if required)", type="password", key="custom_key_field",
    )
    custom_model = st.sidebar.text_input("Model name", key="custom_model_field")
    if custom_url:
        os.environ["LLM_BASE_URL"] = custom_url
    if custom_key:
        os.environ["LLM_API_KEY"] = custom_key
    if custom_model:
        os.environ["LLM_MODEL"] = custom_model

try:
    from llm_provider import validate_config
    validate_config()
    llm_ready = True
    st.sidebar.success(f"{backend_choice} configured.")
except Exception as e:
    llm_ready = False
    st.sidebar.warning(str(e))

st.sidebar.subheader("Mode")
mode = st.sidebar.radio(
    "Which pipeline to run",
    options=["Urban Company", "Any company"],
    help="Urban Company mode uses the original 13-category taxonomy, "
         "validated against 92 hand-labeled reviews (68.5% theme accuracy, "
         "68.1% churn F1). Any-company mode uses a generalized 12-category "
         "taxonomy that should apply to most consumer product/service "
         "companies, but is UNVALIDATED beyond Urban Company -- no ground "
         "truth exists yet for other domains, so treat its output as a "
         "reasonable first pass, not a measured result.",
)
is_custom_mode = mode.startswith("Any company")

company_name = "Urban Company"
if is_custom_mode:
    company_name = st.sidebar.text_input(
        "Company name",
        value="",
        placeholder="e.g. Zomato, Notion, Swiggy",
        help="Used in the classification prompt so the model knows whose "
             "reviews it's reading.",
    )
    st.sidebar.warning(
        "Unvalidated mode: results are a reasonable first pass, not a "
        "measured eval result. Treat priority scores as directional, not "
        "precise, until a hand-labeled sample from this domain confirms "
        "accuracy.",
        icon="⚠️",
    )
    os.environ["VOC_COMPANY_NAME"] = company_name or "this company"

run_key = f"custom:{company_name.strip().lower()}" if is_custom_mode else "urban_company"

# File/script routing based on mode -- keeps the validated Urban Company v1
# run completely separate from any-company v2 runs, so neither ever
# overwrites the other.
if is_custom_mode:
    CLASSIFY_SCRIPT = "classify_reviews_v2.py"
    CITATIONS_SCRIPT = "generate_citations_v2.py"
    RECOMMEND_SCRIPT = "generate_recommendation_v2.py"
    CLASSIFIED_FILE = os.path.join(PROJECT_DIR, "classified_reviews_v2.csv")
    THEME_REPORT_FILE = os.path.join(PROJECT_DIR, "theme_report_v2.csv")
    RECOMMENDATION_FILE = os.path.join(PROJECT_DIR, "prioritized_recommendation_v2.csv")
else:
    CLASSIFY_SCRIPT = "classify_reviews.py"
    CITATIONS_SCRIPT = "generate_citations.py"
    RECOMMEND_SCRIPT = "generate_recommendation.py"
    CLASSIFIED_FILE = os.path.join(PROJECT_DIR, "classified_reviews.csv")
    THEME_REPORT_FILE = os.path.join(PROJECT_DIR, "theme_report.csv")
    RECOMMENDATION_FILE = os.path.join(PROJECT_DIR, "prioritized_recommendation.csv")

CUSTOM_REVIEWS_FILE = os.path.join(PROJECT_DIR, "custom_reviews.csv")
CUSTOM_OUTPUT_OWNER_FILE = os.path.join(PROJECT_DIR, "custom_output_owner.txt")  # tracks
    # which company's data currently sits in the v2 output files, since all
    # any-company runs share the same fixed filenames -- prevents a previous
    # company's leftover results from being shown under a different company's name

st.sidebar.subheader("Input data")

if is_custom_mode:
    st.sidebar.caption(
        f"Upload {company_name or 'this company'}'s reviews CSV to enable analysis."
    )

    sample_csv = (
        "review_id,review_text,rating,date,thumbs_up_count\n"
        "z001,\"Order was delayed by 40 minutes and support didn't respond\",1,2026-07-20,3\n"
        "z002,\"Great food quality but delivery fee is too high\",3,2026-07-21,0\n"
        "z003,\"App keeps crashing when I try to apply a coupon\",2,2026-07-22,5\n"
    )
    st.sidebar.download_button(
        "Download sample CSV format",
        data=sample_csv,
        file_name="reviews_template.csv",
        mime="text/csv",
        help="review_id: any unique ID per row. review_text: the review content. "
             "rating: numeric scale (e.g. 1-5). date: any consistent format. "
             "thumbs_up_count: numeric, use 0 if you don't have this data. "
             "Column names must match exactly, any order.",
    )

    uploaded = st.sidebar.file_uploader(
        "Upload reviews CSV (required for any-company mode)",
        type="csv",
        key="custom_reviews_uploader",
    )
    if uploaded is not None:
        content = uploaded.getvalue().decode("utf-8")
        header = content.splitlines()[0].split(",")
        if not REQUIRED_REVIEW_COLUMNS.issubset(set(h.strip() for h in header)):
            st.sidebar.error(
                f"Uploaded CSV is missing required columns. Needs: {', '.join(sorted(REQUIRED_REVIEW_COLUMNS))}"
            )
        else:
            with open(CUSTOM_REVIEWS_FILE, "w", encoding="utf-8") as f:
                f.write(content)
            st.sidebar.success(f"Saved uploaded reviews for {company_name or 'this company'}.")

    active_reviews_file = CUSTOM_REVIEWS_FILE
    existing_reviews = load_csv(CUSTOM_REVIEWS_FILE)
    if existing_reviews:
        st.sidebar.info(f"{len(existing_reviews)} reviews ready in custom_reviews.csv")
    else:
        st.sidebar.warning(
            "No reviews uploaded yet for this company. Upload a CSV above -- "
            "analysis stays disabled until you do, so no API calls run against "
            "the wrong (or stale) data."
        )
else:
    active_reviews_file = REVIEWS_FILE
    existing_reviews = load_csv(REVIEWS_FILE)
    if not existing_reviews:
        st.sidebar.warning(
            "No urban_company_reviews.csv found in the project folder. Run "
            "pull_urban_company_reviews.py first."
        )


# ---------------------------------------------------------------------------
# Main -- run pipeline
# ---------------------------------------------------------------------------

st.title("Voice of Customer Copilot")
if is_custom_mode and company_name:
    st.caption(f"Analyzing customer feedback for {company_name}.")

missing_company_name = is_custom_mode and not company_name
can_run = llm_ready and bool(existing_reviews) and not missing_company_name

run_clicked = st.button("Analyze reviews", type="primary", disabled=not can_run)

if not can_run and not run_clicked:
    if not llm_ready:
        st.info("Finish configuring your LLM backend in the sidebar to run the pipeline.")
    elif not existing_reviews and is_custom_mode:
        st.info("Upload a reviews CSV for this company in the sidebar to run the pipeline.")
    elif not existing_reviews:
        st.info("No reviews found. Run pull_urban_company_reviews.py first.")
    elif missing_company_name:
        st.info("Enter a company name in the sidebar to run any-company mode.")

if run_clicked:
    progress = st.progress(0, text="Reading & Categorizing Reviews...")
    with st.expander("Show technical details", expanded=False):
        log_box = st.empty()

    ok = run_script_live(CLASSIFY_SCRIPT, log_box)
    if not ok:
        progress.progress(33, text="Reading & Categorizing Reviews -- failed, see technical details above")
        st.stop()
    progress.progress(33, text="Reading & Categorizing Reviews -- complete")

    # Sanity check before spending more API calls on Stage 2/3: if most
    # reviews failed to classify (commonly an invalid/expired API key or a
    # rate-limit issue), stop here with a clear explanation instead of
    # quietly producing a report built on empty data.
    healthy, problem = check_stage1_health(CLASSIFIED_FILE)
    if not healthy:
        st.error(
            f"⚠️ Analysis stopped: {problem} Check your LLM backend settings "
            f"in the sidebar and click Analyze again."
        )
        st.stop()

    progress.progress(34, text="Finding Real Examples for Each Theme...")
    ok = run_script_live(CITATIONS_SCRIPT, log_box)
    if not ok:
        progress.progress(66, text="Finding Real Examples for Each Theme -- failed, see technical details above")
        st.stop()
    progress.progress(66, text="Finding Real Examples for Each Theme -- complete")

    progress.progress(67, text="Prioritizing What to Fix First...")
    ok = run_script_live(RECOMMEND_SCRIPT, log_box)
    if not ok:
        progress.progress(100, text="Prioritizing What to Fix First -- failed, see technical details above")
        st.stop()
    progress.progress(100, text="Analysis complete.")

    if is_custom_mode:
        with open(CUSTOM_OUTPUT_OWNER_FILE, "w", encoding="utf-8") as f:
            f.write(run_key)

    st.success("Analysis complete.")
    st.session_state["pipeline_ran"] = True
    st.session_state["pipeline_just_completed"] = True
    st.session_state["completed_run_key"] = run_key


# ---------------------------------------------------------------------------
# Pending corrections -- if a classification was edited in the Raw
# Classifications tab, the actual regeneration happens here at the top of
# the page (not inline in that tab), so it sits right next to the main
# Analyze action instead of being buried below the table. The "Analyze
# reviews" shortcut button inside the Raw Classifications tab doesn't run
# anything itself -- it sets auto_run_regen_{run_key} and reruns, so this
# same top-of-page block picks it up and runs, scrolled into view, exactly
# like a manual click on "Update Results" would.
# ---------------------------------------------------------------------------

if st.session_state.get(f"needs_regen_{run_key}"):
    auto_triggered = st.session_state.pop(f"auto_run_regen_{run_key}", False)
    regen_disabled = not llm_ready

    if auto_triggered:
        # Streamlit scrolls its own inner [data-testid="stMain"] section, not
        # the browser window -- window.scrollTo() here would be a no-op.
        components.html(
            """<script>
            var mainSection = window.parent.document.querySelector('[data-testid="stMain"]');
            if (mainSection) { mainSection.scrollTo({top: 0, behavior: 'instant'}); }
            </script>""",
            height=0,
        )

    st.info("You have unsaved corrections. Update your results to refresh Theme Details and the Prioritized Roadmap.")
    clicked = st.button(
        "🔁 Update Results",
        key=f"regen_top_{run_key}",
        disabled=regen_disabled,
        type="primary",
        help="Re-runs citation selection and re-ranks the roadmap from your corrected "
             "classifications -- does NOT re-classify from scratch, so your edits aren't "
             "overwritten." if not regen_disabled
             else "Finish configuring your LLM backend in the sidebar to enable this.",
    )

    if auto_triggered and regen_disabled:
        st.warning("Finish configuring your LLM backend in the sidebar, then click Update Results above.")

    if (clicked or auto_triggered) and not regen_disabled:
        regen_progress = st.progress(0, text="Updating results from your corrections...")
        with st.expander("Show technical details", expanded=False):
            regen_log = st.empty()
        ok = run_script_live(CITATIONS_SCRIPT, regen_log)
        if not ok:
            regen_progress.progress(50, text="Citation regeneration failed -- see technical details above.")
            st.stop()
        regen_progress.progress(60, text="Citations updated -- recalculating roadmap...")
        ok2 = run_script_live(RECOMMEND_SCRIPT, regen_log)
        if not ok2:
            regen_progress.progress(100, text="Roadmap regeneration failed -- see technical details above.")
            st.stop()
        regen_progress.progress(100, text="Done.")
        st.session_state[f"needs_regen_{run_key}"] = False
        st.success("Results updated from your corrections.")
        st.rerun()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if is_custom_mode:
    output_owner = None
    if os.path.exists(CUSTOM_OUTPUT_OWNER_FILE):
        with open(CUSTOM_OUTPUT_OWNER_FILE, encoding="utf-8") as f:
            output_owner = f.read().strip()
    results_belong_to_current_selection = (output_owner == run_key)
else:
    results_belong_to_current_selection = True  # Urban Company's file is dedicated, always its own

if results_belong_to_current_selection:
    classified = load_csv(CLASSIFIED_FILE)
    theme_report = load_csv(THEME_REPORT_FILE)
    recommendation = load_csv(RECOMMENDATION_FILE)
else:
    classified, theme_report, recommendation = [], [], []

if classified or theme_report or recommendation:
    dashboard_metrics = compute_dashboard_metrics(classified, theme_report, recommendation)

    history = load_history()
    just_completed = (
        st.session_state.get("pipeline_just_completed")
        and st.session_state.get("completed_run_key") == run_key
    )
    if just_completed:
        prev_snapshot = history.get(run_key)
        deltas = compute_deltas(dashboard_metrics, prev_snapshot)
        history[run_key] = dashboard_metrics
        save_history(history)
        st.session_state["pipeline_just_completed"] = False
        st.session_state[f"last_deltas:{run_key}"] = deltas
    else:
        deltas = st.session_state.get(f"last_deltas:{run_key}")

    render_dashboard(dashboard_metrics, deltas)

    tab_priority, tab_themes, tab_raw, tab_guardrails = st.tabs(
        ["Prioritized Roadmap", "Theme Details & Citations", "Raw Classifications", "Guardrail Activity"]
    )

    with tab_priority:
        st.caption(
            "The single ranked list of what to fix first -- combining how common "
            "each problem is with how likely it is to make customers leave. "
            "This is your action list."
        )
        st.subheader("Prioritized roadmap", anchor=False)
        if recommendation:
            ranked = [r for r in recommendation if r.get("section") == "RANKED"]

            if ranked:
                st.dataframe(
                    [{
                        "Rank": r["rank"],
                        "Theme": prettify_theme(r["theme"]),
                        "Reviews": r["review_count"],
                        "Frequency": float(r["frequency_share"]) * 100,
                        "Churn rate": float(r["churn_rate"]) * 100,
                        "Priority score": float(r["priority_score"]),
                    } for r in ranked],
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "Frequency": st.column_config.NumberColumn(
                            format="%.1f%%",
                            help="frequency_share = this theme's review count \u00f7 total reviews. How common the problem is.",
                        ),
                        "Churn rate": st.column_config.NumberColumn(
                            format="%.1f%%",
                            help="churn_rate = reviews in this theme flagged is_churn_risk=Y \u00f7 total reviews in this theme. How likely this problem is to drive customers away.",
                        ),
                        "Priority score": st.column_config.NumberColumn(
                            format="%.3f",
                            help="priority_score = frequency_share \u00d7 0.5 + churn_rate \u00d7 0.5. Fully deterministic -- no AI judgment involved, just this formula.",
                        ),
                    },
                )
            else:
                st.info(
                    "No theme has enough reviews yet to be ranked with confidence. "
                    "Check the Guardrail Activity tab to see what's being held back and why."
                )
        else:
            st.info("Run the pipeline to see the prioritized roadmap.")

    with tab_themes:
        st.caption(
            "What each theme actually looks like in customers' own words, plus the "
            "AI's summary of the pattern."
        )
        st.subheader(
            "Theme summaries and citations",
            anchor=False,
            help="The model only ever selects WHICH review IDs are representative -- "
                 "the quote text shown here is always fetched by code from the source "
                 "CSV, never generated by the model. This is the anti-hallucination guardrail.",
        )
        if theme_report:
            for row in theme_report:
                with st.expander(f"{prettify_theme(row['theme'])}  ({row.get('review_count', '?')} reviews, {row.get('confidence', '')})"):
                    st.write(row.get("summary", ""))
                    quotes = row.get("cited_quotes", "")
                    if quotes:
                        st.markdown("**Cited quotes (verbatim, code-fetched):**")
                        st.write(quotes)
        else:
            st.info("Run the pipeline to see theme summaries and citations.")

    with tab_raw:
        header_left, header_actions = st.columns([3.4, 2.6])
        with header_left:
            st.subheader("Raw per-review classifications", anchor=False)
            st.caption(
                "Every review the AI read, and the theme + churn-risk flag it assigned. "
                "This is where you catch and fix mistakes."
            )
        with header_actions:
            st.write("")  # vertical alignment nudge to sit level with the subheader
            # Both buttons live in one tight row, side by side, and are always
            # rendered together (Analyze reviews just goes disabled when there's
            # nothing to regenerate) so the pair reads as one action group instead
            # of a lone button with an empty gap next to it.
            btn_col1, btn_col2 = st.columns([1, 1.3], gap="small")
            # This placeholder is filled in further down -- as an Edit button while
            # read-only, or as a Save button (wired to edited_df) once editing starts --
            # so the same slot toggles between the two rather than the old
            # Edit -> Cancel pair that discarded changes instead of saving them.
            action_slot = btn_col1.empty()
            # Filled in below -- a shortcut to trigger the same "Update Results" run
            # from here instead of making the user scroll up. Always visible next to
            # Edit/Save; disabled until there are saved corrections to regenerate.
            analyze_slot = btn_col2.empty()

        if classified:
            review_lookup = {r.get("review_id", ""): r for r in existing_reviews}
            rows = []
            for c in classified:
                rid = c.get("review_id", "")
                src = review_lookup.get(rid, {})
                rows.append({
                    "review_id": rid,
                    "review_text": src.get("review_text", "(original review text not found)"),
                    "theme_1": c.get("theme_1", "") or "",
                    "theme_2_optional": c.get("theme_2_optional", "") or "",
                    "is_churn_risk": c.get("is_churn_risk", "") or "",
                })
            df = pd.DataFrame(rows)
            display_columns = ["review_text", "theme_1", "theme_2_optional", "is_churn_risk"]

            edit_key = f"raw_edit_mode_{run_key}"
            if edit_key not in st.session_state:
                st.session_state[edit_key] = False

            if not st.session_state[edit_key]:
                st.dataframe(
                    df,
                    column_order=display_columns,
                    column_config={
                        "review_text": st.column_config.TextColumn("💬 Review", width="large"),
                        "theme_1": st.column_config.TextColumn("🏷️ Theme", width="medium"),
                        "theme_2_optional": st.column_config.TextColumn("🏷️ Secondary Theme", width="medium"),
                        "is_churn_risk": st.column_config.TextColumn("⚠️ Churn Risk", width="small"),
                    },
                    hide_index=True,
                    width='stretch',
                )
                with action_slot:
                    if st.button("✏️ Edit", key=f"start_edit_{run_key}", width='stretch'):
                        st.session_state[edit_key] = True
                        st.rerun()

                has_pending_corrections = bool(st.session_state.get(f"needs_regen_{run_key}"))
                with analyze_slot:
                    if st.button(
                        "Analyze reviews", key=f"raw_tab_analyze_{run_key}",
                        type="primary", width='stretch',
                        disabled=not has_pending_corrections,
                        help=(
                            "Jumps to the top of the page and regenerates Theme Details "
                            "and the Prioritized Roadmap from the corrections you just saved."
                            if has_pending_corrections
                            else "Save a correction below first -- there's nothing new to analyze yet."
                        ),
                    ):
                        st.session_state[f"auto_run_regen_{run_key}"] = True
                        st.rerun()

            else:
                taxonomy_options = V2_TAXONOMY if is_custom_mode else V1_TAXONOMY
                theme1_options = sorted(set(taxonomy_options) | set(df["theme_1"].unique()))
                theme2_options = [""] + sorted((set(taxonomy_options) | set(df["theme_2_optional"].unique())) - {""})
                churn_options = sorted(set(["Y", "N"]) | set(df["is_churn_risk"].unique()))

                st.info(
                    "**Editing on.** Each column is a separate field -- "
                    "🏷️ **Theme** = the main category, "
                    "🏷️ **Secondary Theme** = an optional second category, "
                    "⚠️ **Churn Risk** = Y or N only. Change any cell, then Save."
                )
                edited_df = st.data_editor(
                    df,
                    column_order=display_columns,
                    column_config={
                        "review_text": st.column_config.TextColumn(
                            "💬 Review", disabled=True, width="large",
                        ),
                        "theme_1": st.column_config.SelectboxColumn(
                            "🏷️ Theme", options=theme1_options, required=True, width="medium",
                            help="The main category this review belongs to.",
                        ),
                        "theme_2_optional": st.column_config.SelectboxColumn(
                            "🏷️ Secondary Theme", options=theme2_options, width="medium",
                            help="Optional -- only set this if the review clearly covers a second, distinct theme.",
                        ),
                        "is_churn_risk": st.column_config.SelectboxColumn(
                            "⚠️ Churn Risk", options=churn_options, required=True, width="small",
                            help="Y if this review signals the customer may stop using the product. N otherwise.",
                        ),
                    },
                    hide_index=True,
                    width='stretch',
                    key=f"raw_editor_{run_key}",
                )

                with action_slot:
                    if st.button("💾 Save", key=f"save_corrections_{run_key}", type="primary", width='stretch'):
                        new_rows = edited_df.to_dict("records")
                        n_changed = log_corrections(rows, new_rows, run_key)
                        with open(CLASSIFIED_FILE, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["review_id", "theme_1", "theme_2_optional", "is_churn_risk"])
                            writer.writeheader()
                            for r in new_rows:
                                writer.writerow({
                                    "review_id": r["review_id"],
                                    "theme_1": r["theme_1"],
                                    "theme_2_optional": r["theme_2_optional"] or "",
                                    "is_churn_risk": r["is_churn_risk"],
                                })
                        st.session_state[edit_key] = False
                        if n_changed:
                            st.session_state[f"needs_regen_{run_key}"] = True
                            st.success(
                                f"Saved {n_changed} change(s). Click **Analyze reviews** above "
                                f"to refresh Theme Details and the Prioritized Roadmap."
                            )
                        else:
                            st.info("No changes detected.")
                        st.rerun()

                with analyze_slot:
                    st.button(
                        "Analyze reviews", key=f"raw_tab_analyze_editing_{run_key}",
                        type="primary", width='stretch', disabled=True,
                        help="Save your changes first -- then this button regenerates "
                             "Theme Details and the Prioritized Roadmap from them.",
                    )

            st.divider()
            st.markdown("**Prefer Excel or Google Sheets?**")
            st.caption("Download the table, correct it there, then upload it back.")
            col_dl, col_ul = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download",
                    data=df.to_csv(index=False),
                    file_name=f"classifications_{run_key.replace(':', '_')}.csv",
                    mime="text/csv",
                    width='stretch',
                )
            with col_ul:
                bulk_upload = st.file_uploader(
                    "⬆️ Upload corrected file", type="csv", key=f"bulk_upload_{run_key}",
                    help="Must include review_id, theme_1, theme_2_optional, is_churn_risk columns "
                         "(review_text is ignored if present -- it's only for your reference).",
                )
                if bulk_upload is not None:
                    upload_df = pd.read_csv(bulk_upload, dtype=str).fillna("")
                    required_cols = {"review_id", "theme_1", "theme_2_optional", "is_churn_risk"}
                    if not required_cols.issubset(set(upload_df.columns)):
                        st.error(f"Missing required columns: {', '.join(sorted(required_cols - set(upload_df.columns)))}")
                    else:
                        new_rows = upload_df.to_dict("records")
                        n_changed = log_corrections(rows, new_rows, run_key)
                        with open(CLASSIFIED_FILE, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["review_id", "theme_1", "theme_2_optional", "is_churn_risk"])
                            writer.writeheader()
                            for r in new_rows:
                                writer.writerow({
                                    "review_id": r["review_id"],
                                    "theme_1": r["theme_1"],
                                    "theme_2_optional": r.get("theme_2_optional", "") or "",
                                    "is_churn_risk": r["is_churn_risk"],
                                })
                        if n_changed:
                            st.session_state[f"needs_regen_{run_key}"] = True
                        st.success(
                            f"Uploaded and saved {n_changed} change(s). Scroll to the top "
                            f"and click **Update Results** to refresh downstream results."
                        )

            st.caption(
                "Note: corrections update this run's results immediately, but the AI model "
                "itself doesn't retrain from them. They're logged in corrections_log.csv so "
                "they can be used to expand the eval ground truth or improve the next prompt "
                "iteration -- a manual next step, not automatic."
            )
        else:
            st.info("Run the pipeline to see raw classifications.")

    with tab_guardrails:
        st.caption(
            "This pipeline has built-in checks that catch the AI when it's "
            "unreliable -- inventing categories, running out of evidence, or "
            "producing a thin sample. The counts below are proof those checks "
            "actually caught something in this run, not just that they exist in the code."
        )
        st.subheader("Guardrail activity", anchor=False)
        invalid_themes = [
            r for r in classified
            if str(r.get("theme_1", "")).startswith("INVALID_THEME_NEEDS_REVIEW")
            or str(r.get("theme_2_optional", "")).startswith("INVALID_THEME_NEEDS_REVIEW")
        ]
        missing = [r for r in classified if r.get("theme_1") == "MISSING_FROM_LLM_RESPONSE"]
        low_conf_themes = [r for r in theme_report if r.get("confidence") != "OK"]
        watch_list_themes = [r for r in recommendation if r.get("section") == "WATCH_LIST"]

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Taxonomy violations caught", len(invalid_themes),
            help="Stage 1: reviews where the model returned a category outside the fixed taxonomy list. Caught and flagged rather than trusted blindly.",
        )
        col2.metric(
            "Low-confidence themes excluded", len(low_conf_themes),
            help="Stage 2: themes with fewer than 3 supporting reviews, skipped rather than forcing a citation onto thin evidence.",
        )
        col3.metric(
            "Small-sample themes -> watch list", len(watch_list_themes),
            help="Stage 3: themes below this run's ranking threshold, held back so a small sample never outranks a statistically robust one.",
        )

        if missing:
            st.warning(f"{len(missing)} review(s) sent to the LLM never came back in its response "
                       f"and were flagged MISSING_FROM_LLM_RESPONSE rather than silently dropped.")

else:
    if is_custom_mode and company_name:
        st.info(f"No results yet for {company_name}. Upload a reviews CSV and click **Analyze reviews** above.")
    elif is_custom_mode:
        st.info("Enter a company name and upload a reviews CSV, then click **Analyze reviews** above.")
    else:
        st.info("Click **Analyze reviews** above to run the pipeline.")