"""
Guardian AI - Module 1
dashboard/app.py
----------------
Main entry point for the Streamlit crime-intelligence dashboard.

Run:
    streamlit run dashboard/app.py

Design: dark navy background, professional blue accents, clean typography,
modern KPI cards, interactive Plotly charts, responsive layout.

Pages:
  1. Overview
  2. Crime Trends
  3. State Analysis
  4. District Analysis
  5. Hotspot Intelligence
  6. Geographical Intelligence
  7. Data Explorer

All figures and statistics are computed from the genuine NCRB dataset
via src/ modules -- no hardcoded or fake values.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Ensure `src` is importable when Streamlit runs from the project root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_master_dataset, clean_for_analysis, get_crime_columns, TOTAL_CRIME_COLUMN
from src.data_validation import validate_dataset, validation_summary_text
from src import analysis
from src import hotspot_detection as hd
from src import visualizations as viz
from src import geoviz

# ---------------------------------------------------------------------------
# Page imports (lazy, so a broken page does not kill the whole app)
# ---------------------------------------------------------------------------
PAGES = {
    "1. Overview": "dashboard.pages.overview",
    "2. Crime Trends": "dashboard.pages.crime_trends",
    "3. State Analysis": "dashboard.pages.state_analysis",
    "4. District Analysis": "dashboard.pages.district_analysis",
    "5. Hotspot Intelligence": "dashboard.pages.hotspot_intelligence",
    "6. Geographical Intelligence": "dashboard.pages.geographical_intelligence",
    "7. Data Explorer": "dashboard.pages.data_explorer",
}

# ---------------------------------------------------------------------------
# Page config + custom theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Guardian AI | Crime Intelligence Dashboard",
    page_icon="\U0001F6E1",  # police car
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Root theme variables */
:root {
  --navy: #0E1A2B;
  --navy-soft: #14253D;
  --blue: #2E86C7;
  --blue-light: #85C1E9;
  --accent: #3498DB;
  --grey: #AAB8C4;
  --danger: #E74C3C;
  --success: #27AE60;
  --warning: #F39C12;
}

/* App background */
.stApp {
  background: var(--navy);
  color: #E8EEF3;
}
header[data-testid="stHeader"] { background: var(--navy); }
section.main > div { background: var(--navy); padding-top: 1rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--navy-soft);
  border-right: 1px solid #2C3E50;
}
section[data-testid="stSidebar"] .css-1d391kg,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label {
  color: #E8EEF3 !important;
}

/* Widget text color */
.stSelectbox > div > div { background: var(--navy-soft); color: #E8EEF3; }
.stMultiSelect > div > div { background: var(--navy-soft); color: #E8EEF3; }
.stSlider > div { color: #E8EEF3; }

/* KPI cards */
.kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0 18px 0; }
.kpi-card {
  background: var(--navy-soft);
  border: 1px solid #2C3E50;
  border-left: 4px solid var(--blue);
  border-radius: 10px;
  padding: 14px 16px;
  flex: 1 1 150px;
  min-width: 140px;
  box-sizing: border-box;
}
.kpi-card .label {
  color: var(--grey);
  font-size: 11px;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.kpi-card .value {
  color: #FFFFFF;
  font-size: 26px;
  font-weight: 700;
  line-height: 1.15;
}
.kpi-card .sub {
  color: var(--blue-light);
  font-size: 11px;
  margin-top: 4px;
}

/* Section headings */
h1, h2, h3, .section-title {
  color: #FFFFFF !important;
  font-weight: 700;
  letter-spacing: 0.2px;
}
.section-title {
  font-size: 20px;
  margin: 6px 0 4px 0;
  padding-bottom: 6px;
  border-bottom: 2px solid var(--blue);
  display: inline-block;
}
.subtitle {
  color: var(--grey);
  font-size: 13px;
  margin-bottom: 16px;
}

/* Info / warning boxes */
.stAlert { background: var(--navy-soft) !important; border: 1px solid #2C3E50; }
.stAlert > div { color: #E8EEF3 !important; }

/* Dataframe */
.stDataFrame { background: var(--navy-soft); }

/* Footer */
.footer {
  margin-top: 30px;
  padding: 14px 0;
  border-top: 1px solid #2C3E50;
  color: var(--grey);
  font-size: 11px;
  text-align: center;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning NCRB dataset...")
def load_data():
    """
    Load + clean the dataset once, cache it. Returns (raw_df, clean_df, validation_report).
    The source CSV is never modified.
    """
    raw = load_master_dataset()
    clean = clean_for_analysis(raw)
    clean = analysis.add_total_crime_column(clean)
    report = validate_dataset(raw)
    return raw, clean, report


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding:10px 6px 6px 6px;'>"
            "<h2 style='margin:0; color:#FFFFFF;'>\U0001F6E1 Guardian AI</h2>"
            "<p style='margin:2px 0 10px 0; font-size:12px; color:#85C1E9;'>"
            "Module 1: Crime Analysis &amp; Intelligence</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Navigation")
        choice = st.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### Data source")
        st.caption("NCRB district-wise IPC crime data, 2017–2022.")
        st.caption("Raw counts are not crime risk (no population data).")
        st.markdown("---")
        st.markdown("### How to run")
        st.code("streamlit run dashboard/app.py", language="bash")
    return choice


def main():
    choice = render_sidebar()

    raw_df, clean_df, report = load_data()
    crime_cols = get_crime_columns(clean_df)

    # Route to the selected page
    module_name = PAGES[choice]
    try:
        mod = __import__(module_name, fromlist=["render"])
        mod.render(clean_df, raw_df, crime_cols, report)
    except Exception as exc:  # show a clear, friendly error instead of crashing
        st.error(f"Unable to load the '{choice}' page.")
        st.exception(exc)

    # Footer
    st.markdown(
        """
        <div class="footer">
          Guardian AI &middot; Module 1: Crime Analysis &amp; Intelligence &middot;
          NCRB district-wise IPC crime data, 2017&ndash;2022 &middot;
          Raw reported counts &ne; crime risk (no population data).
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
