"""
Guardian AI - Module 1
dashboard/pages/hotspot_intelligence.py
---------------------------------------
Page 5: Hotspot Intelligence
Historical high-reporting districts, repeated-hotspot analysis, ranking,
explainable methodology, historical comparisons.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import hotspot_detection as hd
from src import visualizations as viz


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">Hotspot Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        "Historical crime hotspots identified with a transparent, explainable "
        "method (no black-box ML). A district qualifies when it repeatedly "
        "appears among the highest-reporting districts across years.",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- Methodology ----
    with st.expander("Hotspot identification methodology (click to read)", expanded=True):
        st.markdown(hd.methodology_text().replace("\n", "  \n"))
        st.warning(
            "Important: these are HIGH REPORTED CRIME COUNTS, not crime risk. "
            "No population data is available, so rates per 100,000 people cannot "
            "be computed. District-level annual data cannot identify streets or "
            "coordinates."
        )

    st.markdown("---")

    # ---- Hotspot parameters ----
    st.markdown("### Configure hotspot detection")
    c1, c2 = st.columns(2)
    with c1:
        top_n = st.slider("Top-N districts per year", min_value=5, max_value=30, value=10, step=1, key="hs_topn")
    with c2:
        years_required = st.slider(
            "Minimum years in top-N to qualify", min_value=2, max_value=6, value=3, step=1, key="hs_yrs",
            help="Higher = more consistent hotspots only.",
        )

    st.markdown("---")

    # ---- Repeated hotspots table ----
    st.markdown("### Repeated historical hotspots")
    st.caption("Districts appearing in the yearly top-N for at least the selected number of years.")
    hs = hd.repeated_hotspots(clean_df, n=top_n, years_required=years_required)
    if hs.empty:
        st.info("No districts meet the chosen criteria. Try lowering the thresholds.")
    else:
        show_cols = [
            "state", "district", "appearances", "consistency_pct",
            "total_crime_all_years", "avg_yearly_crime", "hotspot_score",
        ]
        st.dataframe(
            hs[show_cols].head(20),
            use_container_width=True, hide_index=True,
            column_config={
                "hotspot_score": st.column_config.NumberColumn("Hotspot score (0-100)", format="%.1f"),
                "total_crime_all_years": st.column_config.NumberColumn("Total crime (all yrs)", format="%,.0f"),
                "consistency_pct": st.column_config.NumberColumn("Consistency %", format="%.1f"),
            },
        )

        # ---- Hotspot comparison chart ----
        st.markdown("### Hotspot ranking (score)")
        fig_hs = viz.plot_hotspot_comparison(clean_df, n=top_n)
        st.plotly_chart(fig_hs, use_container_width=True)

        # ---- Yearly top-N detail ----
        st.markdown("### Yearly top districts (top-N per year)")
        yearly_top = hd.top_n_districts_per_year(clean_df, n=top_n)
        yr_sel = st.selectbox(
            "Year", options=sorted(yearly_top.keys()), index=len(yearly_top) - 1, key="hs_yr",
        )
        st.dataframe(
            yearly_top[yr_sel],
            use_container_width=True, hide_index=True,
            column_config={"total_crime": st.column_config.NumberColumn(format="%,.0f")},
        )

    st.markdown("---")
    st.info(
        "A district with a high score is a repeat high-reporter. It is not "
        "necessarily the most dangerous place &mdash; population and reporting "
        "differences are not adjusted for (see caveats above)."
    )
