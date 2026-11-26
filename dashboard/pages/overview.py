"""
Guardian AI - Module 1
dashboard/pages/overview.py
---------------------------
Page 1: Overview
KPIs + overall trend + key observations for the selected year.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import analysis
from src import visualizations as viz
from src.kpi import kpi_card


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "Reported crime intelligence at a glance. All figures are computed from "
        "the genuine NCRB district-wise IPC dataset (2017&ndash;2022). "
        "<b>Raw counts are not crime risk</b> &mdash; no population data is used.",
        unsafe_allow_html=True,
    )

    # ---- Year selector ----
    years = sorted(clean_df["year"].unique().tolist())
    col_year, _ = st.columns([1, 3])
    with col_year:
        selected_year = st.selectbox("Select year", years, index=len(years) - 1, key="ov_year")

    # ---- KPIs for selected year ----
    k = analysis.kpis_for_year(clean_df, selected_year)

    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    kpi_card("Total reported crime", f"{k['total_crime']:,}", f"Year {k['year']} &middot; {k['record_count']} records")
    kpi_card("States", f"{k['states']:,}", "reporting states/UTs")
    kpi_card("Districts", f"{k['districts']:,}", "reporting districts")
    kpi_card("Crime categories", f"{k['categories']:,}", "IPC offence heads")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Overall trend ----
    st.markdown("### Overall reported crime trend (2017&ndash;2022)")
    st.caption("Line shows the grand total of all IPC crime heads per year.")
    fig_trend = viz.plot_yearly_trend(clean_df)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Year-over-year change ----
    st.markdown("### Year-over-year change")
    st.caption("Absolute and percentage change in total reported crime vs the previous year.")
    fig_change = viz.plot_yearly_change(clean_df)
    st.plotly_chart(fig_change, use_container_width=True)

    # ---- Key observations ----
    st.markdown("### Key observations")
    change = analysis.year_wise_change(clean_df)
    obs = []
    for _, r in change.dropna().iterrows():
        yr = int(r["year"])
        pc = float(r["pct_change"])
        direction = "increase" if pc > 0 else "decrease"
        obs.append(f"In **{yr}**, total reported crime {direction}d by {abs(pc):.1f}% vs {yr-1}.")
    # Highest/lowest year
    yearly = analysis.year_wise_summary(clean_df)
    peak = yearly.loc[yearly["total_crime"].idxmax()]
    low = yearly.loc[yearly["total_crime"].idxmin()]
    obs.append(
        f"Highest reported total: {int(peak['total_crime']):,} in {int(peak['year'])}; "
        f"lowest: {int(low['total_crime']):,} in {int(low['year'])}."
    )
    # 2020 note
    obs.append(
        "The 2020 jump coincides with the pandemic year and reporting changes; "
        "it should not be interpreted as a standalone crime surge without population "
        "and reporting-context adjustment."
    )
    for o in obs:
        st.markdown(f"- {o}")

    st.markdown("---")
    st.info(
        "Data quality: validation passed = "
        f"{'Yes' if report.get('passed') else 'No'}. "
        f"Duplicate district-year records: {report.get('duplicates', {}).get('year_state_district', 0)}. "
        "See Data Explorer for the full filterable table."
    )
