"""
Guardian AI - Module 1
dashboard/pages/crime_trends.py
-------------------------------
Page 2: Crime Trends
Year selector, category filters, trend charts, percentage change, category comparison.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import analysis
from src import visualizations as viz


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">Crime Trends</div>', unsafe_allow_html=True)
    st.markdown(
        "Explore year-wise reported crime trends. Use the filters to focus on a "
        "specific year and/or offence category.",
        unsafe_allow_html=True,
    )

    # ---- Filters ----
    years = sorted(clean_df["year"].unique().tolist())
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_year = st.selectbox("Year", years, index=len(years) - 1, key="ct_year")
    with col2:
        default_cats = list(analysis.top_categories(clean_df, 6)["category"])
        selected_cats = st.multiselect(
            "Crime categories (trend chart)",
            options=crime_cols,
            default=default_cats,
            key="ct_cats",
            help="Select up to a few categories; too many lines become cluttered.",
        )

    st.markdown("---")

    # ---- Year-wise total trend ----
    st.markdown("### Total reported crime by year")
    st.caption("Grand total of all IPC crime heads per year.")
    fig_trend = viz.plot_yearly_trend(clean_df)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Percentage change ----
    st.markdown("### Year-over-year percentage change")
    st.caption("Change in total reported crime vs the previous year.")
    fig_change = viz.plot_yearly_change(clean_df)
    st.plotly_chart(fig_change, use_container_width=True)

    # ---- Category comparison for selected year ----
    st.markdown(f"### Category comparison &mdash; {selected_year}")
    st.caption("Reported counts by offence category for the selected year.")
    df_year = clean_df[clean_df["year"] == selected_year]
    if df_year.empty:
        st.warning(f"No data available for {selected_year}.")
    else:
        cat_year = df_year[crime_cols].sum().sort_values(ascending=False).head(12)
        cat_df = cat_year.reset_index()
        cat_df.columns = ["category", "count"]
        import plotly.graph_objects as go
        fig = go.Figure(
            data=[
                go.Bar(
                    x=cat_df["count"],
                    y=cat_df["category"],
                    orientation="h",
                    marker_color="#2E86C7",
                    hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1A2B",
            plot_bgcolor="#14253D",
            font=dict(color="#AAB8C4", size=11),
            height=520,
            margin=dict(l=10, r=10, t=30, b=60),
            xaxis=dict(title_text="Reported count", gridcolor="#2C3E50"),
            yaxis=dict(title_text="", automargin=True),
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Selected category trends ----
    st.markdown("### Selected category trends over time")
    st.caption("Year-wise totals for the chosen offence categories.")
    if selected_cats:
        fig_cat = viz.plot_category_yearly_trend(clean_df, categories=selected_cats)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Select at least one category from the filter above to see its trend.")

    st.markdown("---")
    st.info(
        "Trend direction compares the first vs last year for each category "
        "(increasing / decreasing / stable). View details in the notebook or "
        "Module 1 report."
    )
