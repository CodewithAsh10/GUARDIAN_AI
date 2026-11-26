"""
Guardian AI - Module 1
dashboard/pages/district_analysis.py
-----------------------------------
Page 4: District Analysis
District ranking, state filter, comparison, district-specific trends.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import analysis
from src import visualizations as viz


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">District Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Rank and compare districts by reported crime. Districts are identified "
        "by their state + district name (district names can repeat across states).",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- State filter ----
    st.markdown("### Filter by state")
    all_states = sorted(clean_df["state"].unique().tolist())
    sel_state = st.selectbox(
        "State (optional &mdash; view all if 'All')",
        options=["All"] + all_states,
        index=0,
        key="da_state",
    )

    # ---- District ranking ----
    st.markdown("### Top districts by total reported crime (all years)")
    if sel_state == "All":
        dist_sum = analysis.district_wise_summary(clean_df)
        fig = viz.plot_top_districts(clean_df, n=15)
    else:
        sub = clean_df[clean_df["state"] == sel_state]
        dist_sum = analysis.district_wise_summary(sub)
        import plotly.graph_objects as go
        top = dist_sum.head(15).iloc[::-1]
        labels = [f"{r['state']} - {r['district']}" for _, r in top.iterrows()]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=top["total_crime"],
                    y=labels,
                    orientation="h",
                    marker_color="#3498DB",
                    hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1A2B", plot_bgcolor="#14253D",
            font=dict(color="#AAB8C4", size=11), height=520,
            margin=dict(l=10, r=10, t=40, b=60),
            title=dict(text=f"Top 15 districts in {sel_state}", x=0.02, xanchor="left"),
            xaxis=dict(title_text="Total reported crime count", gridcolor="#2C3E50"),
            yaxis=dict(title_text="District", automargin=True),
        )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### District table")
    st.dataframe(dist_sum.head(20), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- District-specific trend ----
    st.markdown("### District-specific trend")
    st.caption("Year-wise total reported crime for a chosen district.")
    # Build a district picklist: "State - District"
    dist_list = sorted(
        [f"{r['state']} - {r['district']}" for _, r in dist_sum.head(60).iterrows()]
    )
    sel_dist = st.selectbox("Select district", options=dist_list, index=0, key="da_dist")
    if sel_dist:
        st_name, d_name = sel_dist.split(" - ", 1)
        trend = analysis.district_trend(clean_df, st_name, d_name)
        if trend.empty:
            st.info("No yearly data for this district.")
        else:
            import plotly.graph_objects as go
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=trend["year"].astype(str),
                        y=trend["total_crime"],
                        mode="lines+markers",
                        line=dict(color="#2E86C7", width=3),
                        marker=dict(size=8),
                        hovertemplate="Year: %{x}<br>Total crime: %{y:,.0f}<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1A2B", plot_bgcolor="#14253D",
                font=dict(color="#AAB8C4", size=11), height=420,
                margin=dict(l=10, r=10, t=40, b=60),
                title=dict(text=f"Trend: {sel_dist}", x=0.02, xanchor="left"),
                xaxis=dict(title_text="Year", gridcolor="#2C3E50"),
                yaxis=dict(title_text="Total reported crime count", gridcolor="#2C3E50"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info(
        "High reported counts at district level do not equal crime risk "
        "(population and reporting propensity are not accounted for). "
        "District-level data cannot identify streets or coordinates."
    )
