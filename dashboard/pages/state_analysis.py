"""
Guardian AI - Module 1
dashboard/pages/state_analysis.py
--------------------------------
Page 3: State Analysis
State ranking, comparison, state-specific trends, category-wise state comparison.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import analysis
from src import visualizations as viz


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">State Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Compare states using raw reported crime counts. "
        "<b>Raw counts are not crime risk</b> &mdash; population data is not "
        "available, so a high count may reflect a larger population or more "
        "reporting, not higher risk.",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- State ranking ----
    st.markdown("### State ranking (total reported crime, 2017&ndash;2022)")
    fig_state = viz.plot_state_comparison(clean_df, n=12)
    st.plotly_chart(fig_state, use_container_width=True)

    # ---- Top/bottom tables ----
    state_sum = analysis.state_wise_summary(clean_df)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Highest-reporting states")
        st.dataframe(state_sum.head(8), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("#### Lowest-reporting states")
        st.dataframe(state_sum.sort_values("total_crime", ascending=True).head(8), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- State-specific trend ----
    st.markdown("### State-specific historical trend")
    st.caption("Year-wise total reported crime for selected states.")
    all_states = sorted(clean_df["state"].unique().tolist())
    default_states = list(state_sum.head(5)["state"])
    sel_states = st.multiselect(
        "Select states", options=all_states, default=default_states, key="sa_states",
        help="Pick 2-6 states for a readable comparison.",
    )
    if sel_states:
        fig_st = viz.plot_state_trend(clean_df, states=sel_states)
        st.plotly_chart(fig_st, use_container_width=True)
    else:
        st.info("Select at least one state.")

    st.markdown("---")

    # ---- Category-wise state comparison ----
    st.markdown("### Category-wise state comparison")
    st.caption("Total reported crime for a category across selected states.")
    sel_cat = st.selectbox(
        "Crime category", options=crime_cols, index=0, key="sa_cat",
        help="Choose a category to compare across states.",
    )
    sel_states2 = st.multiselect(
        "States to compare", options=all_states,
        default=list(state_sum.head(6)["state"]), key="sa_states2",
    )
    if sel_states2 and sel_cat:
        comp = analysis.state_category_comparison(clean_df, sel_states2, sel_cat)
        if comp.empty:
            st.info("No data for the selected states/category.")
        else:
            import plotly.graph_objects as go
            comp = comp.sort_values(sel_cat)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=comp[sel_cat],
                        y=comp["state"],
                        orientation="h",
                        marker_color="#3498DB",
                        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1A2B", plot_bgcolor="#14253D",
                font=dict(color="#AAB8C4", size=11),
                height=420, margin=dict(l=10, r=10, t=30, b=60),
                xaxis=dict(title_text="Total reported count", gridcolor="#2C3E50"),
                yaxis=dict(title_text="State", automargin=True),
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a category and at least one state.")

    st.markdown("---")
    st.info(
        "Methodology: each state's total is the sum of all reported IPC heads "
        "across its districts and all years (2017&ndash;2022). See notebook "
        "Section 9 for details."
    )
