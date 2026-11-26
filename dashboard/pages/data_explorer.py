"""
Guardian AI - Module 1
dashboard/pages/data_explorer.py
--------------------------------
Page 7: Data Explorer
Filterable dataset table + CSV download of filtered data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import get_crime_columns, TOTAL_CRIME_COLUMN


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        "Browse and filter the cleaned NCRB dataset. Download the filtered result "
        "as CSV. All data is genuine; nothing is fabricated.",
        unsafe_allow_html=True,
    )

    df = clean_df.copy()
    df[TOTAL_CRIME_COLUMN] = df[TOTAL_CRIME_COLUMN].round(0)

    # ---- Filters ----
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)
    all_states = sorted(df["state"].unique().tolist())
    all_years = sorted(df["year"].unique().tolist())
    all_districts = sorted(df["district"].unique().tolist())

    with f1:
        sel_states = st.multiselect("State", options=all_states, default=[], key="de_states")
    with f2:
        sel_years = st.multiselect("Year", options=all_years, default=[], key="de_years")
    with f3:
        sel_districts = st.multiselect("District", options=all_districts, default=[], key="de_districts")
    with f4:
        sel_cats = st.multiselect(
            "Crime category (filter rows where this category > 0)",
            options=crime_cols,
            default=[],
            key="de_cats",
            help="Shows only rows where the selected category count is greater than 0.",
        )

    # ---- Apply filters ----
    filtered = df
    if sel_states:
        filtered = filtered[filtered["state"].isin(sel_states)]
    if sel_years:
        filtered = filtered[filtered["year"].isin(sel_years)]
    if sel_districts:
        filtered = filtered[filtered["district"].isin(sel_districts)]
    if sel_cats:
        mask = pd.Series(False, index=filtered.index)
        for c in sel_cats:
            mask = mask | (filtered[c] > 0)
        filtered = filtered[mask]

    st.markdown("---")

    # ---- Summary ----
    st.markdown("### Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Matching records", f"{len(filtered):,}")
    c2.metric("Unique states", f"{filtered['state'].nunique():,}")
    c3.metric("Unique districts", f"{filtered['district'].nunique():,}")

    if not filtered.empty:
        total_crime = int(filtered[TOTAL_CRIME_COLUMN].sum())
        st.metric("Total reported crime (filtered)", f"{total_crime:,}")

    # ---- Table ----
    st.markdown("#### Dataset table")
    show_cols = ["year", "state", "district", TOTAL_CRIME_COLUMN] + crime_cols
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(
        filtered[show_cols],
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # ---- Download ----
    st.markdown("#### Download filtered data")
    if not filtered.empty:
        csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"guardian_crime_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="de_download",
        )
    else:
        st.info("No records match the current filters. Adjust filters to enable download.")

    st.markdown("---")
    st.info(
        "Columns include all provenance fields (source_file, source_sheet, etc.) "
        "and every IPC crime category. Use the State/District/Year/Category "
        "filters above to slice the data."
    )
