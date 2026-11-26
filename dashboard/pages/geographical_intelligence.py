"""
Guardian AI - Module 1
dashboard/pages/geographical_intelligence.py
--------------------------------------------
Page 6: Geographical Intelligence

Renders a genuine DISTRICT-LEVEL crime choropleth built from Survey of India /
NWIC district boundary polygons joined to NCRB district-level annual records.

Honesty guarantees:
  - No fabricated coordinates or point-density heatmaps.
  - Boundaries are real administrative polygons (reprojected to WGS84).
  - NCRB data is district-level annual aggregates (area-level, not street-level).
  - Unmatched districts are shown transparently, never silently dropped.

The page also shows matching diagnostics (coverage, unmatched table) so the
map's completeness is auditable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import geoviz
from src import analysis


@st.cache_resource(show_spinner="Loading district boundary polygons (first run may take ~30s)...")
def _load_geo():
    """Load (once) the reprojected/simplified district boundaries + GeoJSON."""
    gdf = geoviz.load_district_geo()
    geojson = geoviz.geodataframe_to_geojson(gdf)
    return gdf, geojson


def render(clean_df, raw_df, crime_cols, report):
    st.markdown('<div class="section-title">Geographical Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        "District-level crime <b>choropleth</b> using genuine Survey of India / "
        "NWIC district boundary polygons joined to NCRB district-level records. "
        "NCRB data is <b>district-level annual aggregates</b> &mdash; the map shows "
        "area-level reported counts, not exact incident locations.",
        unsafe_allow_html=True,
    )

    # ---- Filters -----------------------------------------------------------
    st.markdown("### Filters")
    years = sorted(clean_df["year"].unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        selected_year = st.selectbox(
            "Year", years, index=len(years) - 1, key="gi_year",
            help="NCRB data covers 2017-2022.",
        )
    with c2:
        measure = st.selectbox(
            "Crime measure",
            options=["total_crime"] + crime_cols,
            index=0,
            key="gi_measure",
            format_func=lambda m: "Total reported crime (all IPC heads)"
            if m == "total_crime" else m,
            help="Choose the total or a specific IPC offence head.",
        )

    st.markdown("---")

    # ---- Load boundaries ---------------------------------------------------
    try:
        gdf, geojson = _load_geo()
    except Exception as exc:
        st.error(
            "District boundary data could not be loaded. The geographical "
            "choropleth requires GeoPandas and the district boundary GeoJSON at "
            "`dataset/raw/district_boundaries/district_nwic.GeoJSON`."
        )
        st.exception(exc)
        _render_ranking_alternative(clean_df, selected_year, measure)
        return

    if gdf is None or len(gdf) == 0:
        st.warning(
            "No district boundary polygons are available. Showing the ranking "
            "alternative instead. Place a genuine GeoJSON/GeoPackage/Shapefile in "
            "`dataset/raw/district_boundaries/` to enable the map."
        )
        _render_ranking_alternative(clean_df, selected_year, measure)
        return

    # ---- Match NCRB records to polygons -----------------------------------
    try:
        merged, geo_report = geoviz.match_district_crimes(
            clean_df, selected_year, measure, gdf
        )
    except Exception as exc:
        st.error("Failed to match NCRB records to boundary polygons.")
        st.exception(exc)
        return

    if geo_report.get("error"):
        st.warning(geo_report["error"])
        _render_ranking_alternative(clean_df, selected_year, measure)
        return

    # ---- Build + show the choropleth --------------------------------------
    try:
        measure_label = (
            "Total reported crime" if measure == "total_crime" else measure
        )
        fig = geoviz.build_district_choropleth(
            merged, geojson, measure, selected_year, measure_label=measure_label
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error("Failed to render the district choropleth.")
        st.exception(exc)
        return

    st.caption(
        "Colour intensity = reported count for the selected measure and year. "
        "Darker polygons have no matched data for this year. Hover a district "
        "for its state, district, year and count."
    )

    # ---- Matching diagnostics ---------------------------------------------
    st.markdown("### Map data coverage & matching diagnostics")
    row1 = st.columns(3)
    row1[0].metric("Boundary polygons", f"{geo_report['total_polygons']:,}")
    row1[1].metric("NCRB districts (year)", f"{geo_report['total_ncrb_districts']:,}")
    row1[2].metric("Matched districts", f"{geo_report['matched_polygons']:,}")
    row2 = st.columns(3)
    row2[0].metric("Unmatched records", f"{geo_report['unmatched']:,}")
    row2[1].metric("Match coverage", f"{geo_report['coverage_pct']:.1f}%")
    row2[2].metric(
        "Fully mapped polygons",
        f"{geo_report['matched_polygons']:,} / {geo_report['total_polygons']:,}",
    )

    mc = geo_report.get("method_counts", {})
    st.caption(
        "Match methods: "
        f"exact={mc.get('exact', 0):,}, alias={mc.get('alias', 0):,}, "
        f"spacing={mc.get('squashed', 0):,}, spelling-fuzzy={mc.get('fuzzy', 0):,}, "
        f"unmatched={mc.get('unmatched', 0):,}."
    )

    unmatched = geo_report.get("unmatched_records", [])
    if unmatched:
        with st.expander(
            f"Unmatched records ({len(unmatched)}) &mdash; click to inspect",
            expanded=False,
        ):
            st.caption(
                "These NCRB records have no corresponding polygon in the boundary "
                "file. Common reasons: districts created after the boundary file's "
                "vintage (e.g. the 2022 Andhra Pradesh reorganisation), police "
                "commissionerates, CID/GRP/railway units, or spelling variants not "
                "safely resolvable. They are shown here rather than dropped."
            )
            unm_df = pd.DataFrame(unmatched)
            if "value" in unm_df.columns:
                unm_df["value"] = pd.to_numeric(unm_df["value"], errors="coerce").round(0)
            st.dataframe(unm_df, use_container_width=True, hide_index=True, height=300)

    # ---- Methodology ------------------------------------------------------
    st.markdown("---")
    with st.expander("Methodology & limitations (click to read)"):
        st.markdown(
            "- **Boundaries**: Survey of India / NWIC district polygons "
            "(`district_nwic.GeoJSON`), reprojected from EPSG:7755 to WGS84 "
            "(EPSG:4326) and simplified (~1 km) for fast browser rendering.\n"
            "- **Data**: NCRB district-level annual reported counts, 2017-2022. "
            "Values are summed per district for the selected year and measure.\n"
            "- **Matching**: state+district names normalised (case, punctuation, "
            "`&`/`and`, spaces) and joined on both state and district. A small "
            "explicit alias table plus conservative same-state spelling fuzzing "
            "(similarity >= 0.88 with a clear margin) resolves known variants. No "
            "unsafe broad fuzzy matches are made.\n"
            "- **Unmatched districts** are reported, not hidden. The boundary file "
            "predates some district reorganisations, so newly created districts "
            "and non-administrative police units cannot be mapped.\n"
            "- **Not risk**: raw counts are not crime rates. No population data is "
            "available, so per-100,000 rates cannot be computed.\n"
            "- **Area-level only**: district aggregates cannot identify streets or "
            "individual incident coordinates. No fake points are plotted."
        )


def _render_ranking_alternative(df: pd.DataFrame, year: int, measure: str):
    """State/district ranking fallback when no boundary file is available."""
    import plotly.graph_objects as go

    st.markdown(f"### Geographical ranking &mdash; {measure} ({year})")
    st.caption("Highest-reporting states/districts for the selected year and measure.")
    df_year = df[df["year"] == year]
    if df_year.empty:
        st.warning(f"No data for {year}.")
        return

    rank = (
        df_year.groupby("state")[measure]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    rank.columns = ["state", "value"]
    rank["value"] = rank["value"].round(0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=rank["value"],
                y=rank["state"],
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
        xaxis=dict(title_text=f"Total reported {measure}", gridcolor="#2C3E50"),
        yaxis=dict(title_text="State", automargin=True),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top districts")
    dist_rank = (
        df_year.groupby(["state", "district"])[measure]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    dist_rank.columns = ["state", "district", "value"]
    dist_rank["value"] = dist_rank["value"].round(0)
    dist_rank["label"] = dist_rank["state"] + " - " + dist_rank["district"]
    fig2 = go.Figure(
        data=[
            go.Bar(
                x=dist_rank["value"],
                y=dist_rank["label"],
                orientation="h",
                marker_color="#3498DB",
                hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
            )
        ]
    )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1A2B",
        plot_bgcolor="#14253D",
        font=dict(color="#AAB8C4", size=11),
        height=520,
        margin=dict(l=10, r=10, t=30, b=60),
        xaxis=dict(title_text=f"Total reported {measure}", gridcolor="#2C3E50"),
        yaxis=dict(title_text="District (State)", automargin=True),
    )
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)
