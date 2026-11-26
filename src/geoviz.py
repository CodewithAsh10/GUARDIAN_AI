"""
Guardian AI - Module 1
src/geoviz.py
-------------
Responsibility: Geographical visualisation of crime data.

Maps are DISTRICT-LEVEL CHOROPLETHS built from genuine Survey of India / NWIC
district boundary polygons joined to genuine NCRB district-level annual crime
records. Nothing is fabricated:

  - No invented latitude/longitude points.
  - No point-density "heatmaps".
  - No claims that district aggregates are street-level incident locations.

Key implementation notes (important for correctness):
  - The raw NWIC GeoJSON is in EPSG:7755 (projected metres). It MUST be
    reprojected to EPSG:4326 (WGS84 lat/lon) before it can be drawn on a
    lat/lon map. Without this the polygons fall off-screen and nothing appears.
  - The raw file is ~168 MB; geometries are simplified (~1 km tolerance) and
    cached to dataset/processed/india_districts_wgs84_simplified.geojson so the
    dashboard can load and render quickly.
  - NCRB and boundary district/state names are normalised, matched on
    state+district where possible, with conservative spelling-variant fuzzing
    and a small explicit alias table. No unsafe broad fuzzy matches are made.
  - Districts that cannot be matched (e.g. newly created districts not present
    in the older boundary vintage, or police commissionerates/CID/railway units)
    are preserved and reported transparently, never dropped silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .data_loader import TOTAL_CRIME_COLUMN, add_total_crime_column

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_RAW = PROJECT_ROOT / "dataset" / "raw"
DATASET_PROCESSED = PROJECT_ROOT / "dataset" / "processed"

# Raw NCRB project boundary GeoJSON (Survey of India / NWIC, EPSG:7755).
DISTRICT_GEOJSON_RAW = DATASET_RAW / "district_boundaries" / "district_nwic.GeoJSON"

# Cached, reprojected + simplified district boundaries (EPSG:4326).
SIMPLIFIED_CACHE = DATASET_PROCESSED / "india_districts_wgs84_simplified.geojson"

# Places to look for a user-supplied boundary GeoJSON (legacy / fallback).
BOUNDARY_CANDIDATES = [
    DATASET_PROCESSED / "india_districts.geojson",
    DATASET_PROCESSED / "india_states.geojson",
    DATASET_RAW / "boundaries" / "india_districts.geojson",
    DATASET_RAW / "boundaries" / "india_states.geojson",
    DISTRICT_GEOJSON_RAW,
]

# Default map centre (geographic centre of India) and zoom (legacy folium).
INDIA_CENTRE = [22.0, 78.0]
INDIA_ZOOM = 5

# Simplify tolerance in degrees (~1.1 km). Keeps shapes faithful, cuts size.
SIMPLIFY_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------
def _norm_name(value) -> str:
    """Lowercase, strip, unify &->and, drop punctuation, collapse whitespace."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _squash(normalised: str) -> str:
    """Remove all spaces, for tolerant but still deterministic comparison."""
    return normalised.replace(" ", "")


# State-name canonicalisation (applied to BOTH NCRB and boundary names).
_STATE_CANON = {
    "arunanchal pradesh": "arunachal pradesh",
    "andaman and nicobar island": "andaman nicobar islands",
    "andaman and nicobar islands": "andaman nicobar islands",
    "a and n islands": "andaman nicobar islands",
    "dadra and nagar havelli and daman and diu": "dadra nagar haveli daman diu",
    "d and n haveli and daman and diu": "dadra nagar haveli daman diu",
    "d and n haveli": "dadra nagar haveli daman diu",
    "daman and diu": "dadra nagar haveli daman diu",
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "uttaranchal": "uttarakhand",
    "telengana": "telangana",
    "jammu and kashmir": "jammu kashmir",
}


def _canon_state(value) -> str:
    n = _norm_name(value)
    return _STATE_CANON.get(n, n)


# Well-known district renames / transliterations mapped to a single canonical
# token. Only unambiguous official renames are included; nothing speculative.
_DISTRICT_RENAME = {
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",
    "mysore": "mysuru",
    "mysuru": "mysuru",
    "mangalore": "mangaluru",
    "mangaluru": "mangaluru",
    "calcutta": "kolkata",
    "kolkata": "kolkata",
    "madras": "chennai",
    "chennai": "chennai",
    "bombay": "mumbai",
    "mumbai": "mumbai",
    "gurgaon": "gurugram",
    "gurugram": "gurugram",
    "allahabad": "prayagraj",
    "prayagraj": "prayagraj",
    "pondicherry": "puducherry",
    "puducherry": "puducherry",
}


def _canon_district(value) -> str:
    n = _norm_name(value)
    return _DISTRICT_RENAME.get(n, n)


# Explicit, safe (state, district) alias fixes where NCRB and boundary use
# different spellings for the SAME administrative district.
# Key: (canon_state, canon_district_ncrb) -> canon_district_geo
_DISTRICT_ALIASES: Dict[Tuple[str, str], str] = {
    ("andhra pradesh", "anantapuramu"): "anantapur",
    ("andhra pradesh", "sri potti sriramulu nellore"): "spsr nellore",
    ("andhra pradesh", "ysr"): "y s r",
    ("andhra pradesh", "y s r"): "y s r",
    ("arunachal pradesh", "kameng east"): "east kameng",
    ("arunachal pradesh", "kameng west"): "west kameng",
    ("arunachal pradesh", "siang east"): "east siang",
    ("arunachal pradesh", "siang west"): "west siang",
    ("arunachal pradesh", "siang upper"): "upper siang",
    ("arunachal pradesh", "siang"): "siang",
    ("arunachal pradesh", "subansiri lower"): "lower subansiri",
    ("arunachal pradesh", "subansiri upper"): "upper subansiri",
    ("arunachal pradesh", "kurung kumey"): "kurung kumey",
    ("arunachal pradesh", "pakke kessang"): "pakke kessang",
    ("arunachal pradesh", "pakkessang"): "pakke kessang",
    ("arunachal pradesh", "kra daadi"): "kra daadi",
    ("arunachal pradesh", "kra dadi"): "kra daadi",
    ("arunachal pradesh", "shi yomi"): "shi yomi",
    ("arunachal pradesh", "shiyomi"): "shi yomi",
    ("arunachal pradesh", "leparada"): "leparada",
}


# ---------------------------------------------------------------------------
# GeoPandas lazy import
# ---------------------------------------------------------------------------
def _require_geopandas():
    try:
        import geopandas as gpd  # noqa: F401
        return gpd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "GeoPandas is required for district choropleth maps. "
            "Install it with `pip install geopandas`. "
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Boundary discovery / legacy folium helpers (kept for compatibility)
# ---------------------------------------------------------------------------
def find_boundary_file() -> Optional[Path]:
    """Return the first existing district boundary GeoJSON candidate, or None."""
    if SIMPLIFIED_CACHE.exists():
        return SIMPLIFIED_CACHE
    for cand in BOUNDARY_CANDIDATES:
        if cand.exists():
            return cand
    return None


def load_boundary(path: Path) -> dict:
    """Load a GeoJSON boundary file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_india_map(location: Optional[list] = None, zoom: int = INDIA_ZOOM):
    """Return a Folium map centred on India (legacy helper)."""
    import folium

    loc = location if location is not None else INDIA_CENTRE
    return folium.Map(location=loc, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)


def geographical_ranking(df: pd.DataFrame, year: int, measure: str = "total_crime", n: int = 15) -> pd.DataFrame:
    """State-level ranking fallback when no boundary file exists."""
    df = add_total_crime_column(df)
    sub = df[df["year"] == year]
    g = sub.groupby("state")[measure].sum().reset_index().sort_values(measure, ascending=False).head(n)
    g = g.rename(columns={measure: "value"})
    g["year"] = year
    return g[["state", "value", "year"]]


# ---------------------------------------------------------------------------
# District boundary loading (reproject + simplify + cache)
# ---------------------------------------------------------------------------
def _build_simplified_cache() -> "object":
    """Read raw EPSG:7755 boundaries, reproject to WGS84, simplify, and cache."""
    gpd = _require_geopandas()

    if not DISTRICT_GEOJSON_RAW.exists():
        raise FileNotFoundError(
            f"District boundary GeoJSON not found at {DISTRICT_GEOJSON_RAW}."
        )

    gdf = gpd.read_file(DISTRICT_GEOJSON_RAW)

    # Reproject to WGS84 lat/lon -- ESSENTIAL for correct map placement.
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Repair invalid geometries, drop null/empty.
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    # Simplify to keep size manageable for browser rendering.
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # Keep only the columns we need.
    keep = ["state_name", "district", "geometry"]
    for c in keep:
        if c not in gdf.columns:
            gdf[c] = ""
    gdf = gdf[keep].reset_index(drop=True)

    # Stable unique key for feature-id matching.
    gdf["_geo_key"] = [str(i) for i in range(len(gdf))]

    DATASET_PROCESSED.mkdir(parents=True, exist_ok=True)
    try:
        gdf.to_file(SIMPLIFIED_CACHE, driver="GeoJSON")
    except Exception:
        # Caching is an optimisation; failure to write is not fatal.
        pass
    return gdf


def load_district_geo(force_rebuild: bool = False):
    """
    Load the district boundary GeoDataFrame in EPSG:4326.

    Returns a GeoDataFrame with columns: state_name, district, geometry,
    _geo_key (stable string id). Uses a simplified on-disk cache for speed.
    """
    gpd = _require_geopandas()

    if SIMPLIFIED_CACHE.exists() and not force_rebuild:
        try:
            gdf = gpd.read_file(SIMPLIFIED_CACHE)
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.set_crs(4326, allow_override=True)
            if "_geo_key" not in gdf.columns:
                gdf = gdf.reset_index(drop=True)
                gdf["_geo_key"] = [str(i) for i in range(len(gdf))]
            return gdf
        except Exception:
            # Fall through to rebuild if the cache is corrupt.
            pass

    return _build_simplified_cache()


def geodataframe_to_geojson(gdf) -> dict:
    """Convert a GeoDataFrame to a compact GeoJSON dict with a geo_key property."""
    from shapely.geometry import mapping

    features = []
    for _, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "id": str(r["_geo_key"]),
                "properties": {
                    "geo_key": str(r["_geo_key"]),
                    "state_name": r.get("state_name", ""),
                    "district": r.get("district", ""),
                },
                "geometry": mapping(geom),
            }
        )
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# District matching (NCRB records -> boundary polygons)
# ---------------------------------------------------------------------------
def _build_geo_index(gdf) -> Tuple[Dict, Dict, Dict]:
    """Build lookup structures on the boundary side."""
    by_sd: Dict[Tuple[str, str], int] = {}
    by_sdsq: Dict[Tuple[str, str], int] = {}
    state_dists: Dict[str, List[Tuple[str, str, int]]] = {}

    for i, r in gdf.iterrows():
        sc = _canon_state(r.get("state_name", ""))
        dc = _canon_district(r.get("district", ""))
        dq = _squash(dc)
        by_sd[(sc, dc)] = i
        by_sdsq[(sc, dq)] = i
        state_dists.setdefault(sc, []).append((dc, dq, i))
    return by_sd, by_sdsq, state_dists


def match_district_crimes(
    df: pd.DataFrame,
    year: int,
    measure: str,
    gdf,
) -> Tuple["object", dict]:
    """
    Join NCRB crime records for ``year``/``measure`` to boundary polygons.

    Returns
    -------
    merged_gdf : GeoDataFrame
        Boundary polygons with an added ``value`` column (NaN = no data) plus
        ``ncrb_state`` / ``ncrb_district`` / ``match_method`` for hover/debug.
    report : dict
        Matching diagnostics: totals, matched/unmatched counts, coverage, and
        the list of unmatched NCRB districts with the reason.
    """
    from difflib import SequenceMatcher

    df = add_total_crime_column(df)
    if measure not in df.columns:
        raise KeyError(f"Crime measure column '{measure}' not found in dataset.")

    sub = df[df["year"] == year]
    if sub.empty:
        return gdf.iloc[0:0].copy(), {
            "error": f"No data for year {year}.",
            "total_polygons": int(len(gdf)),
            "total_ncrb_districts": 0,
            "matched": 0,
            "unmatched": 0,
            "coverage_pct": 0.0,
            "unmatched_records": [],
        }

    # NCRB district-year aggregation.
    agg = (
        sub.groupby(["state", "district"])[measure]
        .sum()
        .reset_index()
    )
    total_ncrb = len(agg)

    by_sd, by_sdsq, state_dists = _build_geo_index(gdf)

    # Assign each NCRB row -> geo index (or None). Track method for transparency.
    assigned: Dict[int, float] = {}
    meta: Dict[int, dict] = {}
    unmatched: List[dict] = []
    method_counts = {"exact": 0, "alias": 0, "squashed": 0, "fuzzy": 0, "unmatched": 0}

    for _, row in agg.iterrows():
        raw_state = row["state"]
        raw_dist = row["district"]
        val = float(row[measure])

        sc = _canon_state(raw_state)
        dc = _canon_district(raw_dist)
        dq = _squash(dc)

        geo_idx: Optional[int] = None
        method = "unmatched"

        # 1) exact canonical (state, district)
        if (sc, dc) in by_sd:
            geo_idx = by_sd[(sc, dc)]
            method = "exact"
        # 2) explicit alias table
        elif (sc, dc) in _DISTRICT_ALIASES:
            target = _DISTRICT_ALIASES[(sc, dc)]
            if (sc, target) in by_sd:
                geo_idx = by_sd[(sc, target)]
                method = "alias"
            elif (sc, _squash(target)) in by_sdsq:
                geo_idx = by_sdsq[(sc, _squash(target))]
                method = "alias"
        # 3) squashed (ignore spaces) within same canonical state
        if geo_idx is None and (sc, dq) in by_sdsq:
            geo_idx = by_sdsq[(sc, dq)]
            method = "squashed"
        # 4) conservative fuzzy within the same state (unique best, ratio>=0.88)
        if geo_idx is None:
            best_ratio = 0.0
            best_idx = None
            second_ratio = 0.0
            for (gdc, gdq, gi) in state_dists.get(sc, []):
                ratio = SequenceMatcher(None, dc, gdc).ratio()
                if ratio > best_ratio:
                    second_ratio = best_ratio
                    best_ratio = ratio
                    best_idx = gi
                elif ratio > second_ratio:
                    second_ratio = ratio
            if best_idx is not None and best_ratio >= 0.88 and (best_ratio - second_ratio) >= 0.05:
                geo_idx = best_idx
                method = "fuzzy"

        if geo_idx is None:
            method_counts["unmatched"] += 1
            unmatched.append(
                {
                    "state": raw_state,
                    "district": raw_dist,
                    "value": val,
                    "reason": "no boundary polygon (new/unmapped district or non-administrative unit)",
                }
            )
            continue

        method_counts[method] += 1
        assigned[geo_idx] = assigned.get(geo_idx, 0.0) + val
        # Keep the first NCRB label per polygon for hover text.
        meta.setdefault(
            geo_idx,
            {
                "ncrb_state": raw_state,
                "ncrb_district": raw_dist,
                "match_method": method,
            },
        )

    # Build merged GeoDataFrame (all polygons; value NaN for no-data).
    merged = gdf.copy()
    merged["value"] = merged.index.map(lambda i: assigned.get(i))
    for col in ("ncrb_state", "ncrb_district", "match_method"):
        merged[col] = merged.index.map(
            lambda i, c=col: meta.get(i, {}).get(c)
        )
    merged["year"] = year
    merged["measure"] = measure

    matched_polygons = int(merged["value"].notna().sum())
    matched_rows = sum(
        v for k, v in method_counts.items() if k != "unmatched"
    )
    coverage_pct = (matched_rows / total_ncrb * 100.0) if total_ncrb else 0.0

    report = {
        "year": int(year),
        "measure": measure,
        "total_polygons": int(len(gdf)),
        "total_ncrb_districts": int(total_ncrb),
        "matched_records": int(matched_rows),
        "matched_polygons": matched_polygons,
        "unmatched": int(method_counts["unmatched"]),
        "coverage_pct": round(coverage_pct, 1),
        "method_counts": method_counts,
        "unmatched_records": sorted(
            unmatched, key=lambda d: (d["state"], d["district"])
        ),
    }
    return merged, report


# ---------------------------------------------------------------------------
# Plotly district choropleth
# ---------------------------------------------------------------------------
def build_district_choropleth(
    merged_gdf,
    geojson: dict,
    measure: str,
    year: int,
    measure_label: Optional[str] = None,
):
    """
    Build a Plotly district-level choropleth figure.

    - All boundary polygons are shown (dark base layer) so the map is complete.
    - Matched districts are coloured on a continuous YlOrRd scale.
    - A colour bar acts as the legend.
    - Tooltips show state, district, year and the reported count.
    """
    import plotly.graph_objects as go

    label = measure_label or ("Total reported crime" if measure == "total_crime" else measure)

    keys_all = [str(k) for k in merged_gdf["_geo_key"]]

    fig = go.Figure()

    # Base layer: every district polygon, neutral dark fill + thin outline.
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.geo_key",
            locations=keys_all,
            z=[1] * len(keys_all),
            colorscale=[[0, "#1b2a3a"], [1, "#1b2a3a"]],
            showscale=False,
            marker_line_color="#2C3E50",
            marker_line_width=0.4,
            hoverinfo="skip",
            name="No data",
        )
    )

    matched = merged_gdf[merged_gdf["value"].notna()].copy()

    if not matched.empty:
        matched = matched.sort_values("value")
        z = matched["value"].astype(float)
        zmin = float(z.min())
        zmax = float(z.max())
        # Avoid a flat/zero range.
        if zmax <= zmin:
            zmax = zmin + 1.0

        customdata = [
            [
                str(r.get("ncrb_state") or r.get("state_name") or ""),
                str(r.get("ncrb_district") or r.get("district") or ""),
                int(year),
            ]
            for _, r in matched.iterrows()
        ]

        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                featureidkey="properties.geo_key",
                locations=[str(k) for k in matched["_geo_key"]],
                z=z,
                zmin=zmin,
                zmax=zmax,
                colorscale="YlOrRd",
                marker_line_color="#0E1A2B",
                marker_line_width=0.5,
                colorbar=dict(
                    title=dict(text=f"{label}<br>{year}", side="right"),
                    thickness=14,
                    len=0.7,
                    tickformat=",.0f",
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "State: %{customdata[0]}<br>"
                    "Year: %{customdata[2]}<br>"
                    f"{label}: %{{z:,.0f}}<extra></extra>"
                ),
                name=label,
            )
        )

    # Fit view to India without external basemap tiles.
    fig.update_geos(
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        fitbounds="locations",
        lonaxis=dict(range=[67.0, 98.5]),
        lataxis=dict(range=[6.0, 38.5]),
        projection_type="mercator",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1A2B",
        plot_bgcolor="#0E1A2B",
        font=dict(color="#AAB8C4", size=12),
        margin=dict(l=6, r=6, t=30, b=6),
        height=680,
        showlegend=False,
        title=dict(
            text=f"District crime choropleth \u2014 {label} ({year})",
            x=0.01,
            xanchor="left",
        ),
    )
    return fig
