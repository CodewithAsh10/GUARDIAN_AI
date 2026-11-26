"""
Guardian AI - Module 1
src/hotspot_detection.py
------------------------
Responsibility: Transparent, explainable historical crime-hotspot identification.

METHODOLOGY (documented for faculty review / viva)
--------------------------------------------------
A district is flagged as a historical hotspot using a rule-based, explainable
score -- NOT a black-box model. The score combines two intuitive signals:

  1. MAGNITUDE: How large the district's reported crime total is relative to
     other districts (rank-based).
  2. CONSISTENCY: How many years the district appears among the top-N
     highest-reporting districts (repeat appearance).

Hotspot score (0 to 100):
    score = 0.6 * magnitude_component + 0.4 * consistency_component

  magnitude_component  = (district total crime / max total crime across all
                          districts) * 100        -- 0..100
  consistency_component = (years in top-N / total years analysed) * 100     -- 0..100

A district qualifies as a hotspot if:
    consistency_component >= consistency_threshold   (default: appears in top-N
                                                      in at least 3 of 6 years)

Interpretation guardrails (academic honesty):
  - This identifies HIGH REPORTED CRIME COUNTS, not crime risk.
  - No population data is available, so we CANNOT compute crime rates per
    100,000 people. Raw counts are NOT risk.
  - Data is district-level annual aggregates. This does NOT identify streets
    or coordinates.
  - Reporting behaviour (e.g. registration propensity, one-year data
    substitutions documented in data_quality_note) affects counts.

All functions return new DataFrames and never modify the input.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .data_loader import TOTAL_CRIME_COLUMN, add_total_crime_column
from .analysis import district_wise_summary


def _ensure_total(df: pd.DataFrame) -> pd.DataFrame:
    if "total_crime" not in df.columns:
        df = add_total_crime_column(df)
    return df


# ---------------------------------------------------------------------------
# 1. Top-N districts per year
# ---------------------------------------------------------------------------

def top_n_districts_per_year(
    df: pd.DataFrame, n: int = 10
) -> dict:
    """
    Top-N districts by total crime for each year.

    Returns
    -------
    dict
        {year: DataFrame with columns [state, district, total_crime, rank_in_year]}
    """
    df = _ensure_total(df)
    result: dict = {}
    for yr, sub in df.groupby("year"):
        ranked = (
            sub.groupby(["state", "district"])["total_crime"]
            .sum()
            .reset_index()
            .sort_values("total_crime", ascending=False)
            .head(n)
            .copy()
        )
        ranked["rank_in_year"] = range(1, len(ranked) + 1)
        ranked["year"] = int(yr)
        result[int(yr)] = ranked[["year", "state", "district", "total_crime", "rank_in_year"]]
    return result


# ---------------------------------------------------------------------------
# 2. Consistency / repeated-hotspot analysis
# ---------------------------------------------------------------------------

def repeated_hotspots(
    df: pd.DataFrame, n: int = 10, years_required: int = 3
) -> pd.DataFrame:
    """
    Districts that repeatedly appear in the yearly top-N.

    Returns columns:
        state, district, appearances, years_list, consistency_pct,
        total_crime_all_years, avg_yearly_crime, max_yearly_crime,
        magnitude_component, consistency_component, hotspot_score.
    """
    df = _ensure_total(df)
    years = sorted(df["year"].unique().tolist())
    total_years = len(years)

    # Appearances per district in yearly top-N.
    appearances: dict = {}
    for yr in years:
        sub = df[df["year"] == yr]
        top_n = (
            sub.groupby(["state", "district"])["total_crime"]
            .sum()
            .nlargest(n)
            .index
        )
        for key in top_n:
            appearances.setdefault(key, []).append(int(yr))

    # District total crime across all years.
    district_totals = district_wise_summary(df)
    total_map = {
        (r["state"], r["district"]): float(r["total_crime"])
        for _, r in district_totals.iterrows()
    }
    max_total = max(total_map.values()) if total_map else 1.0

    rows = []
    for (st, dist), yrs in appearances.items():
        if len(yrs) < years_required:
            continue
        total_all = total_map.get((st, dist), 0.0)
        consistency_pct = (len(yrs) / total_years) * 100
        magnitude_component = (total_all / max_total * 100) if max_total > 0 else 0.0
        consistency_component = consistency_pct
        hotspot_score = 0.6 * magnitude_component + 0.4 * consistency_component

        sub = df[(df["state"] == st) & (df["district"] == dist)]
        yearly = sub.groupby("year")["total_crime"].sum()

        rows.append(
            {
                "state": st,
                "district": dist,
                "appearances": int(len(yrs)),
                "years_list": sorted(yrs),
                "consistency_pct": round(consistency_pct, 1),
                "total_crime_all_years": round(total_all, 2),
                "avg_yearly_crime": round(float(yearly.mean()), 2),
                "max_yearly_crime": float(yearly.max()) if len(yearly) else 0.0,
                "magnitude_component": round(magnitude_component, 2),
                "consistency_component": round(consistency_component, 2),
                "hotspot_score": round(hotspot_score, 2),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["hotspot_score", "appearances", "total_crime_all_years"],
            ascending=[False, False, False],
        )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Simple top-N-by-total hotspot ranking (magnitude-only view)
# ---------------------------------------------------------------------------

def magnitude_hotspots(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N districts by total reported crime across all years (magnitude only)."""
    summary = district_wise_summary(df)
    return summary.head(n).copy()


# ---------------------------------------------------------------------------
# 4. Methodology explanation
# ---------------------------------------------------------------------------

def methodology_text() -> str:
    """Human-readable explanation of the hotspot method (for the dashboard + report)."""
    return (
        "Historical Hotspot Identification Methodology\n"
        "------------------------------------------------\n"
        "Approach: rule-based, explainable scoring (no black-box ML).\n\n"
        "1. MAGNITUDE -- A district's share of the maximum district total crime:\n"
        "      magnitude = (district total crime / highest district total crime) x 100\n\n"
        "2. CONSISTENCY -- How often the district appears in the yearly top-N:\n"
        "      consistency = (years in top-N / total years) x 100\n\n"
        "3. HOTSPOT SCORE =\n"
        "      0.6 x magnitude + 0.4 x consistency   (scale 0-100)\n\n"
        "A district qualifies as a hotspot when it appears in the yearly top-N in\n"
        "at least 3 of the 6 analysed years (configurable).\n\n"
        "Important caveats:\n"
        "  - These are HIGH REPORTED CRIME COUNTS, not crime risk.\n"
        "  - No population data => cannot compute rates per 100,000 people.\n"
        "  - District-level annual data => street/coordinates NOT identifiable.\n"
        "  - Reporting variations (see data_quality_note) can affect counts.\n"
    )


if __name__ == "__main__":
    from .data_loader import load_master_dataset, clean_for_analysis

    df = clean_for_analysis(load_master_dataset())
    print(methodology_text())
    print("\nTop-10 magnitude hotspots (all years):")
    print(magnitude_hotspots(df, 10).to_string(index=False))
    print("\nRepeated hotspots (top-10, >=3 years):")
    print(repeated_hotspots(df, n=10, years_required=3).to_string(index=False))