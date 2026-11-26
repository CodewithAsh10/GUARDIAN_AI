"""
Guardian AI - Module 1
src/analysis.py
---------------
Responsibility: Core analytical aggregations over the cleaned NCRB dataset.

All functions are read-only and return new DataFrames/Series. They accept an
already-cleaned DataFrame (see src.data_loader.clean_for_analysis) so the same
logic can be reused by the notebook, the Streamlit dashboard, and the tests.

Key analytical distinction maintained throughout:
  - TOTAL crime = sum of all IPC crime heads (grand total).
  - CATEGORY crime = a specific offence head (e.g. Murder, Theft).
  These are never conflated.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .data_loader import TOTAL_CRIME_COLUMN, _crime_columns, add_total_crime_column


def _ensure_total(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a ``total_crime`` column (computed if missing)."""
    if "total_crime" not in df.columns:
        df = add_total_crime_column(df)
    return df


# ===========================================================================
# A. Overall / year-wise crime trends
# ===========================================================================

def year_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total reported crime and record counts per year.

    Returns columns: year, total_crime, record_count, avg_crime_per_district.
    Sorted by year ascending.
    """
    df = _ensure_total(df)
    g = df.groupby("year").agg(
        total_crime=("total_crime", "sum"),
        record_count=("total_crime", "size"),
        avg_crime_per_district=("total_crime", "mean"),
    ).reset_index().sort_values("year")
    return g


def year_wise_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Absolute and percentage change in total crime year-over-year.

    Returns columns: year, total_crime, abs_change, pct_change.
    The first year has NaN changes.
    """
    summary = year_wise_summary(df)[["year", "total_crime"]].copy()
    summary = summary.sort_values("year").reset_index(drop=True)
    summary["abs_change"] = summary["total_crime"].diff()
    summary["pct_change"] = summary["total_crime"].pct_change() * 100
    return summary


def category_year_wise(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Total of a single crime category per year.

    Returns columns: year, <category>.
    """
    if category not in df.columns:
        raise KeyError(f"Category '{category}' not found in dataset.")
    g = df.groupby("year")[category].sum().reset_index().sort_values("year")
    return g


# ===========================================================================
# B. State-wise analysis
# ===========================================================================

def state_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total reported crime per state across all years.

    Returns columns: state, total_crime, record_count, years_present,
    avg_crime_per_year. Sorted by total_crime descending.
    """
    df = _ensure_total(df)
    g = (
        df.groupby("state")
        .agg(
            total_crime=("total_crime", "sum"),
            record_count=("total_crime", "size"),
            years_present=("year", "nunique"),
            avg_crime_per_year=("total_crime", "mean"),
        )
        .reset_index()
        .sort_values("total_crime", ascending=False)
    )
    return g


def state_year_wise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total reported crime per state per year.

    Returns columns: state, year, total_crime.
    """
    df = _ensure_total(df)
    g = (
        df.groupby(["state", "year"])["total_crime"]
        .sum()
        .reset_index()
        .sort_values(["state", "year"])
    )
    return g


def state_category_comparison(df: pd.DataFrame, states: List[str], category: str) -> pd.DataFrame:
    """
    Total of a given category for selected states across all years.

    Returns columns: state, <category> (total).
    """
    if category not in df.columns:
        raise KeyError(f"Category '{category}' not found in dataset.")
    sub = df[df["state"].isin(states)]
    g = sub.groupby("state")[category].sum().reset_index().sort_values(category, ascending=False)
    return g


def top_states(df: pd.DataFrame, n: int = 10, measure: str = "total_crime") -> pd.DataFrame:
    """Top-N states by a crime measure (default: total_crime)."""
    summary = state_wise_summary(df)
    return summary.head(n)


def bottom_states(df: pd.DataFrame, n: int = 10, measure: str = "total_crime") -> pd.DataFrame:
    """Bottom-N states by a crime measure."""
    summary = state_wise_summary(df)
    return summary.sort_values(measure, ascending=True).head(n)


# ===========================================================================
# C. District-wise analysis
# ===========================================================================

def district_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total reported crime per district (state + district) across all years.

    Returns columns: state, district, total_crime, record_count, years_present.
    Sorted by total_crime descending.
    """
    df = _ensure_total(df)
    g = (
        df.groupby(["state", "district"])
        .agg(
            total_crime=("total_crime", "sum"),
            record_count=("total_crime", "size"),
            years_present=("year", "nunique"),
        )
        .reset_index()
        .sort_values("total_crime", ascending=False)
    )
    return g


def top_districts(df: pd.DataFrame, n: int = 10, state: str | None = None) -> pd.DataFrame:
    """Top-N districts by total crime. Optionally filter to one state."""
    summary = district_wise_summary(df)
    if state is not None:
        summary = summary[summary["state"] == state]
    return summary.head(n)


def district_trend(df: pd.DataFrame, state: str, district: str) -> pd.DataFrame:
    """
    Total crime per year for a specific district.

    Returns columns: year, total_crime.
    """
    df = _ensure_total(df)
    sub = df[(df["state"] == state) & (df["district"] == district)]
    g = sub.groupby("year")["total_crime"].sum().reset_index().sort_values("year")
    return g


def repeated_high_crime_districts(
    df: pd.DataFrame, n: int = 10, years_required: int = 3
) -> pd.DataFrame:
    """
    Districts that repeatedly appear among the top-N by total crime in multiple
    years. Used for explainable hotspot identification (see hotspot_detection).

    Returns columns: state, district, appearances (number of years in top-N),
    avg_yearly_crime, max_yearly_crime, years_list.
    """
    df = _ensure_total(df)
    yearly_top: dict = {}
    for yr, sub in df.groupby("year"):
        top_n = sub.groupby(["state", "district"])["total_crime"].sum().nlargest(n)
        for (st, dist) in top_n.index:
            yearly_top.setdefault((st, dist), []).append(int(yr))

    rows = []
    for (st, dist), years in yearly_top.items():
        if len(years) >= years_required:
            sub = df[(df["state"] == st) & (df["district"] == dist)]
            yearly_crime = sub.groupby("year")["total_crime"].sum()
            rows.append(
                {
                    "state": st,
                    "district": dist,
                    "appearances": len(years),
                    "years_list": sorted(years),
                    "avg_yearly_crime": float(yearly_crime.mean()),
                    "max_yearly_crime": float(yearly_crime.max()),
                    "total_crime_all_years": float(sub["total_crime"].sum()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["appearances", "total_crime_all_years"], ascending=[False, False])
    return out.reset_index(drop=True)


# ===========================================================================
# D. Crime-category analysis
# ===========================================================================

def category_summary(df: pd.DataFrame, include_total: bool = False) -> pd.DataFrame:
    """
    Total per crime category across all years and districts.

    Returns columns: category, total_crime, record_count, avg_per_record,
    pct_of_total. Sorted by total_crime descending.
    """
    crime_cols = _crime_columns(df)
    if include_total and TOTAL_CRIME_COLUMN in df.columns:
        crime_cols = crime_cols + [TOTAL_CRIME_COLUMN]

    totals = df[crime_cols].sum()
    records = (df[crime_cols] > 0).sum()
    grand_total = float(totals.sum())

    out = pd.DataFrame(
        {
            "category": totals.index,
            "total_crime": totals.values,
            "record_count": records.values,
        }
    )
    out["avg_per_record"] = out["total_crime"] / out["record_count"].replace(0, pd.NA)
    out["pct_of_total"] = (out["total_crime"] / grand_total * 100) if grand_total > 0 else 0.0
    out = out.sort_values("total_crime", ascending=False).reset_index(drop=True)
    return out


def top_categories(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N crime categories by total reported crime."""
    return category_summary(df).head(n)


def category_trend_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total per category per year (wide format: index=year, columns=categories).

    Useful for category-wise yearly trend visualisations.
    """
    crime_cols = _crime_columns(df)
    g = df.groupby("year")[crime_cols].sum()
    return g.sort_index()


def trend_direction(df: pd.DataFrame, category: str) -> str:
    """
    Simple trend direction for a category: compare first vs last year total.

    Returns 'Increasing', 'Decreasing', or 'Stable'.
    """
    cy = category_year_wise(df, category)
    if len(cy) < 2:
        return "Insufficient data"
    first = float(cy.iloc[0][category])
    last = float(cy.iloc[-1][category])
    if last > first * 1.05:
        return "Increasing"
    if last < first * 0.95:
        return "Decreasing"
    return "Stable"


# ===========================================================================
# E. KPI-style helpers (used by the dashboard Overview page)
# ===========================================================================

def kpis_for_year(df: pd.DataFrame, year: int) -> dict:
    """Key performance indicators for a selected year."""
    df = _ensure_total(df)
    sub = df[df["year"] == year]
    if sub.empty:
        return {
            "year": year,
            "total_crime": 0,
            "states": 0,
            "districts": 0,
            "categories": 0,
            "record_count": 0,
        }
    return {
        "year": int(year),
        "total_crime": int(sub["total_crime"].sum()),
        "states": int(sub["state"].nunique()),
        "districts": int(sub["district"].nunique()),
        "categories": len(_crime_columns(df)),
        "record_count": int(len(sub)),
    }