"""
Guardian AI - Module 1
src/visualizations.py
---------------------
Responsibility: Create professional, interactive Plotly visualisations for the
crime analysis, plus static image exports saved to visualisations/.

Theme
-----
Dark navy background with professional blue accents, matching the Streamlit
dashboard design. All figures use a consistent template so the notebook and the
dashboard look cohesive.

Every function returns a plotly.graph_objects.Figure (or Express figure) and,
when ``save=True``, writes a static PNG to visualisations/. Figures are
read-only to build: they never modify the input DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .data_loader import TOTAL_CRIME_COLUMN, _crime_columns
from .analysis import (
    year_wise_summary,
    year_wise_change,
    state_wise_summary,
    district_wise_summary,
    top_districts,
    top_categories,
    category_summary,
    category_trend_all,
    add_total_crime_column,
)
from .hotspot_detection import repeated_hotspots

# ---------------------------------------------------------------------------
# Output folder for static visualisations
# ---------------------------------------------------------------------------
VIS_DIR = Path(__file__).resolve().parents[1] / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Theme constants (dark navy + professional blue accents)
# ---------------------------------------------------------------------------
NAVY = "#0E1A2B"
NAVY_SOFT = "#14253D"
BLUE = "#2E86C7"
BLUE_LIGHT = "#85C1E9"
ACCENT = "#3498DB"
GREY_TEXT = "#AAB8C4"
GRID_COLOR = "#2C3E50"
DANGER = "#E74C3C"
WARNING = "#F39C12"
SUCCESS = "#27AE60"

CATEGORY_COLORS = px.colors.qualitative.Safe

FIG_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=NAVY,
    plot_bgcolor=NAVY_SOFT,
    font=dict(family="Segoe UI, Arial, sans-serif", color=GREY_TEXT, size=12),
    title=dict(font=dict(size=16, color="#FFFFFF", family="Segoe UI, Arial, sans-serif")),
    legend=dict(font=dict(size=11, color=GREY_TEXT), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=60, r=40, t=80, b=60),
    xaxis=dict(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        title_font=dict(size=12, color=GREY_TEXT),
        tickfont=dict(size=11, color=GREY_TEXT),
    ),
    yaxis=dict(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        title_font=dict(size=12, color=GREY_TEXT),
        tickfont=dict(size=11, color=GREY_TEXT),
    ),
)


def _style(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    """Apply the standard theme."""
    layout = dict(FIG_LAYOUT)
    layout.pop("title", None)  # title is set explicitly below to avoid duplicate kwarg
    fig.update_layout(
        **layout,
        title=dict(text=title, x=0.02, xanchor="left"),
        height=height,
    )
    return fig


def _save(fig: go.Figure, filename: str) -> None:
    """Save a static PNG to visualisations/ (kaleido or to_image fallback)."""
    try:
        fig.write_image(VIS_DIR / filename, scale=2)
    except Exception as exc:  # pragma: no cover - depends on kaleido install
        print(f"[visualizations] Could not save static image '{filename}': {exc}")


# ===========================================================================
# 1. Overall year-wise crime trend
# ===========================================================================

def plot_yearly_trend(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Line chart of total reported crime per year (2017-2022)."""
    summary = year_wise_summary(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary["year"],
            y=summary["total_crime"],
            mode="lines+markers",
            name="Total reported crime",
            line=dict(color=BLUE, width=3),
            marker=dict(size=10, color=ACCENT, line=dict(width=2, color=NAVY)),
            hovertemplate="Year: %{x}<br>Total crime: %{y:,.0f}<extra></extra>",
        )
    )
    fig = _style(
        fig,
        "Overall Reported Crime Trend (2017-2022)",
        height=500,
    )
    fig.update_xaxes(title_text="Year", tickmode="linear")
    fig.update_yaxes(title_text="Total reported crime count")
    if save:
        _save(fig, "yearly_crime_trend.png")
    return fig


def plot_yearly_change(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Bar chart of year-over-year percentage change in total crime."""
    change = year_wise_change(df).dropna(subset=["pct_change"])
    colors = [SUCCESS if v >= 0 else DANGER for v in change["pct_change"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=change["year"].astype(str),
            y=change["pct_change"],
            marker_color=colors,
            hovertemplate="Year: %{x}<br>Change: %{y:.2f}%<extra></extra>",
        )
    )
    fig = _style(
        fig,
        "Year-over-Year Percentage Change in Total Reported Crime",
        height=500,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Percentage change (%)", zeroline=True, zerolinecolor=GRID_COLOR)
    fig.add_hline(y=0, line_width=1, line_color=GREY_TEXT)
    if save:
        _save(fig, "yearly_pct_change.png")
    return fig


# ===========================================================================
# 2. State-wise comparison
# ===========================================================================

def plot_state_comparison(df: pd.DataFrame, n: int = 10, save: bool = False) -> go.Figure:
    """Horizontal bar chart of top-N states by total reported crime."""
    summary = state_wise_summary(df).head(n).iloc[::-1]  # reverse for barh
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["total_crime"],
            y=summary["state"],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="State: %{y}<br>Total crime: %{x:,.0f}<extra></extra>",
        )
    )
    fig = _style(
        fig,
        f"Top {n} States by Total Reported Crime (2017-2022)",
        height=500,
    )
    fig.update_xaxes(title_text="Total reported crime count")
    fig.update_yaxes(title_text="State", automargin=True)
    if save:
        _save(fig, "top_states.png")
    return fig


# ===========================================================================
# 3. Top districts
# ===========================================================================

def plot_top_districts(df: pd.DataFrame, n: int = 10, save: bool = False) -> go.Figure:
    """Horizontal bar chart of top-N districts by total reported crime."""
    summary = top_districts(df, n).iloc[::-1]
    labels = [f"{s} - {d}" for s, d in zip(summary["state"], summary["district"])]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["total_crime"],
            y=labels,
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="District: %{y}<br>Total crime: %{x:,.0f}<extra></extra>",
        )
    )
    fig = _style(
        fig,
        f"Top {n} Districts by Total Reported Crime (2017-2022)",
        height=520,
    )
    fig.update_xaxes(title_text="Total reported crime count")
    fig.update_yaxes(title_text="District (State)", automargin=True)
    if save:
        _save(fig, "top_districts.png")
    return fig


# ===========================================================================
# 4. Crime category distribution
# ===========================================================================

def plot_category_distribution(df: pd.DataFrame, n: int = 10, save: bool = False) -> go.Figure:
    """Bar chart of top-N crime categories by total reported crime."""
    cats = top_categories(df, n).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats["total_crime"],
            y=cats["category"],
            orientation="h",
            marker_color=CATEGORY_COLORS[: len(cats)],
            hovertemplate="Category: %{y}<br>Total crime: %{x:,.0f}<extra></extra>",
        )
    )
    fig = _style(
        fig,
        f"Top {n} Crime Categories by Total Reported Crime (2017-2022)",
        height=520,
    )
    fig.update_xaxes(title_text="Total reported crime count")
    fig.update_yaxes(title_text="Crime category", automargin=True)
    if save:
        _save(fig, "top_categories.png")
    return fig


def plot_category_pie(df: pd.DataFrame, n: int = 8, save: bool = False) -> go.Figure:
    """Pie chart showing the share of the top-N categories vs all others."""
    summary = category_summary(df)
    top_n = summary.head(n)
    others_total = float(summary.iloc[n:]["total_crime"].sum()) if len(summary) > n else 0.0
    labels = list(top_n["category"]) + (["All other categories"] if others_total > 0 else [])
    values = list(top_n["total_crime"]) + ([others_total] if others_total > 0 else [])
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=CATEGORY_COLORS, line=dict(color=NAVY, width=2)),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
            )
        ]
    )
    fig = _style(
        fig,
        f"Distribution of Reported Crime by Category (Top {n})",
        height=500,
    )
    if save:
        _save(fig, "category_distribution_pie.png")
    return fig


# ===========================================================================
# 5. Category-wise yearly trend
# ===========================================================================

def plot_category_yearly_trend(
    df: pd.DataFrame, categories: Optional[List[str]] = None, save: bool = False
) -> go.Figure:
    """Multi-line chart of yearly totals for selected crime categories."""
    trend = category_trend_all(df)
    if categories is None:
        # default to top 5 categories
        categories = list(top_categories(df, 5)["category"])
    categories = [c for c in categories if c in trend.columns]
    if not categories:
        categories = [TOTAL_CRIME_COLUMN] if TOTAL_CRIME_COLUMN in df.columns else []

    fig = go.Figure()
    palette = CATEGORY_COLORS
    for i, cat in enumerate(categories):
        fig.add_trace(
            go.Scatter(
                x=trend.index.astype(str),
                y=trend[cat],
                mode="lines+markers",
                name=cat,
                line=dict(width=2, color=palette[i % len(palette)]),
                marker=dict(size=7),
                hovertemplate="Year: %{x}<br>" + cat + ": %{y:,.0f}<extra></extra>",
            )
        )
    title = "Year-wise Trend"
    if categories:
        title += " - " + ", ".join(categories[:3]) + (
            f" (+{len(categories)-3} more)" if len(categories) > 3 else ""
        )
    fig = _style(fig, title, height=500)
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Reported crime count")
    if save:
        _save(fig, "category_yearly_trend.png")
    return fig


# ===========================================================================
# 6. District hotspot comparison
# ===========================================================================

def plot_hotspot_comparison(df: pd.DataFrame, n: int = 10, save: bool = False) -> go.Figure:
    """Bar chart of hotspot scores for repeated high-crime districts."""
    hs = repeated_hotspots(df, n=n, years_required=3).head(n).iloc[::-1]
    if hs.empty:
        # Fallback: magnitude-based top districts
        hs = top_districts(df, n).head(n).iloc[::-1]
        hs["hotspot_score"] = hs["total_crime"]
        score_col = "total_crime"
        title = f"Top {n} Districts by Total Crime (fallback: no repeated hotspots)"
    else:
        score_col = "hotspot_score"
        title = f"Top {n} Historical Crime Hotspots (repeat + magnitude score)"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hs[score_col],
            y=[f"{s} - {d}" for s, d in zip(hs["state"], hs["district"])],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="District: %{y}<br>Score: %{x:.1f}<extra></extra>",
        )
    )
    fig = _style(fig, title, height=520)
    fig.update_xaxes(title_text="Hotspot score (0-100)" if score_col == "hotspot_score" else "Total crime")
    fig.update_yaxes(title_text="District (State)", automargin=True)
    if save:
        _save(fig, "hotspot_comparison.png")
    return fig


# ===========================================================================
# 7. State-wise trend (multi-line for selected states)
# ===========================================================================

def plot_state_trend(df: pd.DataFrame, states: Optional[List[str]] = None, save: bool = False) -> go.Figure:
    """Multi-line chart of yearly total crime for selected states."""
    df = add_total_crime_column(df)
    if states is None:
        states = list(state_wise_summary(df).head(6)["state"])
    states = [s for s in states if s in df["state"].unique()]
    fig = go.Figure()
    palette = CATEGORY_COLORS
    for i, st in enumerate(states):
        sub = df[df["state"] == st].groupby("year")["total_crime"].sum()
        fig.add_trace(
            go.Scatter(
                x=sub.index.astype(str),
                y=sub.values,
                mode="lines+markers",
                name=st,
                line=dict(width=2, color=palette[i % len(palette)]),
                marker=dict(size=7),
                hovertemplate="Year: %{x}<br>" + st + ": %{y:,.0f}<extra></extra>",
            )
        )
    fig = _style(
        fig,
        "Year-wise Reported Crime Trend by State",
        height=500,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Total reported crime count")
    if save:
        _save(fig, "state_yearly_trend.png")
    return fig