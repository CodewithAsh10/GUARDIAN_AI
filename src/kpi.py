"""Shared KPI card component for dashboard pages."""
from __future__ import annotations

import streamlit as st


def kpi_card(label: str, value: str, sub: str = "") -> None:
    """Render a KPI card in the Guardian AI dashboard style."""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )