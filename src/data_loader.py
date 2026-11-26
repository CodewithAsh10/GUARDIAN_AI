"""
Guardian AI - Module 1: Crime Analysis, Intelligence & Interactive Dashboard
============================================================================

src/data_loader.py
------------------
Responsibility: Load and expose the cleaned NCRB district-wise IPC crime dataset
(dataset/processed/guardian_crime_master_2017_2022.csv).

Design principles (academic / production grade):
- The SOURCE CSV is NEVER modified on disk. Every function that transforms data
  returns a NEW DataFrame and documents exactly what changed.
- Metadata columns (year, state, district, source provenance) are separated
  from crime-category columns so downstream analysis never accidentally
  aggregates a provenance field as if it were a crime count.
- State-name normalisation is explicit and documented (see NORMALISATION_MAP).
  Raw, un-normalised names are always available via load_master_dataset().

Expected dataset: 5,322 district-year records, years 2017-2022, 143 crime
category columns (incl. 'Total Cognizable IPC crimes').
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "dataset" / "processed"
MASTER_CSV = DATA_PROCESSED / "guardian_crime_master_2017_2022.csv"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
# Provenance / metadata columns. These are NEVER treated as crime counts.
METADATA_COLUMNS: List[str] = [
    "year",
    "state",
    "district",
    "source_file",
    "source_sheet",
    "source_row_excel",
    "source_serial_no",
    "workbook_title",
    "data_quality_note",  # text annotation added during cleaning (not a crime count)
]

# Traceability columns added during cleaning (e.g. by normalise_state_names).
# These are never crime data and must be excluded from crime-column selection.
TRACEABILITY_COLUMNS: List[str] = ["state_original"]

# The grand-total column. Useful as an "overall crime" measure but it is a
# SUM of categories, not an independent category. Kept separate so callers can
# choose category-level vs total-level analysis explicitly.
TOTAL_CRIME_COLUMN = "Total Cognizable IPC crimes"


def _crime_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of crime-category columns.

    Excludes metadata, traceability, the stored grand-total column, and the
    computed ``total_crime`` helper column (added by add_total_crime_column) so
    helper columns are never mistaken for offence categories.
    """
    excluded = (
        set(METADATA_COLUMNS)
        | set(TRACEABILITY_COLUMNS)
        | {TOTAL_CRIME_COLUMN, "total_crime"}
    )
    return [c for c in df.columns if c not in excluded]


def master_columns() -> Tuple[List[str], List[str], str]:
    """Convenience: (metadata_columns, crime_columns, total_column)."""
    return list(METADATA_COLUMNS), _crime_columns(pd.read_csv(MASTER_CSV, nrows=0)), TOTAL_CRIME_COLUMN


# ---------------------------------------------------------------------------
# State-name normalisation
# ---------------------------------------------------------------------------
# The NCRB workbooks label the same state/UT slightly differently across years.
# Left side = value as it appears in the source CSV; right side = canonical name.
# Every mapping below is documented with the evidence in the module docstring
# comments and reflected in the validation report.
#
#   - 'Harayana' (2017 only) is a typo for 'Haryana' (2018-2022).
#   - 'Delhi UT' (2017-2019) and 'Delhi' (2020-2022) are the same UT.
#   - 'D&N Haveli' / 'Daman & Diu' (2017-2019) merged into
#     'D&N Haveli and Daman & Diu' from 2020 onwards.
#   - 'Nagaland #' (2022) is 'Nagaland' with a workbook footnote marker.
NORMALISATION_MAP: dict = {
    "Harayana": "Haryana",
    "Delhi UT": "Delhi",
    "D&N Haveli": "D&N Haveli and Daman & Diu",
    "Daman & Diu": "D&N Haveli and Daman & Diu",
    "Nagaland #": "Nagaland",
    # GeoJSON state name normalisations (for matching with boundary files)
    "Andaman & Nicobar Island": "A&N Islands",
    "Arunanchal Pradesh": "Arunachal Pradesh",
    "Dadra & Nagar Havelli and Daman & Diu": "D&N Haveli and Daman & Diu",
}


def load_master_dataset() -> pd.DataFrame:
    """
    Load the cleaned master CSV exactly as stored on disk.

    Returns
    -------
    pd.DataFrame
        Raw dataset with original column names and original (un-normalised)
        state/district names. No transformation is applied.

    Notes
    -----
    The source file on disk is NOT modified. Use :func:`normalise_state_names`
    or :func:`clean_for_analysis` when a transformed copy is needed.
    """
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {MASTER_CSV}. "
            "Ensure dataset/processed/guardian_crime_master_2017_2022.csv exists."
        )
    df = pd.read_csv(MASTER_CSV)
    return df


def get_metadata_columns() -> List[str]:
    """Return a copy of the metadata column list."""
    return list(METADATA_COLUMNS)


def get_crime_columns(df: pd.DataFrame | None = None) -> List[str]:
    """
    Return crime-category column names.

    Parameters
    ----------
    df : pd.DataFrame, optional
        If provided, columns are derived from this frame. Otherwise the master
        CSV header is inspected (0 rows read).

    Returns
    -------
    List[str]
        Crime-category columns (metadata and the grand-total column excluded).
    """
    if df is None:
        df = pd.read_csv(MASTER_CSV, nrows=0)
    return _crime_columns(df)


def get_total_crime_column() -> str:
    """Return the name of the grand-total crime column."""
    return TOTAL_CRIME_COLUMN


def normalise_state_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``df`` with state names normalised to canonical forms.

    The original frame is not modified. Rows whose state name is not in
    :data:`NORMALISATION_MAP` are left unchanged. A new column
    ``state_original`` is added so the raw name is never lost.

    Returns
    -------
    pd.DataFrame
        Copy with normalised ``state`` column and a ``state_original`` column.
    """
    out = df.copy()
    if "state_original" not in out.columns:
        out["state_original"] = out["state"].astype(str)
    out["state"] = out["state"].replace(NORMALISATION_MAP)
    return out


def clean_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return an analysis-ready copy of the dataset.

    Transformations (all documented, none applied to the source file):
      1. Normalise state names (see NORMALISATION_MAP).
      2. Coerce crime-category columns to numeric; non-numeric cells become NaN
         rather than being silently dropped or guessed.
      3. Fill missing numeric crime values with 0. Rationale: NCRB reports a
         blank cell as "no cases reported under this head", which is
         semantically zero for a count column. This is recorded in the returned
         attribute ``missing_filled_columns``.
      4. Ensure ``year`` is int, ``state``/``district`` are str.

    Returns
    -------
    pd.DataFrame
        Cleaned copy. Carries attribute ``missing_filled_columns`` listing
        columns where NaN was filled with 0, and ``state_changes`` listing
        original->canonical state mappings that were applied.
    """
    # Identify crime columns on the ORIGINAL frame (before state_original is added)
    # so the provenance column is never treated as crime data.
    crime_cols = get_crime_columns(df)

    out = normalise_state_names(df)

    # Protect state_original from numeric coercion: it is traceability metadata,
    # not a crime count.
    crime_cols = [c for c in crime_cols if c in out.columns and c != "state_original"]

    # Record state normalisations actually applied.
    state_changes: dict = {}
    if "state_original" in out.columns:
        for orig, canon in NORMALISATION_MAP.items():
            applied = (out["state_original"] == orig).any()
            if applied:
                state_changes[orig] = canon

    # Coerce to numeric (handles the float64 columns with NaN and any stray text).
    for col in crime_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Track which columns had NaN before filling.
    missing_filled = {
        col: int(out[col].isna().sum()) for col in crime_cols if out[col].isna().any()
    }

    # Semantic rule: blank NCRB cell = 0 reported cases.
    out[crime_cols] = out[crime_cols].fillna(0)

    # Type hygiene.
    out["year"] = out["year"].astype(int)
    out["state"] = out["state"].astype(str)
    out["district"] = out["district"].astype(str)

    # Stamp the documentation attributes on the frame for downstream inspectability.
    out.attrs["missing_filled_columns"] = missing_filled
    out.attrs["state_changes"] = state_changes
    out.attrs["source_file"] = str(MASTER_CSV)

    return out


def add_total_crime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``total_crime`` column = sum of all crime-category columns for each row.

    Uses the grand-total column when present and reliable; otherwise computes
    the row-wise sum of category columns. Returns a copy.
    """
    out = df.copy()
    crime_cols = get_crime_columns(out)
    if TOTAL_CRIME_COLUMN in out.columns and out[TOTAL_CRIME_COLUMN].notna().all():
        out["total_crime"] = out[TOTAL_CRIME_COLUMN]
    else:
        out["total_crime"] = out[crime_cols].sum(axis=1)
    return out