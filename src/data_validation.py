"""
Guardian AI - Module 1
src/data_validation.py
----------------------
Responsibility: Validate the cleaned NCRB dataset and produce a human-readable
validation summary. Every check is read-only (the source CSV is never modified).

Checks performed:
  1. Dimensions (rows, columns)
  2. Column names & data types
  3. Missing values (per column)
  4. Duplicate records (full row and by year-state-district key)
  5. Unique states / districts / years
  6. Records per year
  7. State-name normalisation candidates (variants of the same entity)
  8. Grand-total column consistency vs sum of categories (sampled)
  9. Negative-value scan (crime counts cannot be negative)

Output: a dictionary report that can be printed, logged, or written to a file.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .data_loader import (
    METADATA_COLUMNS,
    TOTAL_CRIME_COLUMN,
    _crime_columns,
    clean_for_analysis,
    load_master_dataset,
    normalise_state_names,
)


def validate_dataset(df: pd.DataFrame | None = None) -> Dict:
    """
    Run the full validation suite on the dataset.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Frame to validate. If None, the master CSV is loaded (raw, unmodified).

    Returns
    -------
    Dict
        Validation report with keys: dataset, columns, missing, duplicates,
        uniqueness, yearly, state_variants, total_consistency, negatives,
        passed (bool), messages (list of human-readable strings).
    """
    if df is None:
        df = load_master_dataset()

    messages: List[str] = []
    passed = True

    report: Dict = {}

    # --- 1. Dimensions -------------------------------------------------------
    n_rows, n_cols = df.shape
    report["dataset"] = {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }
    messages.append(f"Dimensions: {n_rows} rows x {n_cols} columns.")

    # --- 2. Columns & dtypes -------------------------------------------------
    meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    crime_cols = _crime_columns(df)
    report["columns"] = {
        "metadata": meta_cols,
        "crime_category_count": len(crime_cols),
        "crime_columns": crime_cols,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    messages.append(
        f"Columns: {len(meta_cols)} metadata + {len(crime_cols)} crime-category "
        f"columns (+ 1 grand-total column '{TOTAL_CRIME_COLUMN}')."
    )
    # Previously reported: 143 crime categories. That figure counts the
    # grand-total column; the actual independent crime categories here are 142
    # (142 categories + 1 total column = 143 crime-related columns).
    crime_related = len(crime_cols) + (1 if TOTAL_CRIME_COLUMN in df.columns else 0)
    if crime_related != 143:
        passed = False
        messages.append(
            f"WARNING: expected ~143 crime-related columns, found {crime_related}."
        )
    else:
        messages.append(
            f"Crime-related column count matches expectation ({crime_related} = "
            f"{len(crime_cols)} categories + 1 total column)."
        )

    # --- 3. Missing values ---------------------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    report["missing"] = {c: int(v) for c, v in missing.items()}
    report["missing_total_cells"] = int(df.isna().sum().sum())
    if len(missing) == 0:
        messages.append("Missing values: none detected.")
    else:
        messages.append(
            f"Missing values detected in {len(missing)} column(s): "
            + ", ".join(f"{c} ({int(v)})" for c, v in missing.items())
        )

    # --- 4. Duplicates -------------------------------------------------------
    dup_full = int(df.duplicated().sum())
    key_cols = [c for c in ["year", "state", "district"] if c in df.columns]
    dup_key = int(df.duplicated(subset=key_cols).sum()) if len(key_cols) == 3 else None
    report["duplicates"] = {
        "full_row": dup_full,
        "year_state_district": dup_key,
    }
    messages.append(f"Duplicate full rows: {dup_full}.")
    if dup_key is not None:
        messages.append(f"Duplicate (year, state, district) records: {dup_key}.")
    if dup_full > 0 or (dup_key is not None and dup_key > 0):
        passed = False
        messages.append("WARNING: duplicate records found.")

    # --- 5. Uniqueness -------------------------------------------------------
    uniq_states = int(df["state"].nunique()) if "state" in df.columns else None
    uniq_districts = int(df["district"].nunique()) if "district" in df.columns else None
    uniq_years = sorted(df["year"].unique().tolist()) if "year" in df.columns else []
    report["uniqueness"] = {
        "unique_states": uniq_states,
        "unique_districts": uniq_districts,
        "unique_state_district_pairs": int(df.groupby(["state", "district"]).ngroups)
        if {"state", "district"}.issubset(df.columns)
        else None,
        "years": uniq_years,
    }
    messages.append(
        f"Unique states: {uniq_states}; unique districts: {uniq_districts}; "
        f"years: {uniq_years}."
    )

    # --- 6. Records per year -------------------------------------------------
    if "year" in df.columns:
        yearly = df["year"].value_counts().sort_index().to_dict()
        report["yearly"] = {int(k): int(v) for k, v in yearly.items()}
        messages.append("Records per year: " + ", ".join(f"{k}={v}" for k, v in yearly.items()))

    # --- 7. State-name variants (normalisation candidates) -------------------
    variants: Dict[str, List[str]] = {}
    if "state" in df.columns:
        # Group states that are likely the same entity but spelled differently.
        # Detected by inspecting the raw data; kept explicit for transparency.
        candidate_groups = [
            ("Haryana", ["Harayana", "Haryana"]),
            ("Delhi", ["Delhi", "Delhi UT"]),
            ("D&N Haveli and Daman & Diu", ["D&N Haveli", "D&N Haveli and Daman & Diu", "Daman & Diu"]),
            ("Nagaland", ["Nagaland", "Nagaland #"]),
        ]
        for canon, variants_list in candidate_groups:
            present = [v for v in variants_list if v in set(df["state"].astype(str))]
            if len(present) > 1:
                variants[canon] = present
                messages.append(
                    f"State variant group -> {canon}: {present} (should be normalised)."
                )
    report["state_variants"] = variants

    # --- 8. Grand-total consistency (sampled) --------------------------------
    total_ok = None
    total_message = ""
    if TOTAL_CRIME_COLUMN in df.columns and len(crime_cols) > 0:
        # Compare the stored total to the sum of categories on a sample.
        sample = df.head(100)
        computed = sample[_crime_columns(sample)].sum(axis=1)
        stored = sample[TOTAL_CRIME_COLUMN]
        # Allow small rounding / reporting differences; flag large mismatches.
        mismatches = int((abs(computed - stored) > 5).sum())
        total_ok = mismatches == 0
        total_message = (
            f"Grand-total vs category-sum mismatches (>5) in sample of 100: {mismatches}."
        )
        messages.append(total_message)
    report["total_consistency"] = {"sample_mismatches_gt_5": total_ok}

    # --- 9. Negative values --------------------------------------------------
    neg_report: Dict[str, int] = {}
    if len(crime_cols) > 0:
        for col in crime_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                n_neg = int((df[col] < 0).sum())
                if n_neg > 0:
                    neg_report[col] = n_neg
    report["negatives"] = neg_report
    if len(neg_report) == 0:
        messages.append("Negative values: none (all crime counts >= 0).")
    else:
        passed = False
        messages.append(
            "WARNING: negative crime counts found in " + ", ".join(neg_report.keys())
        )

    report["passed"] = passed
    report["messages"] = messages
    return report


def validation_summary_text(report: Dict) -> str:
    """Render the validation report as a readable plain-text summary."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("GUARDIAN AI - DATASET VALIDATION SUMMARY")
    lines.append("=" * 70)
    ds = report.get("dataset", {})
    lines.append(f"Rows: {ds.get('rows')} | Columns: {ds.get('columns')}")
    lines.append(f"Memory usage: {ds.get('memory_usage_mb')} MB")
    lines.append("-" * 70)
    lines.append("Column groups:")
    col = report.get("columns", {})
    lines.append(f"  Metadata columns: {len(col.get('metadata', []))}")
    lines.append(f"  Crime-category columns: {col.get('crime_category_count')}")
    lines.append("-" * 70)
    lines.append("Missing values:")
    if report.get("missing"):
        for c, v in report["missing"].items():
            lines.append(f"  {c}: {v}")
    else:
        lines.append("  None")
    lines.append(f"  Total missing cells: {report.get('missing_total_cells', 0)}")
    lines.append("-" * 70)
    lines.append("Duplicates:")
    dup = report.get("duplicates", {})
    lines.append(f"  Full-row duplicates: {dup.get('full_row')}")
    lines.append(f"  (year, state, district) duplicates: {dup.get('year_state_district')}")
    lines.append("-" * 70)
    lines.append("Uniqueness:")
    uniq = report.get("uniqueness", {})
    lines.append(f"  Unique states: {uniq.get('unique_states')}")
    lines.append(f"  Unique districts: {uniq.get('unique_districts')}")
    lines.append(f"  Unique state-district pairs: {uniq.get('unique_state_district_pairs')}")
    lines.append(f"  Years: {uniq.get('years')}")
    lines.append("-" * 70)
    lines.append("Records per year:")
    for k, v in report.get("yearly", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("-" * 70)
    lines.append("State normalisation candidates:")
    if report.get("state_variants"):
        for canon, variants in report["state_variants"].items():
            lines.append(f"  -> {canon}: {variants}")
    else:
        lines.append("  None detected")
    lines.append("-" * 70)
    lines.append("Negative values:")
    if report.get("negatives"):
        for c, v in report["negatives"].items():
            lines.append(f"  {c}: {v}")
    else:
        lines.append("  None")
    lines.append("-" * 70)
    lines.append(f"Overall validation PASSED: {report.get('passed')}")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    report = validate_dataset()
    print(validation_summary_text(report))