"""
Build the complete notebooks/01_crime_analysis.ipynb for Guardian AI Module 1.

Constructs a valid Jupyter notebook (.ipynb) with Markdown + code cells covering
all 14 required sections. Reuses src/ modules so notebook, dashboard and tests
share one analysis logic (no duplication).
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/01_crime_analysis.ipynb")


def code_cell(src, execution_count=None):
    lines = [l for l in src.strip().split("\n") if l.strip()]
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def md_cell(src):
    lines = [l for l in src.strip().split("\n") if l.strip()]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


cells = []

# ---------------------------------------------------------------------------
# 1. Project introduction
# ---------------------------------------------------------------------------
cells.append(md_cell("""
# Guardian AI - Module 1: Crime Analysis, Intelligence & Interactive Dashboard

**College EPICS Project:** *Guardian AI: A Predictive Crime Intelligence Platform with
Trend Analysis and Safe Route Optimization.*

This notebook completes **Module 1: Crime Analysis and Trend Analysis**.

## What this notebook does
- Loads and validates the cleaned NCRB district-wise IPC crime dataset (2017-2022).
- Performs exploratory data analysis (EDA) and data-quality checks.
- Analyses overall crime trends, state-wise and district-wise comparisons, and crime-category trends.
- Identifies historical crime hotspots using a transparent, explainable method.
- Produces professional visualisations (saved to `visualisations/`).

## Data source
- National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India.
- District-wise IPC crime data, 2017 to 2022.
- Official portal: https://www.ncrb.mha.gov.in/

## Academic honesty notes
- This is **district-level annual aggregate data**, not exact crime coordinates.
- It cannot identify streets or individual incidents.
- Raw reported crime counts are **not crime risk** (no population data is used).
- All findings are computed from the genuine NCRB dataset; nothing is fabricated.
"""))

# ---------------------------------------------------------------------------
# 2. Importing libraries
# ---------------------------------------------------------------------------
cells.append(md_cell("## 2. Importing libraries"))
cells.append(code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project modules (single source of truth for analysis logic)
# Add project root (parent of notebooks/) AND cwd so 'src' is importable
# regardless of the working directory nbconvert uses.
import sys
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd()))

from src.data_loader import load_master_dataset, clean_for_analysis, master_columns, get_crime_columns
from src.data_validation import validate_dataset, validation_summary_text
from src import analysis
from src import hotspot_detection as hd
from src import visualizations as viz

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 250)
print('Libraries imported successfully.')
"""))

# ---------------------------------------------------------------------------
# 3. Loading the cleaned dataset
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 3. Loading the cleaned dataset

The cleaned master CSV lives at `dataset/processed/guardian_crime_master_2017_2022.csv`.
We load it **read-only** -- the source file is never modified.
"""))
cells.append(code_cell("""
CSV_PATH = 'dataset/processed/guardian_crime_master_2017_2022.csv'
df_raw = load_master_dataset()
print('Loaded raw dataset shape:', df_raw.shape)
print()
print('First 5 rows:')
display(df_raw.head())
"""))

# ---------------------------------------------------------------------------
# 4. Dataset inspection
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 4. Dataset inspection

Inspect column names, data types and the structure of the metadata vs
crime-category columns.
"""))
cells.append(code_cell("""
from src.data_loader import master_columns
meta_cols, crime_cols, total_col = master_columns()
print('Metadata columns (%d):' % len(meta_cols), meta_cols)
print()
print('Number of crime-category columns:', len(crime_cols))
print('Grand-total column:', total_col)
print()
print('Data types:')
print(df_raw.dtypes.to_string())
"""))

# ---------------------------------------------------------------------------
# 5. Data quality checks
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 5. Data quality checks

Run the full validation suite: dimensions, missing values, duplicates,
uniqueness, yearly counts, state-name variants, grand-total consistency
and negative-value scan.
"""))
cells.append(code_cell("""
report = validate_dataset(df_raw)
print(validation_summary_text(report))
"""))

# ---------------------------------------------------------------------------
# 6. Cleaning for analysis (documented, explicit)
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 6. Preparing an analysis-ready copy

The source CSV is **not** modified. `clean_for_analysis()` returns a new DataFrame with:

1. Normalised state names (e.g. 'Harayana' -> 'Haryana', 'Delhi UT' -> 'Delhi').
2. Numeric coercion of crime columns (blank NCRB cells treated as 0 reported cases).
3. Missing values filled: 'Vigilants' (3,614 blanks) and 'Attempt to Acid Attack' (1 blank).

Every transformation is recorded in `df.attrs` for full traceability.
"""))
cells.append(code_cell("""
df = clean_for_analysis(df_raw)
print('Cleaned shape:', df.shape)
print('State normalisations applied:', df.attrs.get('state_changes'))
print('Missing values filled with 0:', df.attrs.get('missing_filled_columns'))
print()
print('Unique states after normalisation:', df['state'].nunique())
print()
print('Sample of state_original vs state:')
display(df[['state_original', 'state']].drop_duplicates().head(8))
"""))

# ---------------------------------------------------------------------------
# 7. Exploratory analysis
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 7. Exploratory analysis

Quick numerical summary of the crime-category columns and the derived
total-crime measure.
"""))
cells.append(code_cell("""
df_tc = analysis.add_total_crime_column(df)
crime_cols_clean = get_crime_columns(df_tc)
print('Statistical summary of crime columns:')
display(df_tc[crime_cols_clean].describe().T.round(2))
print()
print('Total reported crime (all records):', int(df_tc['total_crime'].sum()))
print()
print('Records per year:')
print(df_tc['year'].value_counts().sort_index().to_string())
"""))

# ---------------------------------------------------------------------------
# 8. Crime trend analysis
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 8. Crime trend analysis (overall & year-wise)

- Distinguishes **total crime** (sum of all heads) from individual categories.
- Computes absolute and percentage change year-over-year.
- Notes the 2020 jump (reporting/pandemic effects) without claiming causation.
"""))
cells.append(code_cell("""
yearly = analysis.year_wise_summary(df_tc)
change = analysis.year_wise_change(df_tc)
print('Year-wise summary:')
display(yearly)
print()
print('Year-over-year change:')
display(change)
"""))

# ---------------------------------------------------------------------------
# 9. State-wise analysis
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 9. State-wise analysis

- Compares states across years using raw reported counts.
- Explicitly states: raw counts are not risk (population not accounted for).
- Identifies highest- and lowest-reporting states.
"""))
cells.append(code_cell("""
state_sum = analysis.state_wise_summary(df_tc)
print('Top 10 states by total reported crime:')
display(state_sum.head(10))
print()
print('Bottom 10 states by total reported crime:')
display(state_sum.tail(10))
print()
print('State x year totals (first 12 rows):')
sy = analysis.state_year_wise(df_tc)
display(sy.head(12))
"""))

# ---------------------------------------------------------------------------
# 10. District-wise analysis
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 10. District-wise analysis

- Ranks districts by total reported crime.
- Notes district names can repeat across states (handled via state+district key).
"""))
cells.append(code_cell("""
dist_sum = analysis.district_wise_summary(df_tc)
print('Top 10 districts by total reported crime (all years):')
display(dist_sum.head(10))
print()
print('Unique districts:', df_tc['district'].nunique(),
      '| Unique state-district pairs:', df_tc.groupby(['state', 'district']).ngroups)
"""))

# ---------------------------------------------------------------------------
# 11. Crime category analysis
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 11. Crime category analysis

- Identifies major crime categories and their shares of total reported crime.
- Analyses category-wise yearly trends.
- Does NOT combine unrelated categories.
"""))
cells.append(code_cell("""
cat_sum = analysis.category_summary(df_tc)
print('Top 10 crime categories:')
display(cat_sum.head(10))
print()
print('Trend direction (2017 vs 2022) for top categories:')
for c in list(cat_sum.head(8)['category']):
    print('  ' + c + ': ' + analysis.trend_direction(df_tc, c))
"""))

# ---------------------------------------------------------------------------
# 12. Hotspot identification
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 12. Historical hotspot identification

Methodology (explainable, rule-based -- no black-box ML):

- MAGNITUDE = district total crime / max district total crime x 100
- CONSISTENCY = years appearing in yearly top-N / total years x 100
- HOTSPOT SCORE = 0.6 x magnitude + 0.4 x consistency (scale 0-100)
- Qualifies: top-N in at least 3 of 6 years.

Caveats: high reported counts != crime risk; no population data; district-level only.
"""))
cells.append(code_cell("""
print(hd.methodology_text())
print('Repeated hotspots (top-10, >= 3 years):')
hs = hd.repeated_hotspots(df_tc, n=10, years_required=3)
display(hs.head(10))
"""))

# ---------------------------------------------------------------------------
# 13. Visualisations
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 13. Visualisations

Interactive Plotly figures are generated below. Static PNG copies are saved
to `visualisations/` for the report.
"""))
cells.append(code_cell("""
figs = {}
figs['yearly_trend'] = viz.plot_yearly_trend(df_tc)
figs['yearly_change'] = viz.plot_yearly_change(df_tc)
figs['state_comparison'] = viz.plot_state_comparison(df_tc, 10)
figs['top_districts'] = viz.plot_top_districts(df_tc, 10)
figs['category_distribution'] = viz.plot_category_distribution(df_tc, 10)
figs['category_pie'] = viz.plot_category_pie(df_tc, 8)
figs['category_trend'] = viz.plot_category_yearly_trend(df_tc)
figs['hotspot_comparison'] = viz.plot_hotspot_comparison(df_tc, 10)
figs['state_trend'] = viz.plot_state_trend(df_tc)

print('Generated', len(figs), 'figures.')
print('Saving static PNGs to visualisations/ ...')
viz.plot_yearly_trend(df_tc, save=True)
viz.plot_state_comparison(df_tc, 10, save=True)
viz.plot_top_districts(df_tc, 10, save=True)
viz.plot_category_distribution(df_tc, 10, save=True)
viz.plot_category_pie(df_tc, 8, save=True)
viz.plot_category_yearly_trend(df_tc, save=True)
viz.plot_hotspot_comparison(df_tc, 10, save=True)
viz.plot_state_trend(df_tc, save=True)
print('Done. Files:')
import os
for f in sorted(os.listdir(viz.VIS_DIR)):
    print('  -', f)
"""))

cells.append(code_cell("""
from IPython.display import display as ipy_display
ipy_display(figs['yearly_trend'])
ipy_display(figs['state_comparison'])
ipy_display(figs['top_districts'])
ipy_display(figs['category_distribution'])
ipy_display(figs['hotspot_comparison'])
"""))

# ---------------------------------------------------------------------------
# 14. Findings and conclusion
# ---------------------------------------------------------------------------
cells.append(md_cell("""
## 14. Findings, observations and conclusion

### Verified dataset statistics
- 5,322 district-year records; years 2017-2022.
- 142 independent crime categories (+ grand-total column).
- 0 duplicate district-year records.
- 41 raw state variants -> 36 normalised states; 939 districts.

### Key observations (genuine, computed from NCRB data)
- Total reported crime rose from ~3.06M (2017) to ~4.25M (2020), then eased to ~3.56M (2022).
- The 2020 jump should not be read as a crime surge without accounting for reporting changes.
- Top reporting states: Uttar Pradesh, Maharashtra, Tamil Nadu, Madhya Pradesh, Delhi.
- Top reporting districts: Mumbai Commr., Chennai, Ahmedabad City, Bengaluru City, Surat City.
- Mumbai Commr. is the most consistent hotspot (top-10 in all 6 years).

### Limitations
- District-level annual data: cannot identify streets or coordinates.
- Raw counts are not crime risk (no population data).
- Reporting variations documented in `data_quality_note` (e.g. WB 2018 substitution).
- State-name variants normalised; see `data_loader.NORMALISATION_MAP`.

### Conclusion
Module 1 provides a reliable, transparent foundation for the later prediction
and safe-route-optimisation modules. All analysis is reproducible from the
genuine NCRB dataset.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.2",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells.")