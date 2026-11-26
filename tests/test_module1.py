"""
Guardian AI - Module 1
tests/test_module1.py
--------------------
Tests for the crime analysis module:
  - dataset loading
  - data validation
  - duplicate detection
  - missing value handling
  - aggregation correctness
  - year-wise / state-wise / district-wise summaries
  - hotspot identification logic

Run with:
    python -m unittest tests.test_module1 -v
"""

import os
import unittest

import pandas as pd

# Allow `import src` and `import dashboard` when run from project root.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_master_dataset,
    clean_for_analysis,
    get_crime_columns,
    MASTER_CSV,
)
from src.data_validation import validate_dataset
from src import analysis
from src import hotspot_detection as hd


class GuardianModule1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_master_dataset()
        cls.clean = clean_for_analysis(cls.raw)
        cls.clean = analysis.add_total_crime_column(cls.clean)
        cls.crime_cols = get_crime_columns(cls.clean)

    # ---------------- Dataset loading ----------------
    def test_file_exists(self):
        self.assertTrue(MASTER_CSV.exists(), f"Missing {MASTER_CSV}")

    def test_load_shape(self):
        self.assertEqual(self.raw.shape, (5322, 152))

    def test_load_columns_present(self):
        for c in ["year", "state", "district", "Murder (Sec.302 IPC)"]:
            self.assertIn(c, self.raw.columns)

    def test_crime_column_count(self):
        # 142 independent categories (143 incl. grand-total column)
        self.assertEqual(len(self.crime_cols), 142)
        self.assertNotIn("total_crime", self.crime_cols)
        self.assertNotIn("Total Cognizable IPC crimes", self.crime_cols)

    # ---------------- Data validation ----------------
    def test_validation_passes(self):
        report = validate_dataset(self.raw)
        self.assertTrue(report.get("passed"), msg="\n".join(report.get("messages", [])))

    def test_no_duplicates(self):
        report = validate_dataset(self.raw)
        self.assertEqual(report["duplicates"]["full_row"], 0)
        self.assertEqual(report["duplicates"]["year_state_district"], 0)

    def test_no_negative_counts(self):
        report = validate_dataset(self.raw)
        self.assertEqual(len(report.get("negatives", {})), 0)

    # ---------------- Missing value handling ----------------
    def test_missing_filled_with_zero(self):
        # After cleaning, no NaN in crime columns (blanks treated as 0).
        nan_count = self.clean[self.crime_cols].isna().sum().sum()
        self.assertEqual(nan_count, 0)

    def test_vigilants_filled(self):
        # Vigilants had 3,614 blanks in raw; should be 0 after fill.
        self.assertEqual(self.clean["Vigilants"].isna().sum(), 0)
        self.assertEqual(self.raw["Vigilants"].isna().sum(), 3614)

    def test_source_not_modified(self):
        # Reload raw and confirm it still has original missing values (not altered on disk).
        raw2 = load_master_dataset()
        self.assertEqual(raw2["Vigilants"].isna().sum(), 3614)

    # ---------------- Year-wise summaries ----------------
    def test_year_wise_shape_and_order(self):
        ys = analysis.year_wise_summary(self.clean)
        self.assertEqual(len(ys), 6)  # 2017-2022
        self.assertListEqual(list(ys["year"]), [2017, 2018, 2019, 2020, 2021, 2022])
        self.assertTrue((ys["total_crime"] > 0).all())

    def test_year_wise_consistency(self):
        # Sum of yearly totals == grand total
        ys = analysis.year_wise_summary(self.clean)
        self.assertAlmostEqual(ys["total_crime"].sum(), self.clean["total_crime"].sum(), places=0)

    def test_year_wise_change_first_row_nan(self):
        ch = analysis.year_wise_change(self.clean)
        self.assertTrue(pd.isna(ch.loc[0, "abs_change"]))
        self.assertTrue(pd.isna(ch.loc[0, "pct_change"]))

    def test_records_per_year_match_reported(self):
        counts = self.clean.groupby("year").size()
        expected = {2017: 852, 2018: 856, 2019: 874, 2020: 898, 2021: 908, 2022: 934}
        for yr, n in expected.items():
            self.assertEqual(int(counts[yr]), n)

    # ---------------- State-wise summaries ----------------
    def test_state_wise_sorted(self):
        ss = analysis.state_wise_summary(self.clean)
        self.assertTrue(ss["total_crime"].is_monotonic_decreasing)
        self.assertGreaterEqual(ss["state"].nunique(), 30)

    def test_state_year_wise_aggregates(self):
        sy = analysis.state_year_wise(self.clean)
        # state-year sum equals year-wise total
        yearly = analysis.year_wise_summary(self.clean).set_index("year")["total_crime"]
        by_year = sy.groupby("year")["total_crime"].sum()
        for yr in yearly.index:
            self.assertAlmostEqual(by_year[yr], yearly[yr], places=0)

    def test_top_states(self):
        top = analysis.top_states(self.clean, n=5)
        self.assertEqual(len(top), 5)
        self.assertIn("Uttar Pradesh", set(top["state"]))

    # ---------------- District-wise rankings ----------------
    def test_district_summary_not_empty(self):
        ds = analysis.district_wise_summary(self.clean)
        self.assertGreater(len(ds), 0)
        self.assertTrue(ds["total_crime"].is_monotonic_decreasing)

    def test_top_districts(self):
        top = analysis.top_districts(self.clean, n=10)
        self.assertEqual(len(top), 10)
        self.assertEqual(top.iloc[0]["district"], "Mumbai Commr.")

    def test_district_trend(self):
        trend = analysis.district_trend(self.clean, "Maharashtra", "Mumbai Commr.")
        self.assertEqual(len(trend), 6)  # one per year
        self.assertTrue((trend["total_crime"] >= 0).all())

    # ---------------- Aggregation correctness ----------------
    def test_category_summary_shares(self):
        cs = analysis.category_summary(self.clean)
        self.assertAlmostEqual(cs["pct_of_total"].sum(), 100.0, places=1)
        self.assertTrue(cs["total_crime"].is_monotonic_decreasing)

    def test_add_total_crime_column(self):
        df = analysis.add_total_crime_column(self.clean)
        self.assertIn("total_crime", df.columns)
        # total_crime uses NCRB's stored grand-total column ("Total Cognizable IPC crimes")
        # when present and complete (it is authoritative; summing all category columns
        # would double-count because categories include both totals and sub-items).
        self.assertTrue((df["total_crime"] == self.clean["Total Cognizable IPC crimes"]).all())
        self.assertTrue((df["total_crime"] >= 0).all())

    def test_year_wise_uses_stored_total(self):
        # year-wise totals come from the stored grand-total column (~20.8M grand total),
        # not from summing all category columns (which double-counts subtotals).
        ys = analysis.year_wise_summary(self.clean)
        self.assertAlmostEqual(ys["total_crime"].sum(), 20_893_615, places=-3)

    # ---------------- Hotspot identification logic ----------------
    def test_hotspot_repeated_structure(self):
        hs = hd.repeated_hotspots(self.clean, n=10, years_required=3)
        required = {"state", "district", "appearances", "hotspot_score", "consistency_pct"}
        self.assertTrue(required.issubset(set(hs.columns)))
        if not hs.empty:
            self.assertTrue((hs["appearances"] >= 3).all())
            self.assertTrue(hs["hotspot_score"].is_monotonic_decreasing)

    def test_hotspot_score_range(self):
        hs = hd.repeated_hotspots(self.clean, n=10, years_required=2)
        if not hs.empty:
            self.assertTrue((hs["hotspot_score"] >= 0).all())
            self.assertTrue((hs["hotspot_score"] <= 100).all())

    def test_top_n_per_year(self):
        yearly = hd.top_n_districts_per_year(self.clean, n=5)
        self.assertEqual(set(yearly.keys()), {2017, 2018, 2019, 2020, 2021, 2022})
        for yr, dfy in yearly.items():
            self.assertEqual(len(dfy), 5)
            self.assertTrue(dfy["total_crime"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
