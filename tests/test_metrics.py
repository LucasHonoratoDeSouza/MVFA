from __future__ import annotations

import unittest

import pandas as pd

from models.metrics import calculate_all_metrics, classify_rows


def sample_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "Brazil",
                "year": 2021,
                "debt_gdp": 80.0,
                "debt_nominal": 800.0,
                "avg_debt_nominal": 780.0,
                "interest_paid_nominal": 45.0,
                "primary_balance_gdp": -0.5,
                "gdp_real_growth": 4.0,
                "gdp_deflator_inflation": 7.0,
                "policy_rate_nominal": 5.0,
                "yield_10y_nominal": 9.0,
                "inflation_forward_12m": 6.0,
                "m2_nominal": 500.0,
                "gdp_nominal": 1000.0,
                "gdp_real": 600.0,
                "inflation_target": 3.0,
                "tax_revenue_nominal": 200.0,
                "investment_sensitive_nominal": 90.0,
                "investment_total_nominal": 180.0,
                "private_savings_gdp": 15.0,
                "housing_credit_share": 0.25,
                "fx_real_change": 2.0,
                "natural_rate_hlw": 2.0,
                "natural_rate_lm": 1.5,
                "natural_rate_favar": 2.5,
                "reserve_currency_flag": 0,
                "domar_beta": 0.1,
                "threshold_debt_gdp": 80.0,
            },
            {
                "country": "Brazil",
                "year": 2022,
                "debt_gdp": 82.0,
                "debt_nominal": 850.0,
                "avg_debt_nominal": 825.0,
                "interest_paid_nominal": 50.0,
                "primary_balance_gdp": 0.2,
                "gdp_real_growth": 2.5,
                "gdp_deflator_inflation": 5.0,
                "policy_rate_nominal": 7.0,
                "yield_10y_nominal": 10.0,
                "inflation_forward_12m": 4.5,
                "m2_nominal": 540.0,
                "gdp_nominal": 1080.0,
                "gdp_real": 615.0,
                "inflation_target": 3.0,
                "tax_revenue_nominal": 215.0,
                "investment_sensitive_nominal": 95.0,
                "investment_total_nominal": 185.0,
                "private_savings_gdp": 16.0,
                "housing_credit_share": 0.26,
                "fx_real_change": 1.0,
                "natural_rate_hlw": 2.4,
                "natural_rate_lm": 2.0,
                "natural_rate_favar": 2.8,
                "reserve_currency_flag": 0,
                "domar_beta": 0.1,
                "threshold_debt_gdp": 80.0,
            },
            {
                "country": "Brazil",
                "year": 2023,
                "debt_gdp": 84.0,
                "debt_nominal": 900.0,
                "avg_debt_nominal": 875.0,
                "interest_paid_nominal": 54.0,
                "primary_balance_gdp": 0.4,
                "gdp_real_growth": 2.0,
                "gdp_deflator_inflation": 4.0,
                "policy_rate_nominal": 8.0,
                "yield_10y_nominal": 11.0,
                "inflation_forward_12m": 4.0,
                "m2_nominal": 580.0,
                "gdp_nominal": 1160.0,
                "gdp_real": 627.0,
                "inflation_target": 3.0,
                "tax_revenue_nominal": 228.0,
                "investment_sensitive_nominal": 100.0,
                "investment_total_nominal": 190.0,
                "private_savings_gdp": 16.5,
                "housing_credit_share": 0.27,
                "fx_real_change": -1.0,
                "natural_rate_hlw": 2.8,
                "natural_rate_lm": 2.1,
                "natural_rate_favar": 3.0,
                "reserve_currency_flag": 0,
                "domar_beta": 0.1,
                "threshold_debt_gdp": 80.0,
            },
        ]
    )


class MetricTests(unittest.TestCase):
    def test_calculate_all_metrics_returns_expected_columns(self) -> None:
        metrics = calculate_all_metrics(sample_panel())
        for column in ["domar_gap", "wicksell_delta_min", "ema_pct", "isf_pct", "idec", "icra", "status_prelim"]:
            self.assertIn(column, metrics.columns)

    def test_isf_matches_interest_over_revenue(self) -> None:
        metrics = calculate_all_metrics(sample_panel())
        latest = metrics.sort_values("year").iloc[-1]
        self.assertAlmostEqual(latest["isf_pct"], 54.0 / 228.0 * 100.0, places=6)

    def test_wicksell_interval_uses_all_estimators(self) -> None:
        metrics = calculate_all_metrics(sample_panel())
        latest = metrics.sort_values("year").iloc[-1]
        self.assertLessEqual(latest["wicksell_delta_min"], latest["wicksell_delta_max"])

    def test_negative_interest_is_clipped_in_isf(self) -> None:
        panel = sample_panel()
        panel.loc[panel["year"] == 2023, "interest_paid_nominal"] = -10.0
        metrics = calculate_all_metrics(panel)
        latest = metrics.sort_values("year").iloc[-1]
        self.assertEqual(latest["interest_paid_nominal_adj"], 0.0)
        self.assertEqual(latest["isf_pct"], 0.0)

    def test_single_auxiliary_severe_signal_is_limiar(self) -> None:
        row = pd.DataFrame(
            [
                {
                    "country": "Japan",
                    "threshold_debt_gdp": 120.0,
                    "domar_gap": -4.0,
                    "isf_pct": 2.0,
                    "domar_margin_flag": False,
                    "wicksell_strict_signal": True,
                    "wicksell_soft_signal": True,
                    "ema_3y_avg": -1.0,
                    "idec": 0.4,
                    "icra": 0.25,
                    "wicksell_delta_min": 0.5,
                    "ema_pct": -2.0,
                }
            ]
        )
        classified = classify_rows(row)
        self.assertEqual(classified.iloc[0]["status_prelim"], "LIMIAR")

    def test_wicksell_uses_auxiliary_estimators_when_official_series_are_missing(self) -> None:
        panel = sample_panel()
        panel["natural_rate_hlw"] = pd.NA
        panel["natural_rate_lm"] = pd.NA
        panel["natural_rate_favar"] = pd.NA
        panel["natural_rate_hp"] = [1.0, 1.1, 1.2]
        panel["natural_rate_gdp_trend"] = [1.5, 1.6, 1.7]
        metrics = calculate_all_metrics(panel)
        latest = metrics.sort_values("year").iloc[-1]
        self.assertEqual(int(latest["wicksell_estimator_count"]), 2)
        self.assertFalse(pd.isna(latest["wicksell_delta_min"]))
        self.assertFalse(pd.isna(latest["wicksell_delta_max"]))

    def test_marginal_domar_gap_is_not_core_violation(self) -> None:
        row = pd.DataFrame(
            [
                {
                    "country": "Australia",
                    "threshold_debt_gdp": 120.0,
                    "domar_gap": 0.26,
                    "isf_pct": 3.0,
                    "domar_margin_flag": False,
                    "wicksell_strict_signal": False,
                    "wicksell_soft_signal": True,
                    "ema_3y_avg": -1.0,
                    "idec": 0.1,
                    "icra": 0.30,
                    "wicksell_delta_min": -0.4,
                    "ema_pct": -2.0,
                }
            ]
        )
        classified = classify_rows(row)
        self.assertEqual(classified.iloc[0]["status_prelim"], "LIMIAR")
        self.assertFalse(bool(classified.iloc[0]["domar_core_violation"]))

    def test_custom_domar_threshold_changes_boundary_classification(self) -> None:
        row = pd.DataFrame(
            [
                {
                    "country": "Australia",
                    "threshold_debt_gdp": 120.0,
                    "domar_gap": 0.26,
                    "isf_pct": 3.0,
                    "domar_margin_flag": False,
                    "wicksell_strict_signal": False,
                    "wicksell_soft_signal": False,
                    "ema_3y_avg": -1.0,
                    "idec": 0.1,
                    "icra": 0.10,
                    "wicksell_delta_min": -0.4,
                    "ema_pct": -2.0,
                }
            ]
        )
        baseline = classify_rows(row, domar_core_threshold=0.5)
        tighter = classify_rows(row, domar_core_threshold=0.2)
        self.assertEqual(baseline.iloc[0]["status_prelim"], "LIMIAR")
        self.assertEqual(tighter.iloc[0]["status_prelim"], "INSUSTENTAVEL")
        self.assertTrue(bool(tighter.iloc[0]["domar_core_violation"]))


if __name__ == "__main__":
    unittest.main()
