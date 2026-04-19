from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.var_model import _stabilize_coefficients, fit_country_var


class VarModelTests(unittest.TestCase):
    def test_stabilize_coefficients_reduces_explosive_root(self) -> None:
        coefficients = [np.array([[1.05]])]
        adjusted, raw_root, max_root, was_stabilized = _stabilize_coefficients(coefficients, target_root=0.98)
        self.assertTrue(was_stabilized)
        self.assertGreater(raw_root, 1.0)
        self.assertLess(max_root, 1.0)
        self.assertLess(adjusted[0][0, 0], coefficients[0][0, 0])

    def test_fit_country_var_respects_end_year_and_excluded_years(self) -> None:
        rows = []
        for idx, year in enumerate(range(1995, 2025), start=1):
            rows.append(
                {
                    "country": "Testland",
                    "year": year,
                    "gdp_real_growth": 2.0 + 0.1 * np.sin(idx),
                    "gdp_deflator_inflation": 3.0 + 0.1 * np.cos(idx),
                    "yield_10y_nominal": 5.0 + 0.05 * idx,
                    "primary_balance_gdp": -1.0 + 0.02 * idx,
                    "debt_gdp": 50.0 + 0.8 * idx,
                    "fx_real_change": 0.5 * np.sin(idx / 2),
                }
            )
        frame = pd.DataFrame(rows)

        baseline = fit_country_var(frame, end_year=2019, excluded_years=(2020, 2021))
        inclusive = fit_country_var(frame, end_year=2024, excluded_years=())

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(inclusive)
        assert baseline is not None
        assert inclusive is not None
        self.assertEqual(max(baseline.fitted_years), 2019)
        self.assertEqual(max(inclusive.fitted_years), 2024)
        self.assertGreater(len(inclusive.fitted_years), len(baseline.fitted_years))


if __name__ == "__main__":
    unittest.main()
