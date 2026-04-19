from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ADVANCED_ECONOMIES, DEFAULT_JAPAN_ISF_THRESHOLD
from models.var_model import VARFit


@dataclass(slots=True)
class CountrySimulationResult:
    country: str
    summary: pd.DataFrame
    fan_chart: pd.DataFrame
    scenarios: pd.DataFrame


def debt_threshold(country: str, latest_row: pd.Series) -> float:
    if pd.notna(latest_row.get("threshold_debt_gdp")):
        return float(latest_row["threshold_debt_gdp"])
    if country == "Japan":
        return 120.0
    if country in ADVANCED_ECONOMIES:
        return 120.0
    return 80.0


def _next_state(history: np.ndarray, fit: VARFit, shocks: np.ndarray, exog_value: float = 0.0) -> np.ndarray:
    next_values = np.repeat(fit.intercept.reshape(1, -1), repeats=history.shape[0], axis=0)
    for lag_index, coefficient in enumerate(fit.coefficients, start=1):
        next_values += history[:, -lag_index, :] @ coefficient.T
    if fit.exog_coefficients is not None:
        exog = np.full((history.shape[0], len(fit.exog_names)), exog_value)
        next_values += exog @ fit.exog_coefficients.T
    next_values += shocks
    return next_values


def _recalculate_debt(next_values: np.ndarray, history: np.ndarray, fit: VARFit) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(fit.variables)}
    g_idx = name_to_index["gdp_real_growth"]
    pi_idx = name_to_index["gdp_deflator_inflation"]
    r_idx = name_to_index["yield_10y_nominal"]
    s_idx = name_to_index["primary_balance_gdp"]
    b_idx = name_to_index["debt_gdp"]

    prev_b = history[:, -1, b_idx]
    r_real = next_values[:, r_idx] - next_values[:, pi_idx]
    g_real = next_values[:, g_idx]
    primary_balance = next_values[:, s_idx]
    next_values[:, b_idx] = np.maximum(prev_b * (1.0 + (r_real - g_real) / 100.0) - primary_balance, 0.0)
    return next_values


def _quantile_frame(paths: np.ndarray, years: list[int], threshold: float, country: str) -> pd.DataFrame:
    quantiles = np.quantile(paths, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    return pd.DataFrame(
        {
            "country": country,
            "year": years,
            "p10": quantiles[0],
            "p25": quantiles[1],
            "p50": quantiles[2],
            "p75": quantiles[3],
            "p90": quantiles[4],
            "prob_exceed_threshold": (paths > threshold).mean(axis=0),
            "threshold": threshold,
        }
    )


def simulate_monte_carlo(
    country_frame: pd.DataFrame,
    fit: VARFit,
    horizon: int = 10,
    n_paths: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_row = country_frame.sort_values("year").iloc[-1]
    threshold = debt_threshold(fit.country, latest_row)
    years = [int(latest_row["year"]) + step for step in range(1, horizon + 1)]
    horizon_index = min(5, horizon) - 1

    rng = np.random.default_rng(seed)
    shock_draws = rng.multivariate_normal(
        mean=np.zeros(len(fit.variables)),
        cov=fit.covariance,
        size=(n_paths, horizon),
    )

    history = np.repeat(fit.history[np.newaxis, :, :], repeats=n_paths, axis=0)
    debt_index = fit.variables.index("debt_gdp")
    debt_paths = np.zeros((n_paths, horizon))
    isf_paths = np.zeros((n_paths, horizon))
    tax_ratio = 100.0 * latest_row.get("tax_revenue_nominal", np.nan) / latest_row.get("gdp_nominal", np.nan)
    if pd.isna(tax_ratio) or tax_ratio <= 0:
        tax_ratio = np.nan

    for step in range(horizon):
        next_values = _next_state(history, fit, shock_draws[:, step, :], exog_value=0.0)
        next_values = _recalculate_debt(next_values, history, fit)
        debt_paths[:, step] = next_values[:, debt_index]
        if pd.notna(tax_ratio):
            name_to_index = {name: index for index, name in enumerate(fit.variables)}
            pi_idx = name_to_index["gdp_deflator_inflation"]
            r_idx = name_to_index["yield_10y_nominal"]
            nominal_rate = next_values[:, r_idx]
            isf_paths[:, step] = 100.0 * ((nominal_rate / 100.0) * next_values[:, debt_index]) / tax_ratio
        else:
            isf_paths[:, step] = np.nan
        history = np.concatenate([history[:, 1:, :], next_values[:, np.newaxis, :]], axis=1)

    fan_chart = _quantile_frame(debt_paths, years, threshold, fit.country)
    target_year = int(latest_row["year"]) + min(5, horizon)
    probability_5y_debt = float((debt_paths[:, horizon_index] > threshold).mean())
    probability_5y_debt_anytime = float((debt_paths[:, : horizon_index + 1] > threshold).any(axis=1).mean())
    median_debt_5y = float(np.median(debt_paths[:, horizon_index]))

    isf_threshold = DEFAULT_JAPAN_ISF_THRESHOLD if fit.country == "Japan" else 20.0
    if np.isfinite(isf_paths).any():
        probability_5y_isf = float((isf_paths[:, horizon_index] > isf_threshold).mean())
        probability_5y_isf_anytime = float((isf_paths[:, : horizon_index + 1] > isf_threshold).any(axis=1).mean())
        median_isf_5y = float(np.nanmedian(isf_paths[:, horizon_index]))
    else:
        probability_5y_isf = np.nan
        probability_5y_isf_anytime = np.nan
        median_isf_5y = np.nan

    if fit.country == "Japan" and pd.notna(probability_5y_isf):
        probability_5y = probability_5y_isf
        probability_anytime_5y = probability_5y_isf_anytime
        threshold_metric = "isf_pct"
        threshold_level = isf_threshold
    else:
        probability_5y = probability_5y_debt
        probability_anytime_5y = probability_5y_debt_anytime
        threshold_metric = "debt_gdp"
        threshold_level = threshold

    stochastic_status = "SUSTENTAVEL"
    if pd.notna(probability_5y) and probability_5y > 0.30:
        stochastic_status = "INSUSTENTAVEL"
    elif pd.notna(probability_5y) and probability_5y > 0.15:
        stochastic_status = "LIMIAR"

    summary = pd.DataFrame(
        [
            {
                "country": fit.country,
                "threshold": threshold,
                "threshold_metric": threshold_metric,
                "threshold_level": threshold_level,
                "probability_5y": probability_5y,
                "probability_anytime_5y": probability_anytime_5y,
                "probability_5y_debt": probability_5y_debt,
                "probability_anytime_5y_debt": probability_5y_debt_anytime,
                "probability_5y_isf": probability_5y_isf,
                "probability_anytime_5y_isf": probability_5y_isf_anytime,
                "median_debt_5y": median_debt_5y,
                "median_isf_5y": median_isf_5y,
                "target_year": target_year,
                "stochastic_status": stochastic_status,
                "lags": fit.lags,
            }
        ]
    )
    return fan_chart, summary


def build_deterministic_scenarios(country_frame: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    ordered = country_frame.sort_values("year").copy()
    latest = ordered.iloc[-1]
    base = ordered.tail(3).mean(numeric_only=True)

    baseline = {
        "g": base.get("gdp_real_growth", latest["gdp_real_growth"]),
        "pi": base.get("gdp_deflator_inflation", latest["gdp_deflator_inflation"]),
        "r": base.get("yield_10y_nominal", latest["yield_10y_nominal"]),
        "s": base.get("primary_balance_gdp", latest["primary_balance_gdp"]),
    }
    tax_ratio = 100.0 * latest.get("tax_revenue_nominal", np.nan) / latest.get("gdp_nominal", np.nan)

    rows: list[dict[str, float | int | str]] = []
    for scenario_name in ["baseline", "choque_juros", "recessao"]:
        debt = float(latest["debt_gdp"])
        for step in range(1, horizon + 1):
            g = baseline["g"]
            pi = baseline["pi"]
            r = baseline["r"]
            s = baseline["s"]

            if scenario_name == "choque_juros":
                r += 2.0
            elif scenario_name == "recessao" and step <= 2:
                g -= 2.5
                s -= 2.0

            r_real = r - pi
            domar_gap = ((r_real - g) / 100.0) * debt - s
            debt = max(debt * (1.0 + (r_real - g) / 100.0) - s, 0.0)

            isf_pct = np.nan
            if pd.notna(tax_ratio) and tax_ratio > 0:
                isf_pct = 100.0 * ((r / 100.0) * debt) / tax_ratio

            rows.append(
                {
                    "country": latest["country"],
                    "year": int(latest["year"]) + step,
                    "scenario": scenario_name,
                    "debt_gdp": debt,
                    "domar_gap": domar_gap,
                    "isf_pct": isf_pct,
                }
            )

    return pd.DataFrame(rows)


def run_country_analysis(
    country_frame: pd.DataFrame,
    fit: VARFit | None,
    horizon: int = 10,
    n_paths: int = 10_000,
    seed: int = 42,
) -> CountrySimulationResult:
    scenarios = build_deterministic_scenarios(country_frame, horizon=horizon)
    country = country_frame["country"].iloc[0]

    if fit is None:
        summary = pd.DataFrame(
            [
                {
                    "country": country,
                    "threshold": debt_threshold(country, country_frame.sort_values("year").iloc[-1]),
                    "threshold_metric": "debt_gdp",
                    "threshold_level": debt_threshold(country, country_frame.sort_values("year").iloc[-1]),
                    "probability_5y": np.nan,
                    "probability_anytime_5y": np.nan,
                    "probability_5y_debt": np.nan,
                    "probability_anytime_5y_debt": np.nan,
                    "probability_5y_isf": np.nan,
                    "probability_anytime_5y_isf": np.nan,
                    "median_debt_5y": np.nan,
                    "median_isf_5y": np.nan,
                    "target_year": int(country_frame["year"].max()) + min(5, horizon),
                    "stochastic_status": "NAO_ESTIMADO",
                    "lags": np.nan,
                }
            ]
        )
        fan_chart = pd.DataFrame(columns=["country", "year", "p10", "p25", "p50", "p75", "p90", "prob_exceed_threshold", "threshold"])
        return CountrySimulationResult(country=country, summary=summary, fan_chart=fan_chart, scenarios=scenarios)

    fan_chart, summary = simulate_monte_carlo(country_frame, fit, horizon=horizon, n_paths=n_paths, seed=seed)
    return CountrySimulationResult(country=country, summary=summary, fan_chart=fan_chart, scenarios=scenarios)
