from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller


@dataclass(slots=True)
class VARFit:
    country: str
    variables: list[str]
    lags: int
    intercept: np.ndarray
    coefficients: list[np.ndarray]
    exog_names: list[str]
    exog_coefficients: np.ndarray | None
    covariance: np.ndarray
    residuals: np.ndarray
    info_criteria: dict[str, float]
    history: np.ndarray
    fitted_years: list[int]
    n_obs: int
    raw_max_root_modulus: float
    max_root_modulus: float
    is_stable: bool
    was_stabilized: bool
    adf_pvalues: dict[str, float]
    adf_statistics: dict[str, float]


def _companion_matrix(coefficients: list[np.ndarray]) -> np.ndarray:
    variables = coefficients[0].shape[0]
    lags = len(coefficients)
    top_block = np.hstack(coefficients)
    if lags == 1:
        return top_block
    lower_left = np.eye(variables * (lags - 1))
    lower_right = np.zeros((variables * (lags - 1), variables))
    lower_block = np.hstack([lower_left, lower_right])
    return np.vstack([top_block, lower_block])


def _adf_summary(sample: pd.DataFrame, variables: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    pvalues: dict[str, float] = {}
    statistics: dict[str, float] = {}
    for variable in variables:
        series = pd.to_numeric(sample[variable], errors="coerce").dropna()
        if len(series) < 8 or series.nunique() <= 1:
            pvalues[variable] = np.nan
            statistics[variable] = np.nan
            continue
        try:
            stat, pvalue, *_ = adfuller(series, autolag="AIC")
            pvalues[variable] = float(pvalue)
            statistics[variable] = float(stat)
        except Exception:
            pvalues[variable] = np.nan
            statistics[variable] = np.nan
    return pvalues, statistics


def _stabilize_coefficients(coefficients: list[np.ndarray], target_root: float = 0.98) -> tuple[list[np.ndarray], float, float, bool]:
    adjusted = [coefficient.copy() for coefficient in coefficients]
    raw_root = float(np.abs(np.linalg.eigvals(_companion_matrix(coefficients))).max())
    max_root = raw_root
    was_stabilized = False

    iteration = 0
    while max_root >= target_root and iteration < 50:
        shrink = min(0.995, target_root / max(max_root, 1e-8))
        adjusted = [coefficient * shrink for coefficient in adjusted]
        max_root = float(np.abs(np.linalg.eigvals(_companion_matrix(adjusted))).max())
        was_stabilized = True
        iteration += 1

    return adjusted, raw_root, max_root, was_stabilized


def _build_design_matrix(endog: np.ndarray, lags: int, exog: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    observations, variables = endog.shape
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for t in range(lags, observations):
        pieces = [np.array([1.0])]
        for lag in range(1, lags + 1):
            pieces.append(endog[t - lag])
        if exog is not None:
            pieces.append(exog[t])
        rows.append(np.concatenate(pieces))
        targets.append(endog[t])
    return np.vstack(rows), np.vstack(targets)


def _fit_single_var(
    country: str,
    sample: pd.DataFrame,
    variables: list[str],
    lags: int,
    exog_columns: list[str] | None = None,
) -> VARFit:
    endog = sample[variables].to_numpy(dtype=float)
    exog = sample[exog_columns].to_numpy(dtype=float) if exog_columns else None
    design, targets = _build_design_matrix(endog, lags, exog)
    beta = np.linalg.lstsq(design, targets, rcond=None)[0]

    intercept = beta[0]
    coefficients: list[np.ndarray] = []
    cursor = 1
    for _ in range(lags):
        block = beta[cursor : cursor + len(variables)]
        coefficients.append(block.T)
        cursor += len(variables)

    exog_coefficients = None
    if exog_columns:
        exog_coefficients = beta[cursor:].T

    fitted = design @ beta
    residuals = targets - fitted
    residuals = np.asarray(winsorize(residuals, limits=(0.05, 0.05), axis=0))
    dof = max(len(targets) - design.shape[1], 1)
    covariance = residuals.T @ residuals / dof
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-9

    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        logdet = np.log(np.linalg.det(covariance + np.eye(covariance.shape[0]) * 1e-6))
    n_obs = len(targets)
    k = len(variables)
    n_params = design.shape[1] * k
    loglik = -0.5 * n_obs * (k * np.log(2 * np.pi) + logdet + k)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + np.log(max(n_obs, 2)) * n_params

    coefficients, raw_max_root_modulus, max_root_modulus, was_stabilized = _stabilize_coefficients(coefficients)
    history = endog[-lags:]
    adf_pvalues, adf_statistics = _adf_summary(sample, variables)
    return VARFit(
        country=country,
        variables=variables,
        lags=lags,
        intercept=intercept,
        coefficients=coefficients,
        exog_names=exog_columns or [],
        exog_coefficients=exog_coefficients,
        covariance=covariance,
        residuals=residuals,
        info_criteria={"aic": float(aic), "bic": float(bic)},
        history=history,
        fitted_years=sample["year"].astype(int).tolist(),
        n_obs=len(sample),
        raw_max_root_modulus=raw_max_root_modulus,
        max_root_modulus=max_root_modulus,
        is_stable=bool(max_root_modulus < 1.0),
        was_stabilized=was_stabilized,
        adf_pvalues=adf_pvalues,
        adf_statistics=adf_statistics,
    )


def fit_country_var(
    country_frame: pd.DataFrame,
    max_lags: int = 2,
    criterion: str = "bic",
    end_year: int = 2019,
    excluded_years: tuple[int, ...] = (2020, 2021),
) -> VARFit | None:
    variables = [
        "gdp_real_growth",
        "gdp_deflator_inflation",
        "yield_10y_nominal",
        "primary_balance_gdp",
        "debt_gdp",
    ]
    if "fx_real_change" not in country_frame.columns:
        country_frame = country_frame.copy()
        country_frame["fx_real_change"] = np.nan
    if country_frame["country"].iloc[0] != "United States" and country_frame["fx_real_change"].notna().any():
        variables.append("fx_real_change")

    sample = country_frame[country_frame["year"].between(1995, end_year)].copy()
    if excluded_years:
        sample = sample[~sample["year"].isin(list(excluded_years))]
    sample = sample.dropna(subset=variables)
    if len(sample) < 10:
        return None

    sample["gfc_dummy"] = sample["year"].isin([2008, 2009]).astype(float)
    best_fit: VARFit | None = None
    best_score = np.inf
    max_candidate_lag = min(max_lags, max(1, len(sample) // 4))
    for lags in range(1, max_candidate_lag + 1):
        if len(sample) <= lags + 3:
            continue
        fit = _fit_single_var(
            country=country_frame["country"].iloc[0],
            sample=sample,
            variables=variables,
            lags=lags,
            exog_columns=["gfc_dummy"],
        )
        score = fit.info_criteria[criterion]
        if score < best_score:
            best_score = score
            best_fit = fit

    if best_fit is None:
        return None

    latest_history = (
        country_frame.dropna(subset=best_fit.variables)
        .sort_values("year")
        .tail(best_fit.lags)[best_fit.variables]
        .to_numpy(dtype=float)
    )
    if len(latest_history) == best_fit.lags:
        best_fit.history = latest_history
    return best_fit
def fit_panel_var(
    panel: pd.DataFrame,
    max_lags: int = 2,
    criterion: str = "bic",
    end_year: int = 2019,
    excluded_years: tuple[int, ...] = (2020, 2021),
) -> dict[str, VARFit]:
    models: dict[str, VARFit] = {}
    for country, frame in panel.groupby("country"):
        fit = fit_country_var(
            frame.sort_values("year"),
            max_lags=max_lags,
            criterion=criterion,
            end_year=end_year,
            excluded_years=excluded_years,
        )
        if fit is not None:
            models[country] = fit
    return models
