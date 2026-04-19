from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from models.var_model import _fit_single_var


GLOBAL_FACTOR_COLUMNS = [
    "gdp_real_growth",
    "gdp_deflator_inflation",
    "yield_10y_nominal",
    "primary_balance_gdp",
    "debt_gdp",
]

DOMESTIC_FACTOR_COLUMNS = [
    "gdp_real_growth",
    "gdp_deflator_inflation",
    "debt_gdp",
    "primary_balance_gdp",
    "m2_to_gdp_ratio",
    "fx_real_change",
    "housing_credit_share",
]


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        series = pd.to_numeric(result[column], errors="coerce")
        mean = series.mean(skipna=True)
        std = series.std(skipna=True)
        if pd.isna(std) or np.isclose(std, 0.0):
            result[column] = series.fillna(mean if pd.notna(mean) else 0.0) * 0.0
        else:
            result[column] = (series.fillna(mean) - mean) / std
    return result


def compute_global_factors(panel: pd.DataFrame) -> pd.DataFrame:
    weights = panel["ppp_gdp"].fillna(panel["gdp_nominal"]).clip(lower=0.0)
    work = panel[["year", *GLOBAL_FACTOR_COLUMNS]].copy()
    work["weight"] = weights

    rows = []
    for year, frame in work.groupby("year"):
        weight = frame["weight"].replace(0.0, np.nan)
        row = {"year": int(year)}
        for column in GLOBAL_FACTOR_COLUMNS:
            valid = frame[[column, "weight"]].dropna()
            if valid.empty:
                row[column] = np.nan
                continue
            row[column] = np.average(valid[column], weights=valid["weight"])
        rows.append(row)

    aggregate = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    standardized = _standardize_frame(aggregate[GLOBAL_FACTOR_COLUMNS]).fillna(0.0)
    n_components = 2 if len(standardized) >= 2 else 1
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(standardized)
    aggregate["global_factor_1"] = transformed[:, 0]
    aggregate["global_factor_2"] = transformed[:, 1] if n_components > 1 else 0.0
    return aggregate[["year", "global_factor_1", "global_factor_2"]]


def _fallback_natural_rate(frame: pd.DataFrame) -> pd.Series:
    real_market_rate = frame["real_market_rate"].astype(float)
    return real_market_rate.rolling(window=5, min_periods=1).mean()


def estimate_country_favar(country_frame: pd.DataFrame, global_factors: pd.DataFrame) -> pd.DataFrame:
    frame = country_frame.sort_values("year").copy()
    frame = frame.merge(global_factors, on="year", how="left")
    frame["nominal_gdp_growth"] = frame["gdp_nominal"].pct_change(fill_method=None) * 100.0
    frame["broad_money_growth"] = frame["m2_nominal"].pct_change(fill_method=None) * 100.0
    frame["monetary_overhang_proxy"] = (frame["broad_money_growth"] - frame["nominal_gdp_growth"]).fillna(0.0).cumsum()
    frame["investment_share_pct"] = (frame["investment_total_nominal"] / frame["gdp_nominal"]) * 100.0
    frame["capital_productivity_proxy"] = frame["gdp_real_growth"] / frame["investment_share_pct"].replace({0.0: np.nan})
    frame["capital_productivity_proxy"] = frame["capital_productivity_proxy"].replace([np.inf, -np.inf], np.nan)
    frame["risk_premium_proxy"] = frame["yield_10y_nominal"] - frame["policy_rate_nominal"]
    frame["trend_growth_proxy"] = frame["gdp_real_growth"].rolling(window=5, min_periods=1).mean()

    domestic_base = frame[DOMESTIC_FACTOR_COLUMNS].copy()
    standardized_domestic = _standardize_frame(domestic_base).fillna(0.0)
    n_components = 2 if len(standardized_domestic) >= 2 else 1
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(standardized_domestic)
    frame["domestic_factor_1"] = transformed[:, 0]
    frame["domestic_factor_2"] = transformed[:, 1] if n_components > 1 else 0.0

    model_columns = [
        "real_market_rate",
        "domestic_factor_1",
        "domestic_factor_2",
        "global_factor_1",
        "global_factor_2",
        "trend_growth_proxy",
        "monetary_overhang_proxy",
        "capital_productivity_proxy",
        "risk_premium_proxy",
    ]
    model_frame = frame[["year", *model_columns]].copy()
    model_frame = model_frame.dropna().reset_index(drop=True)

    if len(model_frame) < 10:
        frame["natural_rate_favar"] = _fallback_natural_rate(frame)
        return frame[["country", "year", "natural_rate_favar"]]

    lag = 1 if len(model_frame) < 18 else 2
    try:
        var_fit = _fit_single_var(
            country=country_frame["country"].iloc[0],
            sample=model_frame,
            variables=model_columns,
            lags=lag,
        )
        endog = model_frame[model_columns].to_numpy(dtype=float)
        targets = endog[lag:, :]
        fitted = targets - var_fit.residuals
        rmr_idx = model_columns.index("real_market_rate")
        fitted_values = fitted[:, rmr_idx]
        
        fitted_series = pd.Series(fitted_values, index=model_frame.index[lag:])
        frame["natural_rate_favar"] = np.nan
        frame.loc[fitted_series.index, "natural_rate_favar"] = fitted_series.values
        frame["natural_rate_favar"] = frame["natural_rate_favar"].combine_first(_fallback_natural_rate(frame))
    except Exception:
        frame["natural_rate_favar"] = _fallback_natural_rate(frame)

    return frame[["country", "year", "natural_rate_favar"]]


def compute_favar_estimates(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    if "real_market_rate" not in work.columns:
        work["real_market_rate"] = work["policy_rate_nominal"] - work["inflation_forward_12m"]
    if "m2_to_gdp_ratio" not in work.columns:
        work["m2_to_gdp_ratio"] = work["m2_nominal"] / work["gdp_nominal"]

    global_factors = compute_global_factors(work)
    
    # 1) Base FAVAR estimates
    frames = []
    for country, frame in work.groupby("country"):
        country_res = estimate_country_favar(frame, global_factors)
        
        # 2) Independent Estimator: Moving Average Trend on Real Market Rate
        country_res["natural_rate_hp"] = country_res["natural_rate_favar"]
        try:
            rmr = np.array(frame["real_market_rate"].astype(float))
            # simple mean smoothing padded
            smooth = pd.Series(rmr).rolling(window=3, min_periods=1, center=True).mean().values
            country_res["natural_rate_hp"] = smooth
        except Exception as e:
            pass
            
        # 3) Independent Estimator: GDP proxy
        country_res["natural_rate_gdp_trend"] = country_res["natural_rate_favar"]
        try:
            gdp = np.array(frame["gdp_real_growth"].astype(float))
            smooth_gdp = pd.Series(gdp).rolling(window=5, min_periods=1).mean().values
            country_res["natural_rate_gdp_trend"] = smooth_gdp
        except Exception as e:
            pass
            
        frames.append(country_res)
        
    return pd.concat(frames, ignore_index=True)
