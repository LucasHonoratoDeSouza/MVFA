from __future__ import annotations

import numpy as np
import pandas as pd

from config import ADVANCED_ECONOMIES, RISK_WEIGHTS


MANDATORY_COLUMNS = [
    "country",
    "year",
    "debt_gdp",
    "avg_debt_nominal",
    "interest_paid_nominal",
    "primary_balance_gdp",
    "gdp_real_growth",
    "gdp_deflator_inflation",
    "policy_rate_nominal",
    "yield_10y_nominal",
    "inflation_forward_12m",
    "m2_nominal",
    "gdp_nominal",
    "gdp_real",
    "inflation_target",
    "tax_revenue_nominal",
]

OPTIONAL_COLUMNS = [
    "threshold_debt_gdp",
    "domar_beta",
    "fx_real_change",
    "debt_nominal",
    "housing_credit_share",
    "investment_sensitive_nominal",
    "investment_total_nominal",
    "private_savings_gdp",
    "natural_rate_hlw",
    "natural_rate_lm",
    "natural_rate_favar",
    "natural_rate_hp",
    "natural_rate_gdp_trend",
]

OFFICIAL_RSTAR_COLUMNS = ["natural_rate_hlw", "natural_rate_lm", "natural_rate_favar"]
AUXILIARY_RSTAR_COLUMNS = ["natural_rate_hp", "natural_rate_gdp_trend"]
DOMAR_CORE_THRESHOLD = 0.5
ISF_CORE_THRESHOLD = 20.0
ISF_WARNING_THRESHOLD = 10.0


def _rolling_mean(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _normalize_share(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    max_value = series.dropna().abs().max()
    if max_value > 1.5:
        return series / 100.0
    return series


def estimate_domar_betas(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    if "debt_nominal" in work.columns and "gdp_nominal" in work.columns:
        delta_b = work.groupby("country")["debt_nominal"].diff()
        work["debt_change_share_gdp"] = (delta_b / work["gdp_nominal"]) * 100.0
    else:
        work["debt_change_share_gdp"] = work.groupby("country")["debt_gdp"].diff()

    provided = work["domar_beta"] if "domar_beta" in work.columns else pd.Series(index=work.index, dtype=float)
    by_country: dict[str, float] = {}
    for country, frame in work.groupby("country"):
        if country == "United States":
            by_country[country] = 0.0
            continue
        explicit = frame["domar_beta"].dropna() if "domar_beta" in frame.columns else pd.Series(dtype=float)
        if not explicit.empty:
            by_country[country] = float(explicit.iloc[-1])
            continue
        sample = frame[["yield_10y_nominal", "debt_change_share_gdp"]].dropna()
        if len(sample) < 4:
            by_country[country] = 0.0
            continue
        x = np.column_stack([np.ones(len(sample)), sample["debt_change_share_gdp"].to_numpy()])
        y = sample["yield_10y_nominal"].to_numpy()
        beta = np.linalg.lstsq(x, y, rcond=None)[0][1]
        by_country[country] = float(max(beta, 0.0))

    return work["country"].map(by_country).fillna(provided).fillna(0.0)


def calculate_domar_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["interest_paid_nominal_adj"] = result["interest_paid_nominal"].clip(lower=0.0)
    result["interest_paid_net_negative_flag"] = result["interest_paid_nominal"].lt(0).fillna(False)
    nominal_effective_rate = (result["interest_paid_nominal_adj"] / result["avg_debt_nominal"]) * 100.0
    result["effective_nominal_rate"] = nominal_effective_rate.replace([np.inf, -np.inf], np.nan)
    result["effective_real_rate"] = result["effective_nominal_rate"] - result["gdp_deflator_inflation"]
    result["domar_beta_est"] = estimate_domar_betas(result)

    if "debt_nominal" in result.columns and "gdp_nominal" in result.columns:
        delta_b = result.groupby("country")["debt_nominal"].diff()
        result["debt_change_share_gdp"] = (delta_b / result["gdp_nominal"]) * 100.0
    else:
        result["debt_change_share_gdp"] = result.groupby("country")["debt_gdp"].diff()

    reserve_currency = result["reserve_currency_flag"].fillna(0).astype(bool)
    result["marginal_real_rate"] = result["effective_real_rate"]
    result.loc[~reserve_currency, "marginal_real_rate"] = (
        result.loc[~reserve_currency, "effective_real_rate"]
        + result.loc[~reserve_currency, "domar_beta_est"] * result.loc[~reserve_currency, "debt_change_share_gdp"].fillna(0.0)
    )

    result["required_primary_balance_gdp"] = (
        ((result["marginal_real_rate"] - result["gdp_real_growth"]) / 100.0) * result["debt_gdp"]
    )
    result["domar_gap"] = result["required_primary_balance_gdp"] - result["primary_balance_gdp"]
    result["domar_inflation_erosion_flag"] = (
        result["domar_gap"].lt(-15.0)
        & result["effective_real_rate"].lt(-10.0)
        & result["gdp_deflator_inflation"].gt(20.0)
    )

    gap_positive = result["domar_gap"].gt(0).astype(int)
    result["domar_positive_streak"] = (
        gap_positive.groupby(result["country"])
        .transform(lambda series: series.groupby((series == 0).cumsum()).cumsum())
        .astype(int)
    )

    margin_floor = result["required_primary_balance_gdp"].abs().clip(lower=0.25) * 0.10
    result["domar_margin_flag"] = result["domar_gap"].between(-margin_floor, 0.0, inclusive="both")
    return result


def calculate_wicksell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["real_market_rate"] = result["policy_rate_nominal"] - result["inflation_forward_12m"]

    for column in [*OFFICIAL_RSTAR_COLUMNS, *AUXILIARY_RSTAR_COLUMNS]:
        if column not in result.columns:
            result[column] = np.nan

        # Preserve country-level continuity when estimators have end-sample gaps.
        numeric_series = pd.to_numeric(result[column], errors="coerce")
        result[column] = numeric_series.groupby(result["country"]).ffill().bfill()

    result["natural_rate_op_1"] = result["natural_rate_hlw"].combine_first(result["natural_rate_hp"])
    result["natural_rate_op_2"] = result["natural_rate_lm"].combine_first(result["natural_rate_gdp_trend"])
    result["natural_rate_op_3"] = result["natural_rate_favar"]

    estimator_columns = []
    for column in ["natural_rate_op_1", "natural_rate_op_2", "natural_rate_op_3"]:
        delta_column = column.replace("natural_rate_", "delta_w_")
        result[delta_column] = result[column] - result["real_market_rate"]
        estimator_columns.append(delta_column)
        positive_component = result[delta_column].clip(lower=0.0)
        result[delta_column.replace("delta_w_", "i_delta_w_")] = positive_component.groupby(result["country"]).cumsum()

    result["wicksell_official_count"] = result[OFFICIAL_RSTAR_COLUMNS].notna().sum(axis=1)
    result["wicksell_estimator_count"] = result[estimator_columns].notna().sum(axis=1)
    result["wicksell_auxiliary_count"] = (result["wicksell_estimator_count"] - result["wicksell_official_count"]).clip(lower=0)
    result["wicksell_delta_min"] = result[estimator_columns].min(axis=1)
    result["wicksell_delta_max"] = result[estimator_columns].max(axis=1)
    cumulative_columns = [column.replace("delta_w_", "i_delta_w_") for column in estimator_columns]
    result["wicksell_integral_mid"] = result[cumulative_columns].mean(axis=1)
    result["wicksell_strict_signal"] = (result["wicksell_estimator_count"] >= 2) & (result["wicksell_delta_min"] > 0)
    result["wicksell_soft_signal"] = (result["wicksell_estimator_count"] >= 1) & (result["wicksell_delta_max"] > 0)
    return result


def calculate_iem_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["money_velocity"] = result["gdp_nominal"] / result["m2_nominal"]
    result["m2_growth_pct"] = result.groupby("country")["m2_nominal"].pct_change(fill_method=None) * 100.0
    result["velocity_growth_pct"] = result.groupby("country")["money_velocity"].pct_change(fill_method=None) * 100.0
    result["real_output_growth_pct"] = result.groupby("country")["gdp_real"].pct_change(fill_method=None) * 100.0
    result["ema_pct"] = (
        result["m2_growth_pct"]
        + result["velocity_growth_pct"]
        - result["real_output_growth_pct"]
        - result["inflation_target"]
    )
    result["m2_to_gdp_ratio"] = result["m2_nominal"] / result["gdp_nominal"]
    result["delta_m2_to_gdp_ratio"] = result.groupby("country")["m2_to_gdp_ratio"].diff()
    result["ema_3y_avg"] = result.groupby("country")["ema_pct"].transform(_rolling_mean)
    return result


def calculate_isf_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    interest_numerator = result["interest_paid_nominal_adj"] if "interest_paid_nominal_adj" in result.columns else result["interest_paid_nominal"].clip(lower=0.0)
    result["isf_pct"] = (interest_numerator / result["tax_revenue_nominal"]) * 100.0
    result["rollover_gap_pct"] = result["yield_10y_nominal"] - result["effective_nominal_rate"]
    result["isf_pct"] = result["isf_pct"].replace([np.inf, -np.inf], np.nan)

    conditions = [
        result["isf_pct"] < 10,
        result["isf_pct"].between(10, 15, inclusive="left"),
        result["isf_pct"].between(15, 20, inclusive="left"),
        result["isf_pct"] >= 20,
    ]
    choices = ["confortavel", "atencao", "pressao", "dominancia_fiscal"]
    result["isf_zone"] = np.select(conditions, choices, default="indefinido")
    return result


def calculate_idec_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["idec"] = np.nan
    result["idec_source"] = "indisponivel"

    if {"investment_sensitive_nominal", "investment_total_nominal", "private_savings_gdp"}.issubset(result.columns):
        denominator = (result["private_savings_gdp"] / 100.0).replace({0.0: np.nan})
        direct = (result["investment_sensitive_nominal"] / result["investment_total_nominal"]) / denominator
        valid_direct = direct.replace([np.inf, -np.inf], np.nan)
        result.loc[valid_direct.notna(), "idec"] = valid_direct
        result.loc[valid_direct.notna(), "idec_source"] = "fbcf_setorial"

    if "housing_credit_share" in result.columns:
        housing_share = _normalize_share(result["housing_credit_share"])
        denominator = (result["private_savings_gdp"] / 100.0).replace({0.0: np.nan})
        proxy = housing_share / denominator
        mask = result["idec"].isna() & proxy.notna()
        result.loc[mask, "idec"] = proxy[mask]
        result.loc[mask, "idec_source"] = "proxy_credito"

    result["delta_idec"] = result.groupby("country")["idec"].diff()
    return result


def _normalize_min_max(series: pd.Series) -> pd.Series:
    min_value = series.min(skipna=True)
    max_value = series.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(min_value, max_value):
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def calculate_icra(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for metric in RISK_WEIGHTS:
        normalized_column = f"{metric}_norm"
        result[normalized_column] = result.groupby("year")[metric].transform(_normalize_min_max)

    result["icra"] = 0.0
    for metric, weight in RISK_WEIGHTS.items():
        result["icra"] += result[f"{metric}_norm"].fillna(0.0) * weight
    return result


def classify_rows(
    df: pd.DataFrame,
    domar_core_threshold: float = DOMAR_CORE_THRESHOLD,
    isf_core_threshold: float = ISF_CORE_THRESHOLD,
    isf_warning_threshold: float = ISF_WARNING_THRESHOLD,
) -> pd.DataFrame:
    result = df.copy()
    advanced = result["country"].isin(ADVANCED_ECONOMIES)
    default_threshold = np.where(advanced, 120.0, 80.0)
    result["threshold_debt_gdp"] = result["threshold_debt_gdp"].fillna(pd.Series(default_threshold, index=result.index))
    result["domar_core_violation"] = result["domar_gap"] > domar_core_threshold
    result["domar_marginal_positive_flag"] = result["domar_gap"].gt(0.0) & result["domar_gap"].le(domar_core_threshold)

    core_severe_flags = [
        result["domar_core_violation"],
        result["isf_pct"] >= isf_core_threshold,
    ]
    supplementary_severe_flags = [
        result["wicksell_strict_signal"],
        result["ema_3y_avg"] > 0,
        result["idec"] > 1,
        result["icra"] > 0.70,
    ]
    warning_flags = [
        result["domar_margin_flag"],
        result["domar_marginal_positive_flag"],
        result["isf_pct"].between(isf_warning_threshold, isf_core_threshold, inclusive="left"),
        result["wicksell_soft_signal"],
        result["idec"].between(0.80, 1.0, inclusive="left"),
        result["icra"].between(0.40, 0.70, inclusive="left"),
    ]

    result["missing_metric_count"] = result[["domar_gap", "wicksell_delta_min", "ema_pct", "isf_pct", "idec"]].isna().sum(axis=1)
    coverage_warning = result["missing_metric_count"] >= 2
    warning_flags.append(coverage_warning)

    core_severe_count = sum(flag.fillna(False).astype(int) for flag in core_severe_flags)
    supplementary_severe_count = sum(flag.fillna(False).astype(int) for flag in supplementary_severe_flags)
    warning_count = sum(flag.fillna(False).astype(int) for flag in warning_flags)
    result["core_severe_count"] = core_severe_count
    result["supplementary_severe_count"] = supplementary_severe_count
    result["severe_flag_count"] = core_severe_count + supplementary_severe_count
    result["warning_flag_count"] = warning_count

    result["status_prelim"] = np.select(
        [
            (core_severe_count >= 1) | (supplementary_severe_count >= 2),
            (supplementary_severe_count == 1) | (warning_count >= 1),
        ],
        [
            "INSUSTENTAVEL",
            "LIMIAR",
        ],
        default="SUSTENTAVEL",
    )

    driver = pd.Series("Sem violacao observada dominante", index=result.index, dtype="object")
    driver = driver.mask(result["domar_core_violation"], "Violação observada da condição de Domar")
    driver = driver.mask(
        (~result["domar_core_violation"]) & (result["isf_pct"] >= isf_core_threshold),
        "Pressão de serviço fiscal acima do limiar",
    )
    driver = driver.mask(
        (result["core_severe_count"] == 0) & (result["supplementary_severe_count"] >= 2),
        "Acúmulo de sinais auxiliares severos (Wicksell/EMA/IDEC/ICRA)",
    )
    driver = driver.mask(
        (result["status_prelim"] == "LIMIAR") & (result["missing_metric_count"] >= 2),
        "Cobertura incompleta em métricas-chave; classificação conservadora",
    )
    driver = driver.mask(
        (result["status_prelim"] == "LIMIAR") & result["domar_marginal_positive_flag"],
        "Gap de Domar positivo, porém marginal (≤ 0,5 p.p. do PIB)",
    )
    result["status_driver"] = driver
    return result


def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    missing = [column for column in MANDATORY_COLUMNS if column not in result.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Colunas obrigatorias ausentes para o calculo das metricas: {missing_text}")

    for column in OPTIONAL_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    result = calculate_domar_metrics(result)
    result = calculate_wicksell_metrics(result)
    result = calculate_iem_metrics(result)
    result = calculate_isf_metrics(result)
    result = calculate_idec_metrics(result)
    result = calculate_icra(result)
    result = classify_rows(result)
    return result.sort_values(["country", "year"]).reset_index(drop=True)
