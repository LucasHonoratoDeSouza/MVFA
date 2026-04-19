from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bibliography import export_bibliography
from config import (
    COUNTRY_NORMALIZATION,
    CORE_REQUIRED_COLUMNS,
    PANEL_FILE,
    PROCESSED_PANEL_FILE,
    RAW_COLUMN_ALIASES,
    TABLES_DIR,
)
from data_sources import build_real_api_panel


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        key = column.strip().lower()
        renamed[column] = RAW_COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


def normalize_country_names(series: pd.Series) -> pd.Series:
    def _normalize(value: object) -> object:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        mapped = COUNTRY_NORMALIZATION.get(cleaned.lower())
        return mapped or cleaned

    return series.map(_normalize)


def compute_average_debt(df: pd.DataFrame) -> pd.DataFrame:
    if "avg_debt_nominal" in df.columns and df["avg_debt_nominal"].notna().any():
        return df
    if "debt_nominal" not in df.columns:
        return df

    df = df.copy()
    lagged = df.groupby("country")["debt_nominal"].shift(1)
    df["avg_debt_nominal"] = df["debt_nominal"].where(lagged.isna(), (df["debt_nominal"] + lagged) / 2.0)
    return df


def infer_forward_inflation(df: pd.DataFrame) -> pd.DataFrame:
    if "inflation_forward_12m" in df.columns and df["inflation_forward_12m"].notna().any():
        return df

    df = df.copy()
    df["inflation_forward_12m"] = df.groupby("country")["gdp_deflator_inflation"].shift(-1)
    df["inflation_forward_12m"] = df["inflation_forward_12m"].fillna(df["gdp_deflator_inflation"])
    return df


def validate_core_columns(df: pd.DataFrame) -> None:
    missing = [column for column in CORE_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Base de dados incompleta. Colunas obrigatorias ausentes: {missing_text}")


def clean_dataset(input_path: Path | str = PANEL_FILE, output_path: Path | str = PROCESSED_PANEL_FILE) -> pd.DataFrame:
    source = Path(input_path)
    if not source.exists():
        build_real_api_panel(refresh=False)
    if not source.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado em {source}.")

    df = pd.read_csv(source)
    df = normalize_columns(df)
    validate_core_columns(df)

    df = df.copy()
    df["country"] = normalize_country_names(df["country"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)

    preserved_text_columns = {"money_source", "yield_source", "idec_source"}
    object_columns = set(df.select_dtypes(include=["object"]).columns) - {"country"} - preserved_text_columns
    for column in object_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_candidates = [column for column in df.columns if column not in {"country"}]
    for column in numeric_candidates:
        if column in preserved_text_columns:
            continue
        if column == "reserve_currency_flag":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "reserve_currency_flag" in df.columns:
        df["reserve_currency_flag"] = df["reserve_currency_flag"].fillna(0).astype(int)
    else:
        df["reserve_currency_flag"] = df["country"].eq("United States").astype(int)

    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    df = compute_average_debt(df)
    df = infer_forward_inflation(df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    export_bibliography(output_dir=TABLES_DIR)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpa e padroniza a base MVFA.")
    parser.add_argument("--input", default=str(PANEL_FILE))
    parser.add_argument("--output", default=str(PROCESSED_PANEL_FILE))
    args = parser.parse_args()
    clean_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
