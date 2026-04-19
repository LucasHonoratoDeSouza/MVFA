from __future__ import annotations

import argparse

import pandas as pd

from bibliography import export_bibliography
from clean_data import clean_dataset
from config import PANEL_FILE, PROCESSED_PANEL_FILE
from data_sources import build_real_api_panel
from generate_paper import generate_html_paper
from models.metrics import calculate_all_metrics
from models.monte_carlo import run_country_analysis
from models.var_model import fit_panel_var
from output import (
    ensure_output_dirs,
    export_classification_sensitivity,
    export_country_notes,
    export_diagnostic_table,
    export_limitations,
    export_metric_timeseries,
    export_pandemic_sensitivity,
    export_prompt_audit,
    export_reference_appendix,
    export_source_trace,
    export_stochastic_outputs,
    export_var_diagnostics,
)


def run_pipeline(
    raw_path: str,
    processed_path: str,
    base_year: int = 2024,
    horizon: int = 10,
    n_paths: int = 10_000,
    seed: int = 42,
    refresh_data: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    entries = export_bibliography()
    raw_panel, _quality = build_real_api_panel(refresh=refresh_data)
    panel = clean_dataset(raw_path, processed_path)
    metrics = calculate_all_metrics(panel)

    models = fit_panel_var(metrics, end_year=2019, excluded_years=(2020, 2021))
    summaries: list[pd.DataFrame] = []
    fan_charts: dict[str, pd.DataFrame] = {}
    scenarios: dict[str, pd.DataFrame] = {}

    for country, frame in metrics.groupby("country"):
        result = run_country_analysis(
            country_frame=frame.sort_values("year"),
            fit=models.get(country),
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
        )
        summaries.append(result.summary)
        fan_charts[country] = result.fan_chart
        scenarios[country] = result.scenarios

    stochastic_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    inclusive_models = fit_panel_var(metrics, end_year=2024, excluded_years=())
    inclusive_summaries: list[pd.DataFrame] = []
    for country, frame in metrics.groupby("country"):
        inclusive_result = run_country_analysis(
            country_frame=frame.sort_values("year"),
            fit=inclusive_models.get(country),
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
        )
        inclusive_summaries.append(inclusive_result.summary)
    inclusive_stochastic_summary = (
        pd.concat(inclusive_summaries, ignore_index=True) if inclusive_summaries else pd.DataFrame()
    )

    export_diagnostic_table(metrics, stochastic_summary, base_year=base_year)
    export_metric_timeseries(metrics)
    export_var_diagnostics(models)
    export_stochastic_outputs(metrics, fan_charts, stochastic_summary, scenarios)
    export_source_trace(raw_panel, base_year=base_year)
    export_country_notes(metrics, raw_panel, stochastic_summary, base_year=base_year)
    export_classification_sensitivity(metrics, stochastic_summary, base_year=base_year)
    export_pandemic_sensitivity(stochastic_summary, inclusive_stochastic_summary, base_year=base_year)
    export_limitations()
    export_prompt_audit(metrics, stochastic_summary, raw_panel)
    export_reference_appendix(entries)
    generate_html_paper(base_year=base_year)
    return metrics, stochastic_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline completo do MVFA.")
    parser.add_argument("--raw", default=str(PANEL_FILE))
    parser.add_argument("--processed", default=str(PROCESSED_PANEL_FILE))
    parser.add_argument("--base-year", type=int, default=2024)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        raw_path=args.raw,
        processed_path=args.processed,
        base_year=args.base_year,
        horizon=args.horizon,
        n_paths=args.n_paths,
        seed=args.seed,
        refresh_data=args.refresh_data,
    )


if __name__ == "__main__":
    main()
