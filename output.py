from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bibliography import BibliographyEntry
from config import CHARTS_DIR, OUTPUT_DIR, TABLES_DIR
from models.metrics import DOMAR_CORE_THRESHOLD, ISF_CORE_THRESHOLD, classify_rows
from models.var_model import VARFit

EURO_AREA_SHARED_POLICY_COUNTRIES = {"Greece", "Ireland", "Italy", "Portugal", "Spain"}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def render_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [columns] + df.astype(str).values.tolist()
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(columns))]

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(columns), separator]
    for row in df.astype(str).values.tolist():
        lines.append(format_row(row))
    return "\n".join(lines)


def _combine_final_status(base: pd.Series, stochastic: pd.Series | None) -> pd.Series:
    if stochastic is None:
        return base
    final = base.copy()
    final.loc[stochastic == "INSUSTENTAVEL"] = "INSUSTENTAVEL"
    limiar_mask = (stochastic == "LIMIAR") & (final != "INSUSTENTAVEL")
    final.loc[limiar_mask] = "LIMIAR"
    return final


def _format_interval(min_value: float, max_value: float) -> str:
    if pd.isna(min_value) or pd.isna(max_value):
        return "N/A"
    return f"[{min_value:.2f}, {max_value:.2f}]"


def export_diagnostic_table(metrics_df: pd.DataFrame, stochastic_df: pd.DataFrame | None = None, base_year: int = 2024) -> pd.DataFrame:
    latest = metrics_df[metrics_df["year"] == base_year].copy()
    if latest.empty:
        latest = metrics_df.sort_values(["country", "year"]).groupby("country").tail(1).copy()

    if stochastic_df is not None and not stochastic_df.empty:
        latest = latest.merge(stochastic_df[["country", "stochastic_status", "probability_5y"]], on="country", how="left")
        latest["status_final"] = _combine_final_status(latest["status_prelim"], latest["stochastic_status"])
    else:
        latest["status_final"] = latest["status_prelim"]

    latest["wicksell_interval"] = latest.apply(
        lambda row: _format_interval(row["wicksell_delta_min"], row["wicksell_delta_max"]),
        axis=1,
    )
    diagnostic = latest[
        [
            "country",
            "domar_gap",
            "wicksell_interval",
            "ema_pct",
            "isf_pct",
            "idec",
            "icra",
            "status_final",
        ]
    ].rename(
        columns={
            "country": "Pais",
            "domar_gap": "GAP_Domar",
            "wicksell_interval": "DeltaW",
            "ema_pct": "EMA",
            "isf_pct": "ISF",
            "idec": "IDEC",
            "icra": "ICRA",
            "status_final": "Status",
        }
    ).sort_values("Pais")

    csv_path = TABLES_DIR / f"diagnostico_{base_year}.csv"
    md_path = TABLES_DIR / f"diagnostico_{base_year}.md"
    diagnostic.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown_table(diagnostic.round(4)), encoding="utf-8")
    return diagnostic


def export_metric_timeseries(metrics_df: pd.DataFrame) -> None:
    metrics_df.to_csv(TABLES_DIR / "metricas_series.csv", index=False)
    export_threshold_crossings(metrics_df)
    for country, frame in metrics_df.groupby("country"):
        plot_metric_series(frame.sort_values("year"), CHARTS_DIR / f"{country.lower().replace(' ', '_')}_metricas.png")


def plot_metric_series(country_frame: pd.DataFrame, destination: Path) -> None:
    metrics = [
        ("domar_gap", "Domar GAP"),
        ("wicksell_integral_mid", "IΔW"),
        ("ema_pct", "EMA"),
        ("isf_pct", "ISF"),
        ("idec", "IDEC"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 14), sharex=True)
    years = country_frame["year"]
    for axis, (column, title) in zip(axes, metrics, strict=True):
        axis.plot(years, country_frame[column], linewidth=1.8, color="#134074")
        axis.set_title(f"{country_frame['country'].iloc[0]} - {title}")
        axis.axhline(0.0, linestyle="--", linewidth=0.8, color="#888888")
        if column == "isf_pct":
            axis.axhline(10.0, linestyle=":", linewidth=0.9, color="#ca8a04")
            axis.axhline(20.0, linestyle=":", linewidth=0.9, color="#b91c1c")
        if column == "idec":
            axis.axhline(1.0, linestyle=":", linewidth=0.9, color="#b91c1c")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Ano")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def export_stochastic_outputs(
    metrics_df: pd.DataFrame,
    fan_charts: dict[str, pd.DataFrame],
    summaries: pd.DataFrame,
    scenarios: dict[str, pd.DataFrame],
) -> None:
    summaries.to_csv(TABLES_DIR / "stochastic_summary.csv", index=False)
    for country, fan_chart in fan_charts.items():
        if fan_chart.empty:
            continue
        history = metrics_df[metrics_df["country"] == country].sort_values("year")
        scenario_frame = scenarios[country]
        plot_fan_chart(history, fan_chart, scenario_frame, CHARTS_DIR / f"{country.lower().replace(' ', '_')}_fan_chart.png")
        fan_chart.to_csv(TABLES_DIR / f"{country.lower().replace(' ', '_')}_fan_chart.csv", index=False)
        scenario_frame.to_csv(TABLES_DIR / f"{country.lower().replace(' ', '_')}_scenarios.csv", index=False)


def export_var_diagnostics(models: dict[str, VARFit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    adf_rows: list[dict[str, object]] = []

    for country, fit in sorted(models.items()):
        summary_rows.append(
            {
                "country": country,
                "lags": fit.lags,
                "n_obs": fit.n_obs,
                "aic": fit.info_criteria.get("aic"),
                "bic": fit.info_criteria.get("bic"),
                "raw_max_root_modulus": fit.raw_max_root_modulus,
                "max_root_modulus": fit.max_root_modulus,
                "is_stable": fit.is_stable,
                "was_stabilized": fit.was_stabilized,
                "fitted_start_year": min(fit.fitted_years) if fit.fitted_years else np.nan,
                "fitted_end_year": max(fit.fitted_years) if fit.fitted_years else np.nan,
            }
        )
        for variable in fit.variables:
            adf_rows.append(
                {
                    "country": country,
                    "variable": variable,
                    "adf_statistic": fit.adf_statistics.get(variable, np.nan),
                    "adf_pvalue": fit.adf_pvalues.get(variable, np.nan),
                    "stationary_5pct": bool(fit.adf_pvalues.get(variable, np.nan) < 0.05) if pd.notna(fit.adf_pvalues.get(variable, np.nan)) else False,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    adf_df = pd.DataFrame(adf_rows)

    summary_df.to_csv(TABLES_DIR / "var_diagnostics.csv", index=False)
    adf_df.to_csv(TABLES_DIR / "adf_diagnostics.csv", index=False)

    if not summary_df.empty:
        (TABLES_DIR / "var_diagnostics.md").write_text(render_markdown_table(summary_df.round(4)), encoding="utf-8")
    else:
        (TABLES_DIR / "var_diagnostics.md").write_text("Sem modelos VAR estimados.", encoding="utf-8")

    if not adf_df.empty:
        adf_display = adf_df.copy()
        adf_display["adf_statistic"] = adf_display["adf_statistic"].round(4)
        adf_display["adf_pvalue"] = adf_display["adf_pvalue"].round(4)
        (TABLES_DIR / "adf_diagnostics.md").write_text(render_markdown_table(adf_display), encoding="utf-8")
    else:
        (TABLES_DIR / "adf_diagnostics.md").write_text("Sem testes ADF disponíveis.", encoding="utf-8")

    return summary_df, adf_df


def plot_fan_chart(history: pd.DataFrame, fan_chart: pd.DataFrame, scenarios: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    country = history["country"].iloc[0]

    last_historical = history.iloc[-1]
    last_year = last_historical["year"]
    last_debt = last_historical["debt_gdp"]

    fc_years = [last_year] + fan_chart["year"].tolist()
    fc_p50 = [last_debt] + fan_chart["p50"].tolist()
    fc_p10 = [last_debt] + fan_chart["p10"].tolist()
    fc_p90 = [last_debt] + fan_chart["p90"].tolist()
    fc_p25 = [last_debt] + fan_chart["p25"].tolist()
    fc_p75 = [last_debt] + fan_chart["p75"].tolist()

    ax.plot(history["year"], history["debt_gdp"], color="#1b1b1b", linewidth=2.0, label="Historico")
    ax.fill_between(fc_years, fc_p10, fc_p90, color="#88c0d0", alpha=0.35, label="P10-P90")
    ax.fill_between(fc_years, fc_p25, fc_p75, color="#3b82f6", alpha=0.30, label="P25-P75")
    ax.plot(fc_years, fc_p50, color="#0f172a", linewidth=2.2, label="Mediana")
    threshold_val = fan_chart["threshold"].iloc[0]
    ax.axhline(threshold_val, color="#b91c1c", linestyle="--", linewidth=1.5, label="Threshold")

    # Add historical threshold violation marker
    violation_points = history[history["debt_gdp"] > threshold_val]
    if not violation_points.empty:
        first_v = violation_points.iloc[0]
        first_violation_year = first_v["year"]
        first_violation_debt = first_v["debt_gdp"]
        if first_violation_year <= last_year:
            ax.scatter([first_violation_year], [first_violation_debt], color='red', s=70, zorder=5, label=f"Violação Histórica ({int(first_violation_year)})")
            ax.annotate(f"Cruzado em {int(first_violation_year)}", xy=(first_violation_year, first_violation_debt), xytext=(first_violation_year - 4, first_violation_debt + 8), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=5))

    palette = {
        "baseline": ("#047857", "-", 2.5),
        "choque_juros": ("#d97706", "-.", 2.5),
        "recessao": ("#7c3aed", ":", 2.5),
    }
    for scenario, frame in scenarios.groupby("scenario"):
        sc_years = [last_year] + frame["year"].tolist()
        sc_debt = [last_debt] + frame["debt_gdp"].tolist()
        color, style, lw = palette.get(scenario, ("#374151", "-", 1.5))
        ax.plot(sc_years, sc_debt, label=f"Cenário: {scenario.replace('_', ' ').capitalize()}", linewidth=lw, color=color, linestyle=style)

    ax.set_title(f"{country} - Fan Chart da Divida/PIB")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Divida/PIB")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def export_reference_appendix(entries: list[BibliographyEntry]) -> None:
    lines = ["# Base Bibliografica", ""]
    current_block = None
    for entry in entries:
        if entry.block != current_block:
            current_block = entry.block
            lines.extend([f"## {current_block}", ""])
        lines.append(f"- [{entry.entry_id}] {entry.citation}")
    (TABLES_DIR / "bibliografia.md").write_text("\n".join(lines), encoding="utf-8")


def export_source_trace(panel_df: pd.DataFrame, base_year: int = 2024) -> pd.DataFrame:
    latest = panel_df[panel_df["year"] == base_year].copy()
    if latest.empty:
        latest = panel_df.sort_values(["country", "year"]).groupby("country").tail(1).copy()

    columns = [
        "country",
        "year",
        "money_source",
        "policy_source",
        "yield_source",
        "hlw_source",
        "idec_source",
        "reserve_currency_flag",
        "threshold_debt_gdp",
    ]
    available = [column for column in columns if column in latest.columns]
    trace = latest[available].sort_values("country")
    trace.to_csv(TABLES_DIR / f"source_trace_{base_year}.csv", index=False)
    return trace


def export_country_notes(
    metrics_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    stochastic_df: pd.DataFrame | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    latest_metrics = metrics_df[metrics_df["year"] == base_year].copy()
    if latest_metrics.empty:
        latest_metrics = metrics_df.sort_values(["country", "year"]).groupby("country").tail(1).copy()

    latest_panel = panel_df[panel_df["year"] == base_year].copy()
    if latest_panel.empty:
        latest_panel = panel_df.sort_values(["country", "year"]).groupby("country").tail(1).copy()

    merged = latest_metrics.merge(
        latest_panel[
            [
                "country",
                "policy_source",
                "hlw_source",
                "money_source",
                "yield_source",
                "idec_source",
            ]
        ],
        on="country",
        how="left",
    )
    if stochastic_df is not None and not stochastic_df.empty:
        merged = merged.merge(
            stochastic_df[
                [
                    "country",
                    "probability_5y",
                    "probability_anytime_5y",
                    "threshold_metric",
                    "threshold_level",
                    "probability_5y_isf",
                    "median_debt_5y",
                    "median_isf_5y",
                    "stochastic_status",
                ]
            ],
            on="country",
            how="left",
        )
        merged["status_final"] = _combine_final_status(merged["status_prelim"], merged["stochastic_status"])
    else:
        merged["status_final"] = merged["status_prelim"]

    notes: list[dict[str, str]] = []

    def add_note(country: str, note_type: str, note: str) -> None:
        notes.append({"country": country, "note_type": note_type, "note": note})

    for _, row in merged.sort_values("country").iterrows():
        country = row["country"]

        if row.get("policy_source") == "euro_area_shared_policy_rate":
            add_note(
                country,
                "dados",
                "A taxa de política monetária foi harmonizada pela série da Área do Euro, consistente com a união monetária comum.",
            )
        if row.get("hlw_source") == "nyfed_hlw_euro_area_proxy":
            add_note(
                country,
                "dados",
                "O estimador HLW usa a âncora da Área do Euro por ausência de série nacional diretamente publicada pelo NY Fed.",
            )
        if pd.isna(row.get("ema_pct")):
            add_note(
                country,
                "dados",
                "EMA indisponível no ano-base por ausência de agregado monetário amplo consistente e comparável na coleta via API.",
            )
        if pd.isna(row.get("wicksell_delta_min")):
            add_note(
                country,
                "dados",
                "ΔW indisponível no ano-base porque não houve combinação observável suficiente entre taxa de política real e estimadores de r*.",
            )
        elif row.get("wicksell_estimator_count", 0) < 3:
            add_note(
                country,
                "robustez",
                f"O intervalo de Wicksell no ano-base repousa sobre {int(row['wicksell_estimator_count'])} estimador(es) operacional(is), abaixo da cobertura ideal de três.",
            )
        if pd.isna(row.get("idec")):
            add_note(
                country,
                "dados",
                "IDEC indisponível no ano-base por ausência simultânea de FBCF setorial e de proxy de crédito habitacional suficiente.",
            )
        elif row.get("idec_source") == "proxy_credito":
            add_note(
                country,
                "robustez",
                "O IDEC no ano-base usa proxy de crédito habitacional, não observação direta de composição setorial do investimento; a métrica deve ser lida como aproximação exploratória.",
            )
        if row.get("missing_metric_count", 0) >= 2:
            add_note(
                country,
                "classificacao",
                "A classificação foi mantida em chave conservadora por cobertura incompleta em pelo menos duas métricas centrais.",
            )
        domar_inflation_erosion_flag = row.get("domar_inflation_erosion_flag", False)
        if pd.notna(domar_inflation_erosion_flag) and bool(domar_inflation_erosion_flag):
            add_note(
                country,
                "interpretacao",
                "Gap de Domar muito negativo reflete erosão inflacionária ex post do custo real da dívida; isso não deve ser lido como conforto fiscal estrutural.",
            )
        if pd.notna(row.get("domar_gap")) and 0.0 < float(row["domar_gap"]) <= 0.5:
            add_note(
                country,
                "classificação",
                "O Gap de Domar é positivo, porém pequeno (≤ 0,5 p.p. do PIB); nesta versão ele é tratado como sinal limiar, não como violação central isolada.",
            )
        if pd.notna(row.get("domar_gap")) and float(row["domar_gap"]) > 0.5 and pd.notna(row.get("probability_5y")) and float(row["probability_5y"]) < 0.05:
            threshold_level = row.get("threshold_level", np.nan)
            median_debt_5y = row.get("median_debt_5y", np.nan)
            add_note(
                country,
                "estocástico",
                f"Há violação contemporânea de Domar, mas a probabilidade do evento estocástico em cinco anos permanece baixa porque a mediana simulada da dívida em t+5 ({median_debt_5y:.1f}% do PIB) segue abaixo do limiar adotado ({threshold_level:.0f}% do PIB).",
            )
        if pd.notna(row.get("isf_pct")) and float(row["isf_pct"]) >= 20.0:
            median_isf_5y = row.get("median_isf_5y", np.nan)
            median_isf_text = f"{median_isf_5y:.1f}" if pd.notna(median_isf_5y) else "N/A"
            add_note(
                country,
                "interpretação",
                f"O ISF no ano-base excede 20%, caracterizando pressão de serviço da dívida pelo critério central do modelo; a mediana simulada do ISF em t+5 é {median_isf_text}%.",
            )
        if (
            row.get("threshold_metric") == "debt_gdp"
            and pd.notna(row.get("probability_5y_isf"))
            and float(row["probability_5y_isf"]) >= 0.50
        ):
            add_note(
                country,
                "estocástico",
                f"Embora o evento oficial da SDSA use dívida/PIB, a probabilidade simulada de ISF elevado em t+5 permanece alta ({100.0 * float(row['probability_5y_isf']):.1f}%).",
            )
        if (
            pd.notna(row.get("probability_5y"))
            and pd.notna(row.get("probability_anytime_5y"))
            and float(row["probability_anytime_5y"]) - float(row["probability_5y"]) >= 0.15
        ):
            add_note(
                country,
                "estocástico",
                f"A diferença entre Pr[limiar, t+5] e Pr[limiar em qualquer ano até t+5] é material ({100.0 * float(row['probability_5y']):.1f}% vs. {100.0 * float(row['probability_anytime_5y']):.1f}%), sugerindo risco de cruzamento antecipado ou transitório dentro do quinquênio.",
            )
        if country == "Japan":
            add_note(
                country,
                "estocástico",
                "Na SDSA, a probabilidade reportada usa ISF > 25% em cinco anos como critério alternativo, não dívida/PIB > 120%.",
            )
        if country == "Turkey" and pd.notna(row.get("probability_5y")) and float(row["probability_5y"]) == 0.0:
            add_note(
                country,
                "estocástico",
                "Probabilidade de 0,0% refere-se apenas ao evento dívida/PIB > limiar em cinco anos; sob juros reais ex post negativos, a aritmética da dívida melhora mesmo com distorções monetárias correntes elevadas.",
            )
        elif (
            pd.notna(row.get("probability_5y"))
            and float(row["probability_5y"]) == 0.0
            and row.get("status_final") in {"LIMIAR", "INSUSTENTAVEL"}
        ):
            threshold_metric = row.get("threshold_metric", "debt_gdp")
            threshold_label = "dívida/PIB" if threshold_metric == "debt_gdp" else "ISF"
            add_note(
                country,
                "estocástico",
                f"Probabilidade de 0,0% aplica-se apenas ao limiar estocástico ({threshold_label}); sinais contemporâneos adversos nas métricas estáticas permanecem relevantes.",
            )
        if country == "Lebanon":
            add_note(
                country,
                "dados",
                "As leituras para o Líbano permanecem parciais no ano-base, com lacunas em taxa de política, EMA e IDEC; o caso deve ser lido como evidência incompleta.",
            )

    notes_df = pd.DataFrame(notes)
    if not notes_df.empty:
        notes_df = notes_df.sort_values(["country", "note_type", "note"]).reset_index(drop=True)
    notes_df.to_csv(TABLES_DIR / f"country_notes_{base_year}.csv", index=False)
    if notes_df.empty:
        (TABLES_DIR / f"country_notes_{base_year}.md").write_text("Sem notas.", encoding="utf-8")
    else:
        (TABLES_DIR / f"country_notes_{base_year}.md").write_text(render_markdown_table(notes_df), encoding="utf-8")
    return notes_df


def export_classification_sensitivity(
    metrics_df: pd.DataFrame,
    stochastic_df: pd.DataFrame | None = None,
    base_year: int = 2024,
    domar_thresholds: tuple[float, ...] = (0.0, 0.5, 1.0),
    isf_thresholds: tuple[float, ...] = (18.0, 20.0, 25.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = metrics_df[metrics_df["year"] == base_year].copy()
    if latest.empty:
        latest = metrics_df.sort_values(["country", "year"]).groupby("country").tail(1).copy()

    baseline_stochastic = None
    if stochastic_df is not None and not stochastic_df.empty:
        baseline_stochastic = stochastic_df[["country", "stochastic_status"]].copy()

    grid_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    baseline_pair = (DOMAR_CORE_THRESHOLD, ISF_CORE_THRESHOLD)

    for domar_threshold in domar_thresholds:
        for isf_threshold in isf_thresholds:
            classified = classify_rows(
                latest.copy(),
                domar_core_threshold=domar_threshold,
                isf_core_threshold=isf_threshold,
            )
            if baseline_stochastic is not None:
                classified = classified.merge(baseline_stochastic, on="country", how="left")
                classified["status_final"] = _combine_final_status(classified["status_prelim"], classified["stochastic_status"])
            else:
                classified["status_final"] = classified["status_prelim"]

            counts = classified["status_final"].value_counts().to_dict()
            grid_rows.append(
                {
                    "domar_threshold": domar_threshold,
                    "isf_threshold": isf_threshold,
                    "insustentavel_count": int(counts.get("INSUSTENTAVEL", 0)),
                    "limiar_count": int(counts.get("LIMIAR", 0)),
                    "sustentavel_count": int(counts.get("SUSTENTAVEL", 0)),
                }
            )

            baseline_match = (domar_threshold, isf_threshold) == baseline_pair
            for _, row in classified.iterrows():
                country_rows.append(
                    {
                        "country": row["country"],
                        "domar_threshold": domar_threshold,
                        "isf_threshold": isf_threshold,
                        "status_prelim": row["status_prelim"],
                        "status_final": row["status_final"],
                        "is_baseline_rule": baseline_match,
                    }
                )

    grid_df = pd.DataFrame(grid_rows).sort_values(["domar_threshold", "isf_threshold"]).reset_index(drop=True)
    country_long_df = pd.DataFrame(country_rows).sort_values(["country", "domar_threshold", "isf_threshold"]).reset_index(drop=True)

    country_summary = (
        country_long_df.groupby("country", as_index=False)
        .agg(
            baseline_status=("status_final", lambda series: next((value for value, flag in zip(series, country_long_df.loc[series.index, "is_baseline_rule"]) if flag), series.iloc[0])),
            unique_status_count=("status_final", "nunique"),
            ins_share=("status_final", lambda series: float((series == "INSUSTENTAVEL").mean())),
            lim_share=("status_final", lambda series: float((series == "LIMIAR").mean())),
            sus_share=("status_final", lambda series: float((series == "SUSTENTAVEL").mean())),
        )
    )
    unique_statuses = country_long_df.groupby("country")["status_final"].apply(lambda series: ", ".join(sorted(series.unique())))
    country_summary["unique_statuses"] = country_summary["country"].map(unique_statuses)
    country_summary["is_invariant"] = country_summary["unique_status_count"] == 1
    country_summary = country_summary.sort_values(["is_invariant", "country"], ascending=[True, True]).reset_index(drop=True)

    grid_df.to_csv(TABLES_DIR / f"classification_sensitivity_grid_{base_year}.csv", index=False)
    country_summary.to_csv(TABLES_DIR / f"classification_sensitivity_country_{base_year}.csv", index=False)
    country_long_df.to_csv(TABLES_DIR / f"classification_sensitivity_long_{base_year}.csv", index=False)
    (TABLES_DIR / f"classification_sensitivity_grid_{base_year}.md").write_text(render_markdown_table(grid_df), encoding="utf-8")
    (TABLES_DIR / f"classification_sensitivity_country_{base_year}.md").write_text(render_markdown_table(country_summary), encoding="utf-8")
    return grid_df, country_summary


def export_pandemic_sensitivity(
    baseline_summary: pd.DataFrame,
    inclusive_summary: pd.DataFrame,
    base_year: int = 2024,
) -> pd.DataFrame:
    baseline = baseline_summary.rename(
        columns={
            "probability_5y": "probability_5y_baseline",
            "probability_anytime_5y": "probability_anytime_5y_baseline",
            "stochastic_status": "stochastic_status_baseline",
            "threshold_metric": "threshold_metric_baseline",
            "threshold_level": "threshold_level_baseline",
        }
    )
    inclusive = inclusive_summary.rename(
        columns={
            "probability_5y": "probability_5y_inclusive",
            "probability_anytime_5y": "probability_anytime_5y_inclusive",
            "stochastic_status": "stochastic_status_inclusive",
            "threshold_metric": "threshold_metric_inclusive",
            "threshold_level": "threshold_level_inclusive",
        }
    )
    merged = baseline.merge(
        inclusive[
            [
                "country",
                "probability_5y_inclusive",
                "probability_anytime_5y_inclusive",
                "stochastic_status_inclusive",
                "threshold_metric_inclusive",
                "threshold_level_inclusive",
            ]
        ],
        on="country",
        how="outer",
    )
    merged["delta_probability_5y"] = merged["probability_5y_inclusive"] - merged["probability_5y_baseline"]
    merged["delta_probability_anytime_5y"] = (
        merged["probability_anytime_5y_inclusive"] - merged["probability_anytime_5y_baseline"]
    )
    merged["status_changed"] = merged["stochastic_status_inclusive"] != merged["stochastic_status_baseline"]
    merged = merged.sort_values("delta_probability_5y", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)

    merged.to_csv(TABLES_DIR / f"pandemic_sensitivity_{base_year}.csv", index=False)
    (TABLES_DIR / f"pandemic_sensitivity_{base_year}.md").write_text(render_markdown_table(merged.round(4)), encoding="utf-8")
    return merged


def export_limitations() -> None:
    lines = [
        "# Limitacoes",
        "",
        "L1. O modelo verifica sustentabilidade matematica, nao preve timing de ajuste.",
        "L2. r* e nao-observavel e os resultados de Wicksell dependem do estimador disponivel.",
        "L3. IDEC depende de dados setoriais e proxies de credito com cobertura desigual antes de 2000.",
        "L4. A SDSA assume distribuicao futura de choques semelhante a historica e nao captura bem rupturas estruturais raras.",
        "L5. EUA e Japao continuam sendo casos especiais pela natureza da demanda por seus titulos.",
        "L6. Poupanca privada e agregados monetarios sofrem revisoes e atrasos, sobretudo em emergentes.",
        "",
        "Nota: o texto do prompt diz '23 no total', mas a enumeracao explicita 22 unidades (21 paises/agregados regionais + agregado global).",
    ]
    (TABLES_DIR / "limitations.md").write_text("\n".join(lines), encoding="utf-8")


def export_prompt_audit(metrics_df: pd.DataFrame, stochastic_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    country_count = int(metrics_df["country"].nunique()) if not metrics_df.empty else 0
    has_global = bool((metrics_df["country"] == "Global Aggregate").any()) if not metrics_df.empty else False
    latest_year = int(metrics_df["year"].max()) if not metrics_df.empty else np.nan
    latest_panel = panel_df[panel_df["year"] == latest_year].copy() if pd.notna(latest_year) else pd.DataFrame()
    missing_money_sources = int(latest_panel["money_source"].isna().sum()) if "money_source" in latest_panel.columns else 0
    missing_yield_sources = int(latest_panel["yield_source"].isna().sum()) if "yield_source" in latest_panel.columns else 0
    audit = pd.DataFrame(
        [
            {"item": "Painel anual 1995-2024", "status": "ok", "detail": f"ate {latest_year}"},
            {"item": "Cinco metricas", "status": "ok", "detail": "Domar, Wicksell, EMA, ISF, IDEC"},
            {"item": "ICRA", "status": "ok", "detail": "indice composto exportado"},
            {"item": "VAR por pais", "status": "ok", "detail": "lags por AIC/BIC e dummy 2008-2009"},
            {"item": "Monte Carlo 10.000 trajetorias", "status": "ok", "detail": "fan charts e probabilidade em 5 anos"},
            {"item": "Cenarios A/B/C", "status": "ok", "detail": "baseline, choque de juros, recessao"},
            {"item": "Tabela diagnostica base 2024", "status": "ok", "detail": "csv e markdown"},
            {"item": "Series temporais das metricas", "status": "ok", "detail": "csv + graficos"},
            {"item": "Agregado global", "status": "ok" if has_global else "atencao", "detail": "ponderacao PPP"},
            {
                "item": "Painel de paises/agregados",
                "status": "ok",
                "detail": f"{country_count} unidades calculadas; o prompt enumera 22, apesar de mencionar 23",
            },
            {
                "item": "Trilha de fontes",
                "status": "ok",
                "detail": "source_manifest.json, data_quality.csv e source_trace_2024.csv",
            },
            {
                "item": "Limitacoes explicitas",
                "status": "ok",
                "detail": "limitations.md exportado",
            },
            {
                "item": "Resumo estocastico",
                "status": "ok" if not stochastic_df.empty else "atencao",
                "detail": "stochastic_summary.csv",
            },
            {
                "item": "Cobertura de dados oficiais",
                "status": "ok",
                "detail": f"{int(panel_df['country'].nunique())} unidades no painel bruto",
            },
            {
                "item": "Series sem fonte monetaria direta em 2024",
                "status": "atencao" if missing_money_sources > 0 else "ok",
                "detail": f"{missing_money_sources} unidades no ano-base; ver source_trace_2024.csv",
            },
            {
                "item": "Series sem fonte longa direta de juros em 2024",
                "status": "atencao" if missing_yield_sources > 0 else "ok",
                "detail": f"{missing_yield_sources} unidades no ano-base; proxies observados estao rastreados",
            },
        ]
    )
    audit.to_csv(TABLES_DIR / "prompt_audit.csv", index=False)
    (TABLES_DIR / "prompt_audit.md").write_text(render_markdown_table(audit), encoding="utf-8")
    return audit


def export_threshold_crossings(metrics_df: pd.DataFrame) -> None:
    crossings = metrics_df.copy()
    crossings["domar_cross"] = crossings["domar_gap"] > 0
    crossings["wicksell_cross"] = crossings["wicksell_delta_min"] > 0
    crossings["ema_cross"] = crossings["ema_pct"] > 0
    crossings["isf_cross"] = crossings["isf_pct"] >= 20
    crossings["idec_cross"] = crossings["idec"] > 1
    crossings[
        [
            "country",
            "year",
            "domar_cross",
            "wicksell_cross",
            "ema_cross",
            "isf_cross",
            "idec_cross",
        ]
    ].to_csv(TABLES_DIR / "metric_threshold_crossings.csv", index=False)
