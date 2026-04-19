from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from config import (
    DEFAULT_INFLATION_TARGET,
    DEFAULT_THRESHOLD_ADVANCED,
    DEFAULT_THRESHOLD_EMERGING,
    FRED_SERIES,
    IMF_INDICATORS,
    INFLATION_TARGETS,
    OBSERVED_END_YEAR,
    OECD_DIRECT_IDEC_CODES,
    OECD_SENSITIVE_ACTIVITY_PREFIXES,
    PANEL_COUNTRIES,
    PANEL_COUNTRY_NAMES,
    QUALITY_FILE,
    QUARTERLY_FILE,
    RAW_DIR,
    RAW_SOURCES_DIR,
    SOURCE_URLS,
    START_YEAR,
    WORLD_BANK_INDICATORS,
)
from models.rstar import compute_favar_estimates


REQUEST_HEADERS = {
    "User-Agent": "mvfa/0.1 (+research automation)",
}

IMF_SELECTED_CODES = sorted({meta["imf"] for meta in PANEL_COUNTRIES.values()})
WB_SELECTED_CODES = sorted({meta["wb"] for meta in PANEL_COUNTRIES.values()})
BIS_SELECTED_CODES = sorted({meta["bis"] for meta in PANEL_COUNTRIES.values()})
IMF_CODE_TO_COUNTRY = {meta["imf"]: country for country, meta in PANEL_COUNTRIES.items()}
WB_CODE_TO_COUNTRY = {meta["wb"]: country for country, meta in PANEL_COUNTRIES.items()}
BIS_CODE_TO_COUNTRY = {meta["bis"]: country for country, meta in PANEL_COUNTRIES.items()}
OECD_CODE_TO_COUNTRY = {
    meta["oecd"]: country
    for country, meta in PANEL_COUNTRIES.items()
    if meta.get("oecd") is not None
}
WEO_ISO_TO_COUNTRY = {
    "USA": "United States",
    "JPN": "Japan",
    "GBR": "United Kingdom",
    "CAN": "Canada",
    "AUS": "Australia",
    "KOR": "South Korea",
    "CHN": "China",
    "BRA": "Brazil",
    "IND": "India",
    "MEX": "Mexico",
    "TUR": "Turkey",
    "IDN": "Indonesia",
    "SAU": "Saudi Arabia",
    "ARG": "Argentina",
    "GRC": "Greece",
    "PRT": "Portugal",
    "ITA": "Italy",
    "ESP": "Spain",
    "IRL": "Ireland",
    "LBN": "Lebanon",
}
WEO_SUBJECT_MAP = {
    "NGDP_R": "gdp_real_level_weo",
    "NGDP_RPCH": "gdp_real_growth",
    "NGDPD": "gdp_nominal_weo",
    "PPPGDP": "ppp_gdp",
    "NGDP_D": "gdp_deflator_index_weo",
    "GGR_NGDP": "tax_revenue_gdp",
    "GGXCNL_NGDP": "overall_balance_gdp",
    "GGXONLB_NGDP": "primary_balance_gdp",
    "GGXWDG_NGDP": "debt_gdp",
}
OECD_SERIES_CODES = {
    "United States": "USA",
    "Euro Area": "EA20",
    "Japan": "JPN",
    "United Kingdom": "GBR",
    "Canada": "CAN",
    "Australia": "AUS",
    "South Korea": "KOR",
    "China": "CHN",
    "Brazil": "BRA",
    "India": "IND",
    "Mexico": "MEX",
    "Turkey": "TUR",
    "Indonesia": "IDN",
    "Saudi Arabia": "SAU",
    "Argentina": "ARG",
    "Greece": "GRC",
    "Portugal": "PRT",
    "Italy": "ITA",
    "Spain": "ESP",
    "Ireland": "IRL",
    "Lebanon": "LBN",
}
OECD_CODE_TO_COUNTRY_ALL = {code: country for country, code in OECD_SERIES_CODES.items()}
EURO_AREA_SHARED_POLICY_COUNTRIES = {"Greece", "Ireland", "Italy", "Portugal", "Spain"}
OECD_MONETARY_CODES = {
    country: code
    for country, code in OECD_SERIES_CODES.items()
    if country not in {"Argentina", "Saudi Arabia", "Lebanon"}
}
OECD_LONG_RATE_CODES = {
    country: code
    for country, code in OECD_SERIES_CODES.items()
    if country not in {"Argentina", "Saudi Arabia", "Lebanon"}
}
BCRA_VARIABLES = {
    "m2_nominal": 1233,
    "market_rate_proxy": 139,
}


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SOURCES_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_text(url: str, destination: Path, refresh: bool = False, headers: dict[str, str] | None = None) -> str:
    _ensure_dirs()
    if destination.exists() and not refresh:
        return destination.read_text(encoding="utf-8")

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=120, headers=headers or REQUEST_HEADERS)
            response.raise_for_status()
            destination.write_text(response.text, encoding="utf-8")
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(2 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Falha inesperada ao baixar {url}")


def _fetch_bytes(url: str, destination: Path, refresh: bool = False) -> bytes:
    _ensure_dirs()
    if destination.exists() and not refresh:
        return destination.read_bytes()

    response = requests.get(url, timeout=120, headers=REQUEST_HEADERS)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return response.content


def _fetch_json(url: str, destination: Path, refresh: bool = False) -> dict:
    text = _fetch_text(url, destination, refresh=refresh, headers=REQUEST_HEADERS)
    return json.loads(text)


def _fetch_csv(url: str, destination: Path, refresh: bool = False) -> pd.DataFrame:
    text = _fetch_text(url, destination, refresh=refresh, headers=REQUEST_HEADERS)
    return pd.read_csv(io.StringIO(text))


def _optional_marker_path(destination: Path) -> Path:
    return destination.with_suffix(f"{destination.suffix}.missing")


def _fetch_csv_optional(url: str, destination: Path, refresh: bool = False) -> pd.DataFrame:
    _ensure_dirs()
    marker = _optional_marker_path(destination)
    if marker.exists() and not refresh:
        return pd.DataFrame()
    if destination.exists() and not refresh:
        text = destination.read_text(encoding="utf-8")
        return pd.read_csv(io.StringIO(text))

    cached_text = destination.read_text(encoding="utf-8") if destination.exists() else None
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=120, headers=REQUEST_HEADERS)
            if response.status_code == 404 or "NoRecordsFound" in response.text:
                marker.write_text("", encoding="utf-8")
                return pd.DataFrame()
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                if cached_text is not None and attempt >= 2:
                    return pd.read_csv(io.StringIO(cached_text))
                time.sleep(max(retry_after, 2 * (attempt + 1)))
                continue
            response.raise_for_status()
            destination.write_text(response.text, encoding="utf-8")
            if marker.exists():
                marker.unlink()
            return pd.read_csv(io.StringIO(response.text))
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    if cached_text is not None:
        return pd.read_csv(io.StringIO(cached_text))
    if marker.exists():
        return pd.DataFrame()
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Falha inesperada ao baixar {url}")


def fetch_weo_panel(refresh: bool = False) -> pd.DataFrame:
    destination = RAW_SOURCES_DIR / "imf_weo_country_apr2025.tsv"
    text = _fetch_text(SOURCE_URLS["imf_weo_country_tsv"], destination, refresh=refresh, headers=REQUEST_HEADERS)
    raw = pd.read_csv(io.StringIO(text), sep="\t", encoding="utf-16le")
    raw = raw[raw["ISO"].isin(WEO_ISO_TO_COUNTRY)].copy()
    raw["country"] = raw["ISO"].map(WEO_ISO_TO_COUNTRY)
    raw = raw[raw["WEO Subject Code"].isin(WEO_SUBJECT_MAP)].copy()

    year_columns = [column for column in raw.columns if re.fullmatch(r"\d{4}", str(column))]
    panel = (
        raw.melt(
            id_vars=["country", "WEO Subject Code"],
            value_vars=year_columns,
            var_name="year",
            value_name="value",
        )
        .assign(
            year=lambda df: df["year"].astype(int),
            value=lambda df: pd.to_numeric(df["value"], errors="coerce"),
            indicator=lambda df: df["WEO Subject Code"].map(WEO_SUBJECT_MAP),
        )
        .dropna(subset=["value"])
        .pivot_table(index=["country", "year"], columns="indicator", values="value", aggfunc="first")
        .reset_index()
    )

    panel["gdp_deflator_inflation_weo"] = (
        panel.groupby("country")["gdp_deflator_index_weo"].pct_change(fill_method=None) * 100.0
    )
    panel["interest_paid_gdp"] = panel["primary_balance_gdp"] - panel["overall_balance_gdp"]
    return panel


def _extract_eurostat_series(payload: dict, value_name: str) -> pd.DataFrame:
    time_index = payload["dimension"]["time"]["category"]["index"]
    values = payload.get("value", {})
    rows: list[dict[str, object]] = []
    for year_str, offset in time_index.items():
        key = str(offset)
        if key not in values:
            continue
        rows.append(
            {
                "country": "Euro Area",
                "year": int(year_str),
                value_name: float(values[key]),
            }
        )
    return pd.DataFrame(rows)


def fetch_euro_area_fiscal_panel(refresh: bool = False) -> pd.DataFrame:
    base_url = SOURCE_URLS["eurostat_api"]
    queries = {
        "debt_gdp": "gov_10dd_edpt1?geo=EA20&unit=PC_GDP&sector=S13&na_item=GD",
        "overall_balance_gdp": "gov_10a_main?geo=EA20&unit=PC_GDP&sector=S13&na_item=B9",
        "tax_revenue_gdp": "gov_10a_main?geo=EA20&unit=PC_GDP&sector=S13&na_item=TR",
        "interest_paid_gdp": "gov_10a_main?geo=EA20&unit=PC_GDP&sector=S13&na_item=D41PAY",
    }

    panel: pd.DataFrame | None = None
    for name, query in queries.items():
        destination = RAW_SOURCES_DIR / f"eurostat_{name}.json"
        payload = _fetch_json(f"{base_url}{query}", destination, refresh=refresh)
        frame = _extract_eurostat_series(payload, name)
        panel = frame if panel is None else panel.merge(frame, on=["country", "year"], how="outer")

    if panel is None:
        return pd.DataFrame(columns=["country", "year"])
    panel["primary_balance_gdp"] = panel["overall_balance_gdp"] + panel["interest_paid_gdp"]
    return panel


def _parse_oecd_country_series(
    flow_id: str,
    code: str,
    measure: str,
    refresh: bool,
) -> pd.DataFrame:
    url = (
        f"{SOURCE_URLS['oecd_api']}data/OECD.SDD.STES,{flow_id}/{code}.A.{measure}........"
        f"?startPeriod={START_YEAR}&endPeriod={OBSERVED_END_YEAR}&format=csvfilewithlabels"
    )
    destination = RAW_SOURCES_DIR / f"oecd_{flow_id.replace('@', '_')}_{code}_{measure}.csv"
    if not refresh and not destination.exists() and not _optional_marker_path(destination).exists():
        return pd.DataFrame()
    return _fetch_csv_optional(url, destination, refresh=refresh)


def fetch_oecd_monetary_aggregates(refresh: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for country, code in OECD_MONETARY_CODES.items():
        frame = _parse_oecd_country_series("DSD_STES@DF_MONAGG", code, "MABM", refresh=refresh)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["TIME_PERIOD"] = pd.to_numeric(frame["TIME_PERIOD"], errors="coerce")
        frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
        frame = frame.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
        frame = frame[
            frame["UNIT_MEASURE"].eq("XDC")
            & frame["ACTIVITY"].eq("_Z")
            & frame["TRANSFORMATION"].eq("_Z")
            & frame["TIME_HORIZ"].eq("_Z")
            & frame["METHODOLOGY"].eq("N")
        ].copy()
        if frame.empty:
            continue
        frame["adj_rank"] = frame["ADJUSTMENT"].map({"Y": 0, "N": 1}).fillna(2)
        frame = frame.sort_values(["TIME_PERIOD", "adj_rank"]).drop_duplicates(["TIME_PERIOD"], keep="first")
        rows.append(
            frame.assign(country=country, year=lambda df: df["TIME_PERIOD"].astype(int))[["country", "year", "OBS_VALUE"]]
            .rename(columns={"OBS_VALUE": "m3_nominal_oecd"})
        )

    if not rows:
        return pd.DataFrame(columns=["country", "year", "m3_nominal_oecd"])
    return pd.concat(rows, ignore_index=True)


def fetch_oecd_long_rates(refresh: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for country, code in OECD_LONG_RATE_CODES.items():
        frame = _parse_oecd_country_series("DSD_STES@DF_FINMARK", code, "IRLT", refresh=refresh)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["TIME_PERIOD"] = pd.to_numeric(frame["TIME_PERIOD"], errors="coerce")
        frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
        frame = frame.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
        frame = frame[
            frame["UNIT_MEASURE"].eq("PA")
            & frame["ACTIVITY"].eq("_Z")
            & frame["ADJUSTMENT"].eq("_Z")
            & frame["TRANSFORMATION"].eq("_Z")
            & frame["TIME_HORIZ"].eq("_Z")
            & frame["METHODOLOGY"].eq("N")
        ].copy()
        if frame.empty:
            continue
        rows.append(
            frame.assign(country=country, year=lambda df: df["TIME_PERIOD"].astype(int))[["country", "year", "OBS_VALUE"]]
            .rename(columns={"OBS_VALUE": "yield_10y_nominal_oecd"})
        )

    if not rows:
        return pd.DataFrame(columns=["country", "year", "yield_10y_nominal_oecd"])
    return pd.concat(rows, ignore_index=True)


def fetch_bis_total_credit(refresh: bool = False) -> pd.DataFrame:
    key = f"Q.{'+'.join(BIS_SELECTED_CODES)}.H+P.A.M.770.A"
    url = f"{SOURCE_URLS['bis_data_api']}data/dataflow/BIS/WS_TC/2.0/{key}?startPeriod={START_YEAR}-Q1&endPeriod={OBSERVED_END_YEAR}-Q4"
    destination = RAW_SOURCES_DIR / "bis_total_credit_household_private.xml"

    _ensure_dirs()
    if destination.exists() and not refresh:
        xml_text = destination.read_text(encoding="utf-8")
    else:
        response = requests.get(url, timeout=120, headers=REQUEST_HEADERS)
        response.raise_for_status()
        destination.write_text(response.text, encoding="utf-8")
        xml_text = response.text

    series_pattern = re.compile(
        r'<Series[^>]*BORROWERS_CTY="(?P<country>[A-Z]{2})"[^>]*TC_BORROWERS="(?P<borrower>[HP])"'
        r'[^>]*>(?P<body>.*?)</Series>',
        re.DOTALL,
    )
    obs_pattern = re.compile(r'<Obs[^>]*TIME_PERIOD="(?P<period>\d{4})-Q[1-4]"[^>]*OBS_VALUE="(?P<value>[-0-9.]+)"')

    rows: list[dict[str, object]] = []
    for match in series_pattern.finditer(xml_text):
        bis_code = match.group("country")
        borrower = match.group("borrower")
        country = BIS_CODE_TO_COUNTRY.get(bis_code)
        if country is None:
            continue
        for obs in obs_pattern.finditer(match.group("body")):
            rows.append(
                {
                    "country": country,
                    "year": int(obs.group("period")),
                    "borrower": borrower,
                    "value": float(obs.group("value")),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["country", "year", "household_debt_gdp", "private_debt_gdp"])

    quarterly = pd.DataFrame(rows)
    annual = quarterly.groupby(["country", "year", "borrower"], as_index=False)["value"].mean()
    annual = annual.pivot_table(index=["country", "year"], columns="borrower", values="value", aggfunc="first").reset_index()
    return annual.rename(columns={"H": "household_debt_gdp", "P": "private_debt_gdp"})


def _parse_bdl_history_link_date(link: str) -> tuple[int, int, int] | None:
    filename = Path(link).name
    match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Za-z]*(\d{1,2})-(\d{4})", filename, re.IGNORECASE)
    if not match:
        return None
    month_token = match.group(1)[:3].upper()
    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    if month_token not in month_map:
        return None
    return int(match.group(3)), month_map[month_token], int(match.group(2))


def fetch_bdl_money_supply(refresh: bool = False) -> pd.DataFrame:
    destination = RAW_SOURCES_DIR / "bdl_money_supply_history.html"
    html = _fetch_text(SOURCE_URLS["bdl_money_supply_history"], destination, refresh=refresh, headers=REQUEST_HEADERS)
    links = re.findall(r'href="([^"]+\.xlsx)"', html, flags=re.IGNORECASE)
    candidates: dict[int, str] = {}
    for link in links:
        parsed = _parse_bdl_history_link_date(link)
        if parsed is None:
            continue
        year, month, day = parsed
        if year < 2018 or year > OBSERVED_END_YEAR:
            continue
        score = month * 100 + day
        existing = candidates.get(year)
        if existing is None:
            candidates[year] = link
            continue
        existing_parsed = _parse_bdl_history_link_date(existing)
        if existing_parsed is None:
            candidates[year] = link
            continue
        existing_score = existing_parsed[1] * 100 + existing_parsed[2]
        if score > existing_score:
            candidates[year] = link

    rows: list[dict[str, object]] = []
    for year, link in sorted(candidates.items()):
        file_name = Path(link).name
        file_destination = RAW_SOURCES_DIR / f"bdl_{file_name}"
        content = _fetch_bytes(urljoin(SOURCE_URLS["bdl_base_url"], link), file_destination, refresh=refresh)
        book = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
        if book.empty:
            continue
        labels = book.iloc[:, 0].astype(str).str.strip().str.upper()
        for target, value_name in [("M 2", "m2_nominal_bdl"), ("M 3", "m3_nominal_bdl")]:
            if target not in set(labels):
                continue
            row = book.loc[labels == target].iloc[0]
            numeric_values = pd.to_numeric(row.iloc[1:-1], errors="coerce").dropna()
            if numeric_values.empty:
                continue
            value = numeric_values.iloc[-1]
            if pd.isna(value):
                continue
            rows.append({"country": "Lebanon", "year": year, value_name: float(value) * 1_000_000_000.0})

    if not rows:
        return pd.DataFrame(columns=["country", "year", "m2_nominal_bdl", "m3_nominal_bdl"])

    panel = pd.DataFrame(rows)
    return panel.groupby(["country", "year"], as_index=False).first()


def fetch_bcra_series(variable_id: int, refresh: bool = False) -> pd.DataFrame:
    destination = RAW_SOURCES_DIR / f"bcra_variable_{variable_id}.json"
    start_date = f"{START_YEAR}-01-01"
    end_date = f"{OBSERVED_END_YEAR}-12-31"
    base_url = (
        f"{SOURCE_URLS['bcra_api']}monetarias/{variable_id}"
        f"?desde={start_date}&hasta={end_date}&limit=3000"
    )

    _ensure_dirs()
    if destination.exists() and not refresh:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    else:
        offset = 0
        pages: list[dict] = []
        while True:
            response = requests.get(f"{base_url}&offset={offset}", timeout=120, headers=REQUEST_HEADERS)
            response.raise_for_status()
            payload_page = response.json()
            pages.append(payload_page)
            page_results = payload_page.get("results", [])
            detail = page_results[0].get("detalle", []) if page_results else []
            if len(detail) < 3000:
                break
            offset += 3000
            time.sleep(0.2)
        payload = {"pages": pages}
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    pages = payload.get("pages", [])
    rows: list[dict[str, object]] = []
    for page in pages:
        page_results = page.get("results", [])
        if not page_results:
            continue
        for item in page_results[0].get("detalle", []):
            rows.append(
                {
                    "date": pd.to_datetime(item["fecha"]),
                    "value": pd.to_numeric(item["valor"], errors="coerce"),
                }
            )
    frame = pd.DataFrame(rows).dropna(subset=["date", "value"])
    if frame.empty:
        return pd.DataFrame(columns=["date", "value"])
    return frame.sort_values("date").reset_index(drop=True)


def fetch_bcra_panel(refresh: bool = False) -> pd.DataFrame:
    m2_series = fetch_bcra_series(BCRA_VARIABLES["m2_nominal"], refresh=refresh)
    rate_series = fetch_bcra_series(BCRA_VARIABLES["market_rate_proxy"], refresh=refresh)
    if m2_series.empty and rate_series.empty:
        return pd.DataFrame(columns=["country", "year", "m2_nominal_bcra", "market_rate_proxy_bcra"])

    frames: list[pd.DataFrame] = []
    if not m2_series.empty:
        m2_series["year"] = m2_series["date"].dt.year
        m2_annual = (
            m2_series.sort_values("date")
            .groupby("year", as_index=False)
            .tail(1)[["year", "value"]]
            .rename(columns={"value": "m2_nominal_bcra"})
        )
        m2_annual["m2_nominal_bcra"] = m2_annual["m2_nominal_bcra"] * 1_000_000.0
        frames.append(m2_annual)

    if not rate_series.empty:
        rate_series["year"] = rate_series["date"].dt.year
        rate_annual = (
            rate_series.groupby("year", as_index=False)["value"]
            .mean()
            .rename(columns={"value": "market_rate_proxy_bcra"})
        )
        frames.append(rate_annual)

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="year", how="outer")
    panel["country"] = "Argentina"
    return panel[["country", "year", *[column for column in panel.columns if column not in {"country", "year"}]]]


def fetch_imf_indicator(indicator: str, start_year: int, end_year: int, refresh: bool = False) -> pd.DataFrame:
    years = ",".join(str(year) for year in range(start_year, end_year + 1))
    rows: list[dict[str, object]] = []
    for code in IMF_SELECTED_CODES:
        url = f"{SOURCE_URLS['imf_datamapper_api']}{indicator}/{code}?periods={years}"
        destination = RAW_SOURCES_DIR / f"imf_{indicator}_{code}_{start_year}_{end_year}.json"
        payload = _fetch_json(url, destination, refresh=refresh)
        series = payload["values"][indicator].get(code)
        if not isinstance(series, dict):
            continue
        for year_str, value in series.items():
            if value is None:
                continue
            rows.append(
                {
                    "country": IMF_CODE_TO_COUNTRY[code],
                    "year": int(year_str),
                    indicator: float(value),
                }
            )
    return pd.DataFrame(rows)


def build_imf_panel(start_year: int, end_year: int, refresh: bool = False) -> pd.DataFrame:
    panel = fetch_weo_panel(refresh=refresh)
    return panel[panel["year"].between(start_year, end_year)].copy()


def fetch_world_bank_indicator(indicator: str, refresh: bool = False) -> pd.DataFrame:
    code_string = ";".join(WB_SELECTED_CODES)
    url = (
        f"{SOURCE_URLS['world_bank_api']}country/{code_string}/indicator/{indicator}"
        "?format=json&per_page=20000"
    )
    destination = RAW_SOURCES_DIR / f"wb_{indicator}.json"
    payload = _fetch_json(url, destination, refresh=refresh)
    rows = payload[1]

    output: list[dict[str, object]] = []
    for row in rows:
        year = row.get("date")
        value = row.get("value")
        code = row.get("countryiso3code")
        if value is None or code not in WB_CODE_TO_COUNTRY:
            continue
        output.append(
            {
                "country": WB_CODE_TO_COUNTRY[code],
                "year": int(year),
                indicator: float(value),
            }
        )
    return pd.DataFrame(output)


def build_world_bank_panel(refresh: bool = False) -> pd.DataFrame:
    frames = [
        fetch_world_bank_indicator(indicator_code, refresh=refresh)
        for indicator_code in WORLD_BANK_INDICATORS.values()
    ]
    panel = None
    reverse_map = {value: key for key, value in WORLD_BANK_INDICATORS.items()}
    for frame in frames:
        if frame.empty:
            continue
        value_column = next(column for column in frame.columns if column not in {"country", "year"})
        frame = frame.rename(columns={value_column: reverse_map[value_column]})
        if panel is None:
            panel = frame
        else:
            panel = panel.merge(frame, on=["country", "year"], how="outer")
    return panel if panel is not None else pd.DataFrame(columns=["country", "year"])


def fetch_bis_policy_rates(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"M.{'+'.join(BIS_SELECTED_CODES)}"
    url = f"{SOURCE_URLS['bis_data_api']}data/dataflow/BIS/WS_CBPOL/1.0/{key}?format=csvfile"
    destination = RAW_SOURCES_DIR / "bis_cbpol_monthly.csv"
    monthly = _fetch_csv(url, destination, refresh=refresh)
    monthly["country"] = monthly["REF_AREA"].map(BIS_CODE_TO_COUNTRY)
    monthly["TIME_PERIOD"] = pd.to_datetime(monthly["TIME_PERIOD"], format="%Y-%m")
    monthly["OBS_VALUE"] = pd.to_numeric(monthly["OBS_VALUE"], errors="coerce")
    monthly = monthly.dropna(subset=["country", "OBS_VALUE"])
    monthly["year"] = monthly["TIME_PERIOD"].dt.year
    annual = (
        monthly.groupby(["country", "year"], as_index=False)["OBS_VALUE"]
        .mean()
        .rename(columns={"OBS_VALUE": "policy_rate_nominal"})
    )
    return monthly, annual


def fetch_bis_eer(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"M.R.B.{'+'.join(BIS_SELECTED_CODES)}"
    url = f"{SOURCE_URLS['bis_data_api']}data/dataflow/BIS/WS_EER/1.0/{key}?format=csvfile"
    destination = RAW_SOURCES_DIR / "bis_eer_monthly.csv"
    monthly = _fetch_csv(url, destination, refresh=refresh)
    monthly["country"] = monthly["REF_AREA"].map(BIS_CODE_TO_COUNTRY)
    monthly["TIME_PERIOD"] = pd.to_datetime(monthly["TIME_PERIOD"], format="%Y-%m")
    monthly["OBS_VALUE"] = pd.to_numeric(monthly["OBS_VALUE"], errors="coerce")
    monthly = monthly.dropna(subset=["country", "OBS_VALUE"])
    monthly["year"] = monthly["TIME_PERIOD"].dt.year
    annual_level = (
        monthly.groupby(["country", "year"], as_index=False)["OBS_VALUE"]
        .mean()
        .rename(columns={"OBS_VALUE": "real_effective_exchange_rate"})
    )
    annual_level["fx_real_change"] = annual_level.groupby("country")["real_effective_exchange_rate"].pct_change() * 100.0
    annual = annual_level[["country", "year", "fx_real_change"]]
    return monthly, annual


def fetch_nyfed_hlw(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = RAW_SOURCES_DIR / "nyfed_hlw_current_estimates.xlsx"
    _fetch_bytes(SOURCE_URLS["nyfed_hlw_xlsx"], destination, refresh=refresh)
    raw = pd.read_excel(destination, sheet_name="HLW Estimates", header=4)
    renamed = raw.rename(
        columns={
            raw.columns[0]: "date",
            raw.columns[10]: "US",
            raw.columns[11]: "Canada",
            raw.columns[12]: "Euro Area",
        }
    )[["date", "US", "Canada", "Euro Area"]]
    renamed["date"] = pd.to_datetime(renamed["date"], format="mixed", errors="coerce")
    quarterly = (
        renamed.melt(id_vars="date", var_name="country", value_name="natural_rate_hlw")
        .dropna(subset=["date", "natural_rate_hlw"])
        .assign(
            country=lambda df: df["country"].replace({"US": "United States"}),
            year=lambda df: df["date"].dt.year,
        )
    )
    annual = (
        quarterly.groupby(["country", "year"], as_index=False)["natural_rate_hlw"]
        .mean()
    )
    return quarterly, annual


def fetch_richmond_lm(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = RAW_SOURCES_DIR / "richmond_lm.html"
    html = _fetch_text(SOURCE_URLS["richmond_lm_page"], destination, refresh=refresh)
    start = html.find("dates,lower16,median,upper84")
    if start == -1:
        raise ValueError("Nao foi possivel localizar o CSV embutido do Richmond Fed.")
    text_block = html[start:].split("<", 1)[0]
    quarterly = pd.read_csv(io.StringIO(text_block))
    quarterly["date"] = pd.to_datetime(quarterly["dates"], format="%d-%b-%y")
    quarterly["country"] = "United States"
    quarterly["year"] = quarterly["date"].dt.year
    quarterly = quarterly.rename(columns={"median": "natural_rate_lm"})
    annual = quarterly.groupby(["country", "year"], as_index=False)["natural_rate_lm"].mean()
    return quarterly[["country", "date", "natural_rate_lm", "year"]], annual


def fetch_fred_series(refresh: bool = False) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for series_id, filename in FRED_SERIES.items():
        destination = RAW_SOURCES_DIR / filename
        if destination.exists():
            frames[series_id] = pd.read_csv(destination)
    return frames


def build_quarterly_panel(
    bis_policy_monthly: pd.DataFrame,
    bis_eer_monthly: pd.DataFrame,
    hlw_quarterly: pd.DataFrame,
    lm_quarterly: pd.DataFrame,
    fred_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    policy_q = bis_policy_monthly.copy()
    policy_q["quarter"] = policy_q["TIME_PERIOD"].dt.to_period("Q").astype(str)
    policy_q = (
        policy_q.groupby(["country", "quarter"], as_index=False)["OBS_VALUE"]
        .mean()
        .rename(columns={"OBS_VALUE": "policy_rate_nominal"})
    )

    eer_q = bis_eer_monthly.copy()
    eer_q["quarter"] = eer_q["TIME_PERIOD"].dt.to_period("Q").astype(str)
    eer_q = (
        eer_q.groupby(["country", "quarter"], as_index=False)["OBS_VALUE"]
        .mean()
        .rename(columns={"OBS_VALUE": "real_effective_exchange_rate"})
    )
    eer_q["fx_real_change_qoq"] = eer_q.groupby("country")["real_effective_exchange_rate"].pct_change() * 100.0

    hlw_q = hlw_quarterly.copy()
    hlw_q["quarter"] = hlw_q["date"].dt.to_period("Q").astype(str)
    hlw_q = hlw_q[["country", "quarter", "natural_rate_hlw"]]

    lm_q = lm_quarterly.copy()
    lm_q["quarter"] = lm_q["date"].dt.to_period("Q").astype(str)
    lm_q = lm_q[["country", "quarter", "natural_rate_lm"]]

    quarter_panel = policy_q.merge(eer_q, on=["country", "quarter"], how="outer")
    quarter_panel = quarter_panel.merge(hlw_q, on=["country", "quarter"], how="outer")
    quarter_panel = quarter_panel.merge(lm_q, on=["country", "quarter"], how="outer")

    us_fred = None
    if {"FEDFUNDS", "M2SL", "GDPC1", "PCEPI"}.issubset(fred_frames):
        def _prep_fred(series_id: str, value_name: str) -> pd.DataFrame:
            frame = fred_frames[series_id].copy()
            frame["date"] = pd.to_datetime(frame["observation_date"])
            frame[value_name] = pd.to_numeric(frame[series_id], errors="coerce")
            frame["quarter"] = frame["date"].dt.to_period("Q").astype(str)
            return frame.groupby("quarter", as_index=False)[value_name].mean()

        us_fred = _prep_fred("FEDFUNDS", "fred_fedfunds")
        us_fred = us_fred.merge(_prep_fred("M2SL", "fred_m2"), on="quarter", how="outer")
        us_fred = us_fred.merge(_prep_fred("GDPC1", "fred_real_gdp"), on="quarter", how="outer")
        us_fred = us_fred.merge(_prep_fred("PCEPI", "fred_pcepi"), on="quarter", how="outer")
        us_fred["country"] = "United States"
        quarter_panel = quarter_panel.merge(us_fred, on=["country", "quarter"], how="outer")

    quarter_panel.to_csv(QUARTERLY_FILE, index=False)
    return quarter_panel


def fetch_oecd_table8(refresh: bool = False) -> pd.DataFrame:
    if not OECD_DIRECT_IDEC_CODES:
        return pd.DataFrame(columns=["country", "year", "investment_total_nominal", "investment_sensitive_nominal"])

    key = ".".join(
        [
            "A",
            "+".join(OECD_DIRECT_IDEC_CODES),
            "S1",
            "S1",
            "P51G",
            "N11G",
            "",
            "_Z",
            "XDC",
            "V",
            "N",
            "T0302",
        ]
    )
    url = (
        "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE8/"
        f"{key}?startPeriod={START_YEAR}&endPeriod={OBSERVED_END_YEAR}"
        "&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
    )
    destination = RAW_SOURCES_DIR / "oecd_table8_current_prices.csv"
    df = _fetch_csv(url, destination, refresh=refresh)
    if df.empty:
        return pd.DataFrame(columns=["country", "year", "investment_total_nominal", "investment_sensitive_nominal"])

    df["country"] = df["REF_AREA"].map(OECD_CODE_TO_COUNTRY)
    df["year"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce").astype("Int64")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["country", "year", "OBS_VALUE"])

    total = (
        df[df["ACTIVITY"] == "_T"]
        .groupby(["country", "year"], as_index=False)["OBS_VALUE"]
        .sum()
        .rename(columns={"OBS_VALUE": "investment_total_nominal"})
    )
    sensitive = (
        df[df["ACTIVITY"].isin(OECD_SENSITIVE_ACTIVITY_PREFIXES)]
        .groupby(["country", "year"], as_index=False)["OBS_VALUE"]
        .sum()
        .rename(columns={"OBS_VALUE": "investment_sensitive_nominal"})
    )
    return total.merge(sensitive, on=["country", "year"], how="outer")


def _build_source_quality(panel: pd.DataFrame) -> pd.DataFrame:
    tracked_columns = [
        "gdp_nominal",
        "gdp_real",
        "gdp_real_growth",
        "gdp_deflator_inflation",
        "debt_gdp",
        "primary_balance_gdp",
        "interest_paid_nominal",
        "policy_rate_nominal",
        "yield_10y_nominal",
        "m2_nominal",
        "private_savings_gdp",
        "investment_total_nominal",
        "investment_sensitive_nominal",
        "natural_rate_hlw",
        "natural_rate_lm",
        "natural_rate_favar",
        "natural_rate_hp",
        "natural_rate_gdp_trend",
    ]
    rows = []
    for country, frame in panel.groupby("country"):
        row = {"country": country}
        for column in tracked_columns:
            row[f"{column}_coverage"] = frame[column].notna().mean()
        rows.append(row)
    quality = pd.DataFrame(rows).sort_values("country")
    quality.to_csv(QUALITY_FILE, index=False)
    return quality


def _apply_yield_fallbacks(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["yield_source"] = None
    if "yield_10y_nominal_oecd" in work.columns:
        direct_oecd = work["yield_10y_nominal_oecd"].notna()
        work.loc[direct_oecd, "yield_source"] = "oecd_finmark_irlt"
        work["yield_10y_nominal"] = work["yield_10y_nominal_oecd"]
    else:
        work["yield_10y_nominal"] = np.nan

    if "market_rate_proxy_bcra" in work.columns:
        direct_bcra = work["yield_10y_nominal"].isna() & work["market_rate_proxy_bcra"].notna()
        work.loc[direct_bcra, "yield_10y_nominal"] = work.loc[direct_bcra, "market_rate_proxy_bcra"]
        work.loc[direct_bcra, "yield_source"] = "bcra_badlar_proxy"

    if "effective_nominal_rate_raw" in work.columns:
        realized_funding_cost = work["yield_10y_nominal"].isna() & work["effective_nominal_rate_raw"].notna()
        work.loc[realized_funding_cost, "yield_10y_nominal"] = work.loc[realized_funding_cost, "effective_nominal_rate_raw"]
        work.loc[realized_funding_cost, "yield_source"] = "realized_funding_cost_proxy"

    if "lending_rate_wb" in work.columns:
        direct_wb = work["yield_10y_nominal"].isna() & work["lending_rate_wb"].notna()
        work.loc[direct_wb, "yield_10y_nominal"] = work.loc[direct_wb, "lending_rate_wb"]
        work.loc[direct_wb, "yield_source"] = "world_bank_lending_rate_proxy"
    return work


def _apply_shared_euro_area_anchors(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["policy_source"] = np.where(work["policy_rate_nominal"].notna(), "bis_cbpol", None)
    work["hlw_source"] = np.where(work["natural_rate_hlw"].notna(), "nyfed_hlw", None)

    euro_anchor = (
        work[work["country"] == "Euro Area"][["year", "policy_rate_nominal", "natural_rate_hlw"]]
        .rename(
            columns={
                "policy_rate_nominal": "policy_rate_nominal_euro_area",
                "natural_rate_hlw": "natural_rate_hlw_euro_area",
            }
        )
    )
    work = work.merge(euro_anchor, on="year", how="left")

    euro_member_mask = work["country"].isin(EURO_AREA_SHARED_POLICY_COUNTRIES)
    policy_fill = euro_member_mask & work["policy_rate_nominal"].isna() & work["policy_rate_nominal_euro_area"].notna()
    work.loc[policy_fill, "policy_rate_nominal"] = work.loc[policy_fill, "policy_rate_nominal_euro_area"]
    work.loc[policy_fill, "policy_source"] = "euro_area_shared_policy_rate"

    hlw_fill = euro_member_mask & work["natural_rate_hlw"].isna() & work["natural_rate_hlw_euro_area"].notna()
    work.loc[hlw_fill, "natural_rate_hlw"] = work.loc[hlw_fill, "natural_rate_hlw_euro_area"]
    work.loc[hlw_fill, "hlw_source"] = "nyfed_hlw_euro_area_proxy"

    return work.drop(columns=["policy_rate_nominal_euro_area", "natural_rate_hlw_euro_area"])


def _append_global_aggregate(panel: pd.DataFrame) -> pd.DataFrame:
    value_weighted_columns = [
        "gdp_real_growth",
        "gdp_deflator_inflation",
        "debt_gdp",
        "primary_balance_gdp",
        "policy_rate_nominal",
        "yield_10y_nominal",
        "inflation_forward_12m",
        "private_savings_gdp",
        "housing_credit_share",
        "fx_real_change",
        "natural_rate_hlw",
        "natural_rate_lm",
        "natural_rate_favar",
        "natural_rate_hp",
        "natural_rate_gdp_trend",
    ]
    additive_columns = [
        "gdp_nominal",
        "gdp_real",
        "ppp_gdp",
        "debt_nominal",
        "avg_debt_nominal",
        "interest_paid_nominal",
        "tax_revenue_nominal",
        "m2_nominal",
        "investment_total_nominal",
        "investment_sensitive_nominal",
    ]

    rows = []
    base = panel[panel["country"].isin(PANEL_COUNTRY_NAMES)].copy()
    for year, frame in base.groupby("year"):
        weight = frame["ppp_gdp"].fillna(frame["gdp_nominal"]).clip(lower=0.0)
        row: dict[str, object] = {
            "country": "Global Aggregate",
            "year": int(year),
            "reserve_currency_flag": 0,
            "inflation_target": DEFAULT_INFLATION_TARGET,
            "threshold_debt_gdp": DEFAULT_THRESHOLD_ADVANCED,
            "yield_source": "ppp_weighted_aggregate",
            "policy_source": "ppp_weighted_aggregate",
            "hlw_source": "ppp_weighted_aggregate",
            "idec_source": "ppp_weighted_aggregate",
        }
        for column in additive_columns:
            row[column] = frame[column].sum(min_count=1)
        for column in value_weighted_columns:
            valid = frame[[column]].copy()
            valid["weight"] = weight
            valid = valid.dropna(subset=[column, "weight"])
            row[column] = np.average(valid[column], weights=valid["weight"]) if not valid.empty else np.nan
        rows.append(row)

    aggregate = pd.DataFrame(rows)
    return pd.concat([panel, aggregate], ignore_index=True, sort=False)


def _coalesce_suffix_columns(panel: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    work = panel.copy()
    for column in column_names:
        left = f"{column}_x"
        right = f"{column}_y"
        if left in work.columns or right in work.columns:
            left_series = work[left] if left in work.columns else pd.Series(np.nan, index=work.index)
            right_series = work[right] if right in work.columns else pd.Series(np.nan, index=work.index)
            work[column] = left_series.combine_first(right_series)
            drop_columns = [name for name in [left, right] if name in work.columns]
            work = work.drop(columns=drop_columns)
    return work


def build_real_api_panel(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    imf_panel = build_imf_panel(START_YEAR, OBSERVED_END_YEAR, refresh=refresh)
    euro_area_panel = fetch_euro_area_fiscal_panel(refresh=refresh)
    wb_panel = build_world_bank_panel(refresh=refresh)
    wb_panel = wb_panel.rename(
        columns={
            "gdp_nominal": "gdp_nominal_wb",
            "gdp_real": "gdp_real_wb",
            "ppp_gdp_wb": "ppp_gdp_wb",
            "gdp_deflator_inflation": "gdp_deflator_inflation_wb",
            "broad_money_current_lcu": "broad_money_current_lcu_wb",
            "broad_money_gdp": "broad_money_gdp_wb",
            "gross_savings_gdp": "gross_savings_gdp_wb",
            "lending_rate": "lending_rate_wb",
        }
    )
    bis_policy_monthly, bis_policy_annual = fetch_bis_policy_rates(refresh=refresh)
    bis_eer_monthly, bis_eer_annual = fetch_bis_eer(refresh=refresh)
    bis_credit_annual = fetch_bis_total_credit(refresh=refresh)
    bcra_panel = fetch_bcra_panel(refresh=refresh)
    hlw_quarterly, hlw_annual = fetch_nyfed_hlw(refresh=refresh)
    lm_quarterly, lm_annual = fetch_richmond_lm(refresh=refresh)
    fred_frames = fetch_fred_series(refresh=refresh)
    oecd_panel = fetch_oecd_table8(refresh=refresh)
    oecd_monetary = fetch_oecd_monetary_aggregates(refresh=refresh)
    oecd_long_rates = fetch_oecd_long_rates(refresh=refresh)
    bdl_money = fetch_bdl_money_supply(refresh=refresh)

    frames = [
        imf_panel[["country", "year"]],
        euro_area_panel[["country", "year"]],
        wb_panel[["country", "year"]],
        bis_policy_annual[["country", "year"]],
        bis_eer_annual[["country", "year"]],
        bis_credit_annual[["country", "year"]],
        bcra_panel[["country", "year"]],
        hlw_annual[["country", "year"]],
        lm_annual[["country", "year"]],
        oecd_panel[["country", "year"]],
        oecd_monetary[["country", "year"]],
        oecd_long_rates[["country", "year"]],
        bdl_money[["country", "year"]],
    ]
    panel = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(["country", "year"]).reset_index(drop=True)
    for frame in [
        imf_panel,
        euro_area_panel,
        wb_panel,
        bis_policy_annual,
        bis_eer_annual,
        bis_credit_annual,
        bcra_panel,
        hlw_annual,
        lm_annual,
        oecd_panel,
        oecd_monetary,
        oecd_long_rates,
        bdl_money,
    ]:
        panel = panel.merge(frame, on=["country", "year"], how="left")
    panel = _coalesce_suffix_columns(
        panel,
        ["debt_gdp", "tax_revenue_gdp", "interest_paid_gdp", "overall_balance_gdp", "primary_balance_gdp"],
    )

    panel = panel[panel["country"].isin(PANEL_COUNTRY_NAMES)].copy()
    panel = panel[panel["year"].between(START_YEAR, OBSERVED_END_YEAR)].copy()
    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    panel["gdp_nominal"] = panel["gdp_nominal_weo"].where(panel["gdp_nominal_weo"].notna(), panel["gdp_nominal_wb"] / 1_000_000_000.0)
    panel["gdp_real"] = panel["gdp_real_level_weo"].where(panel["gdp_real_level_weo"].notna(), panel["gdp_real_wb"] / 1_000_000_000.0)
    panel["gdp_real_growth"] = panel["gdp_real_growth"].where(
        panel["gdp_real_growth"].notna(),
        panel.groupby("country")["gdp_real"].pct_change(fill_method=None) * 100.0,
    )
    panel["gdp_deflator_inflation"] = panel["gdp_deflator_inflation_weo"].where(
        panel["gdp_deflator_inflation_weo"].notna(),
        panel["gdp_deflator_inflation_wb"],
    )
    panel["ppp_gdp"] = panel["ppp_gdp"].where(panel["ppp_gdp"].notna(), panel["ppp_gdp_wb"] / 1_000_000_000.0)
    panel["broad_money_gdp"] = panel["broad_money_gdp_wb"]
    panel["private_savings_gdp"] = panel["gross_savings_gdp_wb"]
    panel["gdp_nominal"] = panel["gdp_nominal"] * 1_000_000_000.0
    panel["gdp_real"] = panel["gdp_real"] * 1_000_000_000.0
    panel["ppp_gdp"] = panel["ppp_gdp"] * 1_000_000_000.0

    panel["inflation_target"] = panel["country"].map(INFLATION_TARGETS).fillna(DEFAULT_INFLATION_TARGET)
    panel["reserve_currency_flag"] = panel["country"].map(lambda name: PANEL_COUNTRIES[name]["reserve_currency_flag"]).astype(int)
    panel["threshold_debt_gdp"] = panel["country"].map(
        lambda name: DEFAULT_THRESHOLD_ADVANCED if PANEL_COUNTRIES[name]["advanced"] else DEFAULT_THRESHOLD_EMERGING
    )

    panel["debt_nominal"] = (panel["debt_gdp"] / 100.0) * panel["gdp_nominal"]
    panel["tax_revenue_nominal"] = (panel["tax_revenue_gdp"] / 100.0) * panel["gdp_nominal"]
    panel["interest_paid_nominal"] = (panel["interest_paid_gdp"] / 100.0) * panel["gdp_nominal"]
    panel["m2_nominal"] = np.nan
    panel["money_source"] = None
    direct_bcra_money = panel["m2_nominal_bcra"].notna()
    panel.loc[direct_bcra_money, "m2_nominal"] = panel.loc[direct_bcra_money, "m2_nominal_bcra"]
    panel.loc[direct_bcra_money, "money_source"] = "bcra_m2"
    direct_bdl_money = panel["m2_nominal"].isna() & panel["m2_nominal_bdl"].notna()
    panel.loc[direct_bdl_money, "m2_nominal"] = panel.loc[direct_bdl_money, "m2_nominal_bdl"]
    panel.loc[direct_bdl_money, "money_source"] = "bdl_money_supply_m2"
    direct_oecd_money = panel["m2_nominal"].isna() & panel["m3_nominal_oecd"].notna()
    panel.loc[direct_oecd_money, "m2_nominal"] = panel.loc[direct_oecd_money, "m3_nominal_oecd"] * 1_000_000.0
    panel.loc[direct_oecd_money, "money_source"] = "oecd_monagg_m3"
    wb_money_nominal = panel["m2_nominal"].isna() & panel["broad_money_current_lcu_wb"].notna()
    panel.loc[wb_money_nominal, "m2_nominal"] = panel.loc[wb_money_nominal, "broad_money_current_lcu_wb"]
    panel.loc[wb_money_nominal, "money_source"] = "world_bank_broad_money_lcu"
    wb_money = panel["m2_nominal"].isna() & panel["broad_money_gdp"].notna() & panel["gdp_nominal"].notna()
    panel.loc[wb_money, "m2_nominal"] = (panel.loc[wb_money, "broad_money_gdp"] / 100.0) * panel.loc[wb_money, "gdp_nominal"]
    panel.loc[wb_money, "money_source"] = "world_bank_broad_money_ratio"

    panel["housing_credit_share"] = panel["household_debt_gdp"] / panel["private_debt_gdp"]
    panel["idec_source"] = np.where(panel["investment_total_nominal"].notna(), "oecd_table8_activity", "bis_household_share_proxy")

    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    panel["avg_debt_nominal"] = panel.groupby("country")["debt_nominal"].transform(lambda s: s.where(s.shift(1).isna(), (s + s.shift(1)) / 2.0))
    panel["effective_nominal_rate_raw"] = (panel["interest_paid_nominal"] / panel["avg_debt_nominal"]) * 100.0
    panel["effective_nominal_rate_raw"] = panel["effective_nominal_rate_raw"].replace([np.inf, -np.inf], np.nan)
    panel["inflation_forward_12m"] = panel.groupby("country")["gdp_deflator_inflation"].shift(-1)
    panel["inflation_forward_12m"] = panel["inflation_forward_12m"].fillna(panel["gdp_deflator_inflation"])
    panel = _apply_yield_fallbacks(panel)
    panel = _apply_shared_euro_area_anchors(panel)
    panel["real_market_rate"] = panel["policy_rate_nominal"] - panel["inflation_forward_12m"]
    panel["m2_to_gdp_ratio"] = panel["m2_nominal"] / panel["gdp_nominal"]

    favar = compute_favar_estimates(panel)
    panel = panel.merge(favar, on=["country", "year"], how="left")

    panel = _append_global_aggregate(panel)
    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    panel.to_csv(Path(RAW_DIR) / "panel.csv", index=False)
    quality = _build_source_quality(panel)

    build_quarterly_panel(
        bis_policy_monthly=bis_policy_monthly,
        bis_eer_monthly=bis_eer_monthly,
        hlw_quarterly=hlw_quarterly,
        lm_quarterly=lm_quarterly,
        fred_frames=fred_frames,
    )
    return panel, quality
