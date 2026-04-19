from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_SOURCES_DIR = RAW_DIR / "sources"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"
BOOKS_FILE = ROOT_DIR / "livros.txt"
PROMPT_FILE = ROOT_DIR / "prompt.txt"
PANEL_FILE = RAW_DIR / "panel.csv"
QUARTERLY_FILE = RAW_DIR / "quarterly_panel.csv"
QUALITY_FILE = RAW_DIR / "data_quality.csv"
PANEL_TEMPLATE_FILE = RAW_DIR / "panel_template.csv"
PROCESSED_PANEL_FILE = PROCESSED_DIR / "panel_clean.csv"

START_YEAR = 1994
OBSERVED_END_YEAR = 2024
FORWARD_END_YEAR = 2025
SDSA_ESTIMATION_END_YEAR = 2019
PANEL_BASE_YEAR = 2024

PANEL_COUNTRIES = {
    "United States": {
        "imf": "USA",
        "wb": "USA",
        "bis": "US",
        "oecd": "USA",
        "hlw": "US",
        "lm": "US",
        "reserve_currency_flag": 1,
        "advanced": 1,
    },
    "Euro Area": {
        "imf": "EURO",
        "wb": "EMU",
        "bis": "XM",
        "oecd": None,
        "hlw": "Euro Area",
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Japan": {
        "imf": "JPN",
        "wb": "JPN",
        "bis": "JP",
        "oecd": "JPN",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "United Kingdom": {
        "imf": "GBR",
        "wb": "GBR",
        "bis": "GB",
        "oecd": "GBR",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Canada": {
        "imf": "CAN",
        "wb": "CAN",
        "bis": "CA",
        "oecd": "CAN",
        "hlw": "Canada",
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Australia": {
        "imf": "AUS",
        "wb": "AUS",
        "bis": "AU",
        "oecd": "AUS",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "South Korea": {
        "imf": "KOR",
        "wb": "KOR",
        "bis": "KR",
        "oecd": "KOR",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "China": {
        "imf": "CHN",
        "wb": "CHN",
        "bis": "CN",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Brazil": {
        "imf": "BRA",
        "wb": "BRA",
        "bis": "BR",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "India": {
        "imf": "IND",
        "wb": "IND",
        "bis": "IN",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Mexico": {
        "imf": "MEX",
        "wb": "MEX",
        "bis": "MX",
        "oecd": "MEX",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Turkey": {
        "imf": "TUR",
        "wb": "TUR",
        "bis": "TR",
        "oecd": "TUR",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Indonesia": {
        "imf": "IDN",
        "wb": "IDN",
        "bis": "ID",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Saudi Arabia": {
        "imf": "SAU",
        "wb": "SAU",
        "bis": "SA",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Argentina": {
        "imf": "ARG",
        "wb": "ARG",
        "bis": "AR",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Greece": {
        "imf": "GRC",
        "wb": "GRC",
        "bis": "GR",
        "oecd": "GRC",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
    "Portugal": {
        "imf": "PRT",
        "wb": "PRT",
        "bis": "PT",
        "oecd": "PRT",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Italy": {
        "imf": "ITA",
        "wb": "ITA",
        "bis": "IT",
        "oecd": "ITA",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Spain": {
        "imf": "ESP",
        "wb": "ESP",
        "bis": "ES",
        "oecd": "ESP",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Ireland": {
        "imf": "IRL",
        "wb": "IRL",
        "bis": "IE",
        "oecd": "IRL",
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 1,
    },
    "Lebanon": {
        "imf": "LBN",
        "wb": "LBN",
        "bis": "LB",
        "oecd": None,
        "hlw": None,
        "lm": None,
        "reserve_currency_flag": 0,
        "advanced": 0,
    },
}

PANEL_COUNTRY_NAMES = list(PANEL_COUNTRIES)
ADVANCED_ECONOMIES = {name for name, meta in PANEL_COUNTRIES.items() if meta["advanced"]}
EMERGING_ECONOMIES = set(PANEL_COUNTRY_NAMES) - ADVANCED_ECONOMIES

INFLATION_TARGETS = {
    "United States": 2.0,
    "Euro Area": 2.0,
    "Japan": 2.0,
    "United Kingdom": 2.0,
    "Canada": 2.0,
    "Australia": 2.5,
    "South Korea": 2.0,
    "China": 3.0,
    "Brazil": 3.0,
    "India": 4.0,
    "Mexico": 3.0,
    "Turkey": 5.0,
    "Indonesia": 2.5,
    "Saudi Arabia": 2.0,
    "Argentina": 2.0,
    "Greece": 2.0,
    "Portugal": 2.0,
    "Italy": 2.0,
    "Spain": 2.0,
    "Ireland": 2.0,
    "Lebanon": 2.0,
}

COUNTRY_NORMALIZATION = {
    "eua": "United States",
    "estados unidos": "United States",
    "usa": "United States",
    "uk": "United Kingdom",
    "reino unido": "United Kingdom",
    "japao": "Japan",
    "japão": "Japan",
    "zona do euro": "Euro Area",
    "euro area": "Euro Area",
    "brasil": "Brazil",
    "grecia": "Greece",
    "grécia": "Greece",
    "coreia do sul": "South Korea",
    "corea do sul": "South Korea",
    "turquia": "Turkey",
    "turkiye": "Turkey",
    "méxico": "Mexico",
    "arabia saudita": "Saudi Arabia",
    "arábia saudita": "Saudi Arabia",
}

RAW_COLUMN_ALIASES = {
    "pais": "country",
    "país": "country",
    "ano": "year",
    "divida_pib": "debt_gdp",
    "dívida_pib": "debt_gdp",
    "divida_nominal": "debt_nominal",
    "dívida_nominal": "debt_nominal",
    "divida_media_nominal": "avg_debt_nominal",
    "juros_pagos_nominal": "interest_paid_nominal",
    "saldo_primario_pib": "primary_balance_gdp",
    "saldo_primário_pib": "primary_balance_gdp",
    "crescimento_real_pib": "gdp_real_growth",
    "inflacao_deflator": "gdp_deflator_inflation",
    "inflação_deflator": "gdp_deflator_inflation",
    "taxa_politica_nominal": "policy_rate_nominal",
    "taxa_10y_nominal": "yield_10y_nominal",
    "inflacao_futura_12m": "inflation_forward_12m",
    "inflação_futura_12m": "inflation_forward_12m",
    "receita_tributaria_nominal": "tax_revenue_nominal",
    "m2": "m2_nominal",
    "pib_nominal": "gdp_nominal",
    "pib_real": "gdp_real",
    "meta_inflacao": "inflation_target",
    "meta_inflação": "inflation_target",
    "investimento_sensivel_nominal": "investment_sensitive_nominal",
    "investimento_sensível_nominal": "investment_sensitive_nominal",
    "investimento_total_nominal": "investment_total_nominal",
    "poupanca_privada_pib": "private_savings_gdp",
    "poupança_privada_pib": "private_savings_gdp",
    "credito_imobiliario_share": "housing_credit_share",
    "crédito_imobiliário_share": "housing_credit_share",
    "variacao_cambio_real": "fx_real_change",
    "variação_câmbio_real": "fx_real_change",
    "taxa_natural_hlw": "natural_rate_hlw",
    "taxa_natural_lm": "natural_rate_lm",
    "taxa_natural_favar": "natural_rate_favar",
}

TEMPLATE_COLUMNS = [
    "country",
    "year",
    "gdp_nominal",
    "gdp_real",
    "gdp_real_growth",
    "gdp_deflator_inflation",
    "ppp_gdp",
    "debt_gdp",
    "debt_nominal",
    "avg_debt_nominal",
    "primary_balance_gdp",
    "interest_paid_nominal",
    "tax_revenue_nominal",
    "policy_rate_nominal",
    "yield_10y_nominal",
    "inflation_forward_12m",
    "m2_nominal",
    "private_savings_gdp",
    "investment_sensitive_nominal",
    "investment_total_nominal",
    "housing_credit_share",
    "fx_real_change",
    "natural_rate_hlw",
    "natural_rate_lm",
    "natural_rate_favar",
    "inflation_target",
    "reserve_currency_flag",
    "domar_beta",
    "threshold_debt_gdp",
    "yield_source",
    "idec_source",
]

CORE_REQUIRED_COLUMNS = [
    "country",
    "year",
    "debt_gdp",
    "primary_balance_gdp",
    "gdp_real_growth",
    "gdp_deflator_inflation",
    "policy_rate_nominal",
    "m2_nominal",
    "gdp_nominal",
    "gdp_real",
    "inflation_target",
]

RISK_WEIGHTS = {
    "domar_gap": 0.30,
    "wicksell_integral_mid": 0.25,
    "ema_pct": 0.20,
    "isf_pct": 0.15,
    "idec": 0.10,
}

IMF_INDICATORS = {
    "gdp_real_growth": "NGDP_RPCH",
    "gdp_nominal_imf": "NGDPD",
    "ppp_gdp": "PPPGDP",
    "debt_gdp": "G_XWDG_G01_GDP_PT",
    "primary_balance_gdp": "GGXONLB_G01_GDP_PT",
    "revenue_gdp": "GGR_G01_GDP_PT",
    "interest_paid_gdp": "ie",
    "overall_balance_gdp": "GGXCNL_G01_GDP_PT",
    "real_long_term_yield": "rltir",
    "household_debt_gdp": "HH_ALL",
    "private_debt_gdp": "Privatedebt_all",
}

WORLD_BANK_INDICATORS = {
    "gdp_nominal": "NY.GDP.MKTP.CD",
    "gdp_real": "NY.GDP.MKTP.KD",
    "ppp_gdp_wb": "NY.GDP.MKTP.PP.CD",
    "gdp_deflator_inflation": "NY.GDP.DEFL.KD.ZG",
    "broad_money_current_lcu": "FM.LBL.BMNY.CN",
    "broad_money_gdp": "FM.LBL.BMNY.GD.ZS",
    "gross_savings_gdp": "NY.GNS.ICTR.ZS",
    "lending_rate": "FR.INR.LEND",
}

FRED_SERIES = {
    "FEDFUNDS": "fedfunds_rate.csv",
    "M2SL": "m2_usa.csv",
    "M2V": "m2_velocity_usa.csv",
    "GDPC1": "real_gdp_usa.csv",
    "PCEPI": "pce_deflator_usa.csv",
}

OECD_DIRECT_IDEC_CODES = sorted(
    {
        meta["oecd"]
        for meta in PANEL_COUNTRIES.values()
        if meta.get("oecd") is not None
    }
)

OECD_SENSITIVE_ACTIVITY_PREFIXES = (
    "F",
    "L68",
    "D35",
    "E36",
    "E37",
    "E38",
    "E39",
    "H49",
    "H50",
    "H51",
    "H52",
    "H53",
)

DEFAULT_INFLATION_TARGET = 2.0
DEFAULT_THRESHOLD_ADVANCED = 120.0
DEFAULT_THRESHOLD_EMERGING = 80.0
DEFAULT_JAPAN_ISF_THRESHOLD = 25.0

SOURCE_URLS = {
    "imf_datamapper_api": "https://www.imf.org/external/datamapper/api/v1/",
    "imf_weo_country_tsv": "https://www.imf.org/-/media/files/publications/weo/weo-database/2025/april/weoapr2025all.xls",
    "world_bank_api": "https://api.worldbank.org/v2/",
    "eurostat_api": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/",
    "bis_api_doc": "https://stats.bis.org/api-doc/v2/",
    "bis_data_api": "https://stats.bis.org/api/v2/",
    "oecd_api": "https://sdmx.oecd.org/public/rest/",
    "nyfed_hlw_xlsx": "https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/Holston_Laubach_Williams_current_estimates.xlsx",
    "richmond_lm_page": "https://www.richmondfed.org/research/national_economy/natural_rate_interest",
    "bdl_money_supply_history": "https://bdl.gov.lb/moneysupplyhistory.php",
    "bdl_base_url": "https://www.banqueduliban.gov.lb/",
    "bcra_api": "https://api.bcra.gob.ar/estadisticas/v4.0/",
}
