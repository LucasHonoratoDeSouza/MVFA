from __future__ import annotations

import argparse
import json

from config import PANEL_FILE, QUALITY_FILE, QUARTERLY_FILE, RAW_DIR, SOURCE_URLS
from data_sources import build_real_api_panel


def write_source_manifest() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "panel_output": str(PANEL_FILE),
        "quarterly_output": str(QUARTERLY_FILE),
        "quality_output": str(QUALITY_FILE),
        "sources": SOURCE_URLS,
        "notes": [
            "panel.csv e construido inteiramente a partir de APIs e downloads oficiais.",
            "quarterly_panel.csv agrega as series de maior frequencia disponiveis.",
            "data_quality.csv resume a cobertura efetiva por pais.",
        ],
    }
    (RAW_DIR / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa dados oficiais via API e constroi o painel anual do MVFA.")
    parser.add_argument("--refresh", action="store_true", help="Rebaixa todas as fontes oficiais, ignorando cache local.")
    args = parser.parse_args()

    write_source_manifest()
    panel, quality = build_real_api_panel(refresh=args.refresh)
    print(f"Painel anual salvo em {PANEL_FILE} com {len(panel)} linhas.")
    print(f"Relatorio de qualidade salvo em {QUALITY_FILE} com {len(quality)} paises/agregados.")


if __name__ == "__main__":
    main()
