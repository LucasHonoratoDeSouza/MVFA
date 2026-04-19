from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from config import BOOKS_FILE, TABLES_DIR


ENTRY_RE = re.compile(r"^\[(?P<entry_id>[A-Z]\d+)\]\s+(?P<citation>.+)$")
BLOCK_RE = re.compile(r"^BLOCO\s+\d+\s+—\s+(?P<block>.+)$")


@dataclass(slots=True)
class BibliographyEntry:
    entry_id: str
    block: str
    citation: str
    details: list[str]


METRIC_REFERENCE_MAP = {
    "domar_gap": ["A1", "A2", "D1", "D2", "D3", "D4", "D5"],
    "wicksell_gap": ["A3", "A5", "C1", "C2", "C3", "C4", "C5", "C7"],
    "ema": ["A1", "B2", "F1", "F2"],
    "isf": ["D2", "D3", "D5", "E1", "E2"],
    "idec": ["A3", "A4", "A6", "B1", "B2", "B4"],
    "sdsa": ["E1", "E2", "E3", "E4", "E5", "E6", "H5"],
}


def parse_bibliography(path: Path | None = None) -> list[BibliographyEntry]:
    source = path or BOOKS_FILE
    current_block = "Sem bloco"
    entries: list[BibliographyEntry] = []
    current_entry: BibliographyEntry | None = None

    with Path(source).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            stripped = line.strip()

            block_match = BLOCK_RE.match(stripped)
            if block_match:
                current_block = block_match.group("block").strip()
                continue

            entry_match = ENTRY_RE.match(stripped)
            if entry_match:
                if current_entry is not None:
                    entries.append(current_entry)
                current_entry = BibliographyEntry(
                    entry_id=entry_match.group("entry_id"),
                    block=current_block,
                    citation=entry_match.group("citation").strip(),
                    details=[],
                )
                continue

            if current_entry is not None and stripped:
                current_entry.details.append(stripped)

    if current_entry is not None:
        entries.append(current_entry)

    return entries


def build_metric_reference_index(entries: list[BibliographyEntry]) -> dict[str, list[dict[str, str]]]:
    lookup = {entry.entry_id: entry for entry in entries}
    index: dict[str, list[dict[str, str]]] = {}
    for metric, entry_ids in METRIC_REFERENCE_MAP.items():
        index[metric] = []
        for entry_id in entry_ids:
            entry = lookup.get(entry_id)
            if entry is None:
                continue
            index[metric].append(
                {
                    "entry_id": entry.entry_id,
                    "block": entry.block,
                    "citation": entry.citation,
                }
            )
    return index


def export_bibliography(
    path: Path | None = None,
    output_dir: Path | None = None,
) -> list[BibliographyEntry]:
    entries = parse_bibliography(path)
    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "bibliografia.json"
    csv_path = out_dir / "bibliografia.csv"
    index_path = out_dir / "bibliografia_metricas.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(entry) for entry in entries], handle, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_id", "block", "citation", "details"])
        for entry in entries:
            writer.writerow([entry.entry_id, entry.block, entry.citation, " | ".join(entry.details)])

    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(build_metric_reference_index(entries), handle, ensure_ascii=False, indent=2)

    return entries
