from __future__ import annotations

import unittest

from bibliography import build_metric_reference_index, parse_bibliography


class BibliographyTests(unittest.TestCase):
    def test_parse_bibliography_finds_entries(self) -> None:
        entries = parse_bibliography()
        self.assertGreater(len(entries), 20)
        self.assertEqual(entries[0].entry_id, "A1")

    def test_metric_index_contains_domar_references(self) -> None:
        entries = parse_bibliography()
        metric_index = build_metric_reference_index(entries)
        domar_ids = [item["entry_id"] for item in metric_index["domar_gap"]]
        self.assertIn("D1", domar_ids)


if __name__ == "__main__":
    unittest.main()
