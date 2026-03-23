"""
Collect latest rows from metric_3d.csv files and aggregate.

This script searches recursively for files named `metric_3d.csv` under a base
directory (default: `output/`). For each found CSV, it extracts the last
non-empty data row and writes it into a single aggregated CSV, adding a column
that indicates which dataset/scenario the row belongs to by parsing the
directory path (e.g., `output/<dataset>/<timestamp>/metric_3d.csv` → dataset is
`<dataset>`).

Usage:
  zsh> python data_collector.py --base output --out metrics_3d_all.csv

Notes:
- The original columns are preserved; an extra column `dataset` is added at the
  beginning for clarity.
- If multiple `metric_3d.csv` files share different headers, the union of
  columns is used and missing values are filled blank.
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple


def find_metric_csvs(base_dir: str) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f == "metrics_3d.csv":
                matches.append(os.path.join(root, f))
    return sorted(matches)


def parse_dataset_from_path(csv_path: str, base_dir: str) -> str:
    # Expected layout: base_dir/<dataset>/<timestamp>/metric_3d.csv
    # Fallback: use the immediate parent directory name.
    rel = os.path.relpath(csv_path, start=base_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        return parts[0]
    parent = os.path.basename(os.path.dirname(csv_path))
    return parent or "unknown"


def read_last_row(csv_path: str) -> Tuple[List[str], Optional[Dict[str, str]]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        last: Optional[Dict[str, str]] = None
        for row in reader:
            # Skip entirely empty rows
            if not row:
                continue
            # Consider a row empty if all values are empty strings
            if all((v is None) or (str(v).strip() == "") for v in row.values()):
                continue
            last = row
        return headers, last


def aggregate_metrics(base_dir: str, out_csv: str) -> int:
    csv_paths = find_metric_csvs(base_dir)
    if not csv_paths:
        return 0

    # Collect all headers to build a superset across files
    header_union: List[str] = []
    rows: List[Dict[str, str]] = []

    for p in csv_paths:
        headers, last = read_last_row(p)
        if last is None:
            continue
        for h in headers:
            if h not in header_union:
                header_union.append(h)
        dataset = parse_dataset_from_path(p, base_dir)
        enriched = {"dataset": dataset}
        # Preserve original values
        for h in headers:
            enriched[h] = last.get(h, "")
        rows.append(enriched)

    if not rows:
        return 0

    # Final headers: dataset + union of all per-file headers (in discovery order)
    final_headers = ["dataset"] + header_union

    # Ensure output directory exists
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writeheader()
        for r in rows:
            # Fill missing keys with blank
            for h in final_headers:
                if h not in r:
                    r[h] = ""
            writer.writerow(r)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate last rows of metric_3d.csv")
    parser.add_argument(
        "--base",
        default="output",
        help="Base directory to search recursively for metric_3d.csv",
    )
    parser.add_argument(
        "--out",
        default="metrics_3d_all.csv",
        help="Path to write aggregated CSV",
    )
    args = parser.parse_args()

    count = aggregate_metrics(args.base, args.out)
    print(f"Aggregated {count} rows into {args.out}")


if __name__ == "__main__":
    main()

