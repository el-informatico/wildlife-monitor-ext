#!/usr/bin/env python
"""Export the ETL manifest (parquet) to JSON for the G1 batch driver.

The driver (scripts/g1_batch_load.py) runs on infra/.venv, which has boto3
but no pyarrow; the ETL manifest is parquet. This exporter is the only
piece that needs pyarrow, so it runs on any interpreter that has it (the
pipeline venv) and writes a plain-JSON extract the driver consumes:

    <pipeline-python> scripts/export_manifest_json.py \
        --parquet ../processed/manifest.parquet \
        --out build/media-manifest.json [--dataset african_wildlife]

The parquet file is opened read-only; nothing under processed/ is touched.
Output rows keep only the fields the driver needs (see FIELDS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = (
    "image_id",
    "source_dataset",
    "rel_path",
    "species",
    "location_id",
    "camera_ts",
    "is_night",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path,
                        help="ETL manifest parquet (read-only)")
    parser.add_argument("--out", required=True, type=Path,
                        help="destination JSON (list of row objects)")
    parser.add_argument("--dataset", default=None,
                        help="optional source_dataset filter")
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("error: pyarrow is required — run this on the pipeline venv",
              file=sys.stderr)
        return 2

    if not args.parquet.exists():
        print(f"error: {args.parquet} not found", file=sys.stderr)
        return 2

    table = pq.read_table(args.parquet)
    rows = table.select(list(FIELDS)).to_pylist()
    if args.dataset:
        rows = [r for r in rows if r["source_dataset"] == args.dataset]
    rows.sort(key=lambda r: (r["source_dataset"], r["location_id"], r["camera_ts"], r["image_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
