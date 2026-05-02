"""One-shot helper: upload the local ViGoEmotions CSVs to a Modal Volume.

Run from the project root:

    python scripts/upload_data.py

This pushes `data/vigoemotions/{train,val,test}.csv` (and the README) into the
Modal Volume named `vigoemotions-data`. Inside the training container the files
are visible at `/data/train.csv`, `/data/val.csv`, `/data/test.csv`.

You only need to run this once; rerun it if the local CSVs change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

DEFAULT_VOLUME_NAME = "vigoemotions-data"
FILES_TO_UPLOAD = ["train.csv", "val.csv", "test.csv", "README.md"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/vigoemotions",
        help="Local directory containing train.csv / val.csv / test.csv.",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default=DEFAULT_VOLUME_NAME,
        help="Modal Volume name (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the volume.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist.", file=sys.stderr)
        return 1

    missing = [name for name in FILES_TO_UPLOAD[:3] if not (data_dir / name).exists()]
    if missing:
        print(f"ERROR: Missing required files in {data_dir}: {missing}", file=sys.stderr)
        return 1

    print(f"Uploading from {data_dir} -> Modal Volume '{args.volume_name}'")
    vol = modal.Volume.from_name(args.volume_name, create_if_missing=True)

    with vol.batch_upload(force=args.force) as batch:
        for name in FILES_TO_UPLOAD:
            local = data_dir / name
            if not local.exists():
                print(f"  skip (not found): {name}")
                continue
            remote = f"/{name}"
            print(f"  + {local}  ->  {remote}  ({local.stat().st_size:,} bytes)")
            batch.put_file(local.as_posix(), remote)

    print("Done. Verifying volume contents...")
    for entry in vol.listdir("/"):
        size = getattr(entry, "size", "?")
        print(f"  /{entry.path}  ({size} bytes)")

    print(
        "\nNext step:\n"
        "  modal run modal_app.py::train "
        "--config-path configs/visobert_baseline.yaml --run-name visobert-baseline-v1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
