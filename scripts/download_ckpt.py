"""Download a run's artifacts (TensorBoard logs, best.pt, metrics.json) from
the `vigoemotions-runs` Modal Volume to a local `artifacts/<run_name>/` folder.

Usage:
    python scripts/download_ckpt.py --run-name visobert-baseline-v1
    python scripts/download_ckpt.py --run-name visobert-baseline-v1 --tb-only
    python scripts/download_ckpt.py --list      # just list runs in the volume

After download, view TensorBoard locally:
    tensorboard --logdir artifacts/visobert-baseline-v1/tb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

DEFAULT_VOLUME_NAME = "vigoemotions-runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=None,
                        help="The run directory under /runs/ to download.")
    parser.add_argument("--volume-name", type=str, default=DEFAULT_VOLUME_NAME)
    parser.add_argument("--out-dir", type=str, default="artifacts",
                        help="Local output root (default: ./artifacts).")
    parser.add_argument("--tb-only", action="store_true",
                        help="Only download the tb/ subfolder + metrics.json (skip best.pt).")
    parser.add_argument("--list", action="store_true",
                        help="List available runs and exit.")
    return parser.parse_args()


def _iter_volume(vol: modal.Volume, root: str = "/"):
    """Yield (path, is_dir, size) for every entry under `root` recursively."""
    stack = [root]
    while stack:
        current = stack.pop()
        for entry in vol.listdir(current):
            path = entry.path if entry.path.startswith("/") else f"/{entry.path}"
            is_dir = getattr(entry, "type", None) == 2 or getattr(entry, "is_dir", False)
            if is_dir:
                stack.append(path)
                yield path, True, 0
            else:
                yield path, False, getattr(entry, "size", 0)


def main() -> int:
    args = parse_args()
    vol = modal.Volume.from_name(args.volume_name, create_if_missing=False)

    if args.list:
        print(f"Runs in volume '{args.volume_name}':")
        try:
            for entry in vol.listdir("/"):
                print(f"  /{entry.path}")
        except Exception as e:
            print(f"  (failed to list: {e})", file=sys.stderr)
            return 1
        return 0

    if not args.run_name:
        print("ERROR: --run-name is required (or use --list).", file=sys.stderr)
        return 1

    out_root = Path(args.out_dir) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    remote_root = f"/{args.run_name}"

    print(f"Downloading run '{args.run_name}' from volume '{args.volume_name}' -> {out_root}")

    files: list[tuple[str, int]] = []
    for path, is_dir, size in _iter_volume(vol, remote_root):
        if is_dir:
            continue
        if args.tb_only and not (path.startswith(f"{remote_root}/tb/") or path.endswith("metrics.json") or path.endswith("config.json")):
            continue
        files.append((path, size))

    if not files:
        print("  (nothing to download)")
        return 0

    for remote_path, size in files:
        rel = remote_path[len(remote_root) + 1:]
        local_path = out_root / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  + {remote_path}  ->  {local_path}  ({size} bytes)")
        with local_path.open("wb") as f:
            for chunk in vol.read_file(remote_path):
                f.write(chunk)

    print("\nDone.")
    print(f"  TensorBoard:  tensorboard --logdir {out_root.as_posix()}/tb")
    metrics_path = out_root / "metrics.json"
    if metrics_path.exists():
        print(f"  Metrics:      {metrics_path}")
    ckpt_path = out_root / "best.pt"
    if ckpt_path.exists():
        print(f"  Checkpoint:   {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
