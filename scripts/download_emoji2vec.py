"""Download pretrained emoji2vec.bin (300-dim) and save/upload it.

Usage:
    python scripts/download_emoji2vec.py
"""
import os
import sys
import urllib.request
from pathlib import Path

EMOJI2VEC_URL = "https://github.com/uclmr/emoji2vec/raw/master/pre-trained/emoji2vec.bin"
LOCAL_PATH = Path("data/emoji2vec.bin")
VOLUME_NAME = "vigoemotions-data"


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom headers to avoid basic bot blocking
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Saved locally to {dest} ({len(data):,} bytes)")


def upload_to_modal(local_path: Path, volume_name: str) -> None:
    try:
        import modal
    except ImportError:
        print("Modal library not installed. Skipping volume upload. File remains stored locally.")
        return

    print(f"Uploading {local_path} to Modal Volume '{volume_name}'...")
    try:
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)
        with vol.batch_upload() as batch:
            batch.put_file(local_path.as_posix(), "/emoji2vec.bin")
        print("Successfully uploaded to Modal volume.")
    except Exception as e:
        print(f"Warning: Failed to upload to Modal volume: {e}")


def main() -> int:
    try:
        download_file(EMOJI2VEC_URL, LOCAL_PATH)
        upload_to_modal(LOCAL_PATH, VOLUME_NAME)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
