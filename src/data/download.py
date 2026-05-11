"""
download.py

Fetches the UCI Individual Household Electric Power Consumption dataset and
extracts it to data/raw/.

Idempotency: if the extracted text file already exists the function returns
immediately without re-downloading, so it is safe to run multiple times.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "household_power_consumption.zip"
EXPECTED_FILE = RAW_DIR / "household_power_consumption.txt"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

EXPECTED_HEADER = (
    "Date;Time;Global_active_power;Global_reactive_power;"
    "Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3"
)


def _verify_file(path: Path) -> None:
    """Raise if the extracted file's header does not match the known schema."""
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    if first_line != EXPECTED_HEADER:
        raise ValueError(
            f"Unexpected header in {path}.\n"
            f"Expected: {EXPECTED_HEADER}\n"
            f"Got:      {first_line}"
        )


def download_dataset(url: str = DATA_URL) -> None:
    """
    Download and extract the UCI power consumption zip to data/raw/.

    The zip contains one semicolon-delimited text file (~130 MB uncompressed).
    After extraction the zip is deleted to save disk space.  Header validation
    guards against silent upstream file changes.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXPECTED_FILE.exists():
        print(f"[skip] Dataset already present at:\n       {EXPECTED_FILE}")
        _verify_file(EXPECTED_FILE)
        return

    print(f"Downloading from {url} …")
    response = requests.get(url, stream=True, timeout=180)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with ZIP_PATH.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=65_536):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  ({downloaded // 1_048_576} MB)", end="", flush=True)
    print()

    print(f"Extracting to {RAW_DIR} …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    ZIP_PATH.unlink()

    if not EXPECTED_FILE.exists():
        raise FileNotFoundError(
            f"Extraction completed but {EXPECTED_FILE.name} was not found. "
            "The zip's internal structure may have changed upstream."
        )

    _verify_file(EXPECTED_FILE)
    size_mb = EXPECTED_FILE.stat().st_size / 1_048_576
    print(f"Done — {EXPECTED_FILE.name} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    download_dataset()
