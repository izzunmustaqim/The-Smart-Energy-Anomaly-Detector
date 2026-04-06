"""
UCI Individual Household Electric Power Consumption adapter.

Downloads the dataset from the UCI repository (if not cached),
extracts it, and returns a clean DataFrame with proper dtypes.

Dataset: https://archive.ics.uci.edu/dataset/235
License: CC BY 4.0
"""

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from processor.config import settings
from processor.ingestion.base import DataSource

logger = logging.getLogger(__name__)

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_FILENAME = "household_power_consumption.txt"


class UCIAdapter(DataSource):
    """
    Loads the UCI Individual Household Electric Power Consumption dataset.

    On first run, downloads the ~20 MB ZIP file and extracts the
    semicolon-delimited text file to `data/raw/`. Subsequent runs
    read from the cached file.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or settings.raw_data_dir

    # ── Public API ─────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        """Download (if needed) and parse the UCI power dataset."""
        txt_path = self._data_dir / UCI_FILENAME

        if not txt_path.exists():
            logger.info("Dataset not found locally — downloading from UCI …")
            self._download(txt_path)
        else:
            logger.info("Using cached dataset at %s", txt_path)

        return self._parse(txt_path)

    # ── Private helpers ────────────────────────────────────────────
    def _download(self, target: Path) -> None:
        """Download and extract the UCI ZIP archive."""
        target.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(UCI_DATASET_URL, timeout=120)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # The ZIP contains the .txt file (possibly nested)
            txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
            if not txt_members:
                raise FileNotFoundError(
                    "No .txt file found inside the UCI ZIP archive."
                )
            logger.info("Extracting %s → %s", txt_members[0], target)
            with zf.open(txt_members[0]) as src, open(target, "wb") as dst:
                dst.write(src.read())

        logger.info("Download complete — %s bytes", target.stat().st_size)

    @staticmethod
    def _parse(path: Path) -> pd.DataFrame:
        """
        Parse the semicolon-delimited text file into a DataFrame.

        The file uses ';' as separator and '?' for missing values.
        We combine the Date + Time columns into a single datetime index.
        """
        logger.info("Parsing %s …", path.name)

        df = pd.read_csv(
            path,
            sep=";",
            na_values=["?", ""],
            low_memory=False,
            dtype={
                "Global_active_power": "float64",
                "Global_reactive_power": "float64",
                "Voltage": "float64",
                "Global_intensity": "float64",
                "Sub_metering_1": "float64",
                "Sub_metering_2": "float64",
                "Sub_metering_3": "float64",
            },
        )

        # Combine Date + Time into a single datetime column
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S",
        )
        df.drop(columns=["Date", "Time"], inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        logger.info(
            "Loaded %s rows × %s columns  |  range: %s → %s",
            len(df),
            len(df.columns),
            df.index.min(),
            df.index.max(),
        )
        return df
