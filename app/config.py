"""
Streamlit app configuration.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Configuration for the Streamlit frontend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    data_dir: Path = Path("data")
    db_path: Path = Path("db/anomalies.db")

    # App theme
    app_title: str = "⚡ Smart Energy Anomaly Detector"
    app_icon: str = "⚡"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"


app_settings = AppSettings()
