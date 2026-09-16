"""Building classes backed by prepared annual load and PV profiles.

Run ``BG_data_prepare.ipynb`` first to create the four hourly CSV files.
Each concrete building class accepts a date and exposes that day's 24-hour
load and photovoltaic energy curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
PROFILE_COLUMNS = ["load_kwh", "pv_kwh", "outdoor_temperature_c"]
ENERGY_COLUMNS = ["load_kwh", "pv_kwh"]


class Building:
    """Base class for a building with a prepared annual hourly profile."""

    name: ClassVar[str]
    building_type: ClassVar[str]
    profile_filename: ClassVar[str]

    def __init__(
        self,
        date: str | pd.Timestamp,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> None:
        self.date = self._normalize_date(date)
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / self.profile_filename
        self.profile = self._read_daily_profile()
        self.load_curve = self.profile["load_kwh"].to_numpy(dtype=float, copy=True)
        self.pv_curve = self.profile["pv_kwh"].to_numpy(dtype=float, copy=True)
        self.temperature_curve = self.profile["outdoor_temperature_c"].to_numpy(
            dtype=float, copy=True
        )

    @staticmethod
    def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
        try:
            selected = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid building profile date: {value!r}") from exc
        if pd.isna(selected):
            raise ValueError(f"Invalid building profile date: {value!r}")
        if selected.tz is not None:
            selected = selected.tz_localize(None)
        return selected.normalize()

    def _read_annual_profile(self) -> pd.DataFrame:
        if not self.data_file.is_file():
            raise FileNotFoundError(
                f"Prepared profile not found: {self.data_file}. "
                "Run BG_data_prepare.ipynb first."
            )

        annual = pd.read_csv(self.data_file)
        expected_csv_columns = ["timestamp", *PROFILE_COLUMNS]
        if annual.columns.tolist() != expected_csv_columns:
            raise ValueError(
                f"{self.data_file.name} must have columns "
                f"{expected_csv_columns}; got {annual.columns.tolist()}."
            )

        annual["timestamp"] = pd.to_datetime(annual["timestamp"], errors="coerce")
        if annual["timestamp"].isna().any():
            raise ValueError(f"{self.data_file.name} contains invalid timestamps.")
        if annual["timestamp"].duplicated().any():
            raise ValueError(f"{self.data_file.name} contains duplicate timestamps.")

        annual = annual.set_index("timestamp")
        annual.index.name = "timestamp"
        if not annual.index.is_monotonic_increasing:
            raise ValueError(
                f"{self.data_file.name} timestamps must be in ascending order."
            )

        annual[PROFILE_COLUMNS] = annual[PROFILE_COLUMNS].apply(
            pd.to_numeric, errors="raise"
        )
        if annual[PROFILE_COLUMNS].isna().any().any():
            raise ValueError(f"{self.data_file.name} contains missing profile values.")
        if (annual[ENERGY_COLUMNS] < 0).any().any():
            raise ValueError(f"{self.data_file.name} contains negative energy values.")

        years = annual.index.year.unique()
        if len(years) != 1:
            raise ValueError(
                f"{self.data_file.name} must contain exactly one calendar year."
            )
        year = int(years[0])
        expected_index = pd.date_range(
            f"{year}-01-01 00:00:00",
            f"{year}-12-31 23:00:00",
            freq="h",
            name="timestamp",
        )
        if not annual.index.equals(expected_index):
            raise ValueError(
                f"{self.data_file.name} must contain every hour of {year} "
                f"exactly once ({len(expected_index)} rows)."
            )
        return annual

    def _read_daily_profile(self) -> pd.DataFrame:
        annual = self._read_annual_profile()
        expected_day = pd.date_range(
            self.date,
            periods=24,
            freq="h",
            name="timestamp",
        )
        daily = annual.reindex(expected_day)
        if daily.isna().any().any():
            available_year = int(annual.index[0].year)
            raise ValueError(
                f"{self.date.date()} is not a complete day in "
                f"{self.data_file.name}; available year is {available_year}."
            )
        return daily

    def get_load_curve(self) -> np.ndarray:
        """Return a copy of the selected day's 24-hour load-energy curve."""
        return self.load_curve.copy()

    def get_pv_curve(self) -> np.ndarray:
        """Return a copy of the selected day's 24-hour PV-energy curve."""
        return self.pv_curve.copy()

    def get_temperature_curve(self) -> np.ndarray:
        """Return a copy of the selected day's 24-hour outdoor-temperature curve."""
        return self.temperature_curve.copy()

    def parameters(self) -> dict[str, Any]:
        """Return the identifying parameters for this building instance."""
        return {
            "name": self.name,
            "building_type": self.building_type,
            "date": self.date.strftime("%Y-%m-%d"),
            "data_file": str(self.data_file),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(date={self.date.strftime('%Y-%m-%d')!r}, "
            f"data_file={str(self.data_file)!r})"
        )


class ResidentialBuilding1(Building):
    """Residence using Residential House load and SQ1 PV data."""

    name = "Residential 1"
    building_type = "Residential"
    profile_filename = "residential_1_2023.csv"


class ResidentialBuilding2(Building):
    """Residence using Residential House C1 load and SQ Apartment PV data."""

    name = "Residential 2"
    building_type = "Residential"
    profile_filename = "residential_2_2023.csv"


class LargeCommercialBuilding1(Building):
    """Supermarket load paired with Indoor Sports Centre PV data."""

    name = "Large Commercial 1"
    building_type = "Large commercial"
    profile_filename = "large_commercial_1_2023.csv"


class LargeCommercialBuilding2(Building):
    """Retail Standalone load paired with Library PV data."""

    name = "Large Commercial 2"
    building_type = "Large commercial"
    profile_filename = "large_commercial_2_2023.csv"


def default_buildings(
    date: str | pd.Timestamp,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> list[Building]:
    """Create all four building profiles for the same date."""
    return [
        ResidentialBuilding1(date, data_dir),
        ResidentialBuilding2(date, data_dir),
        LargeCommercialBuilding1(date, data_dir),
        LargeCommercialBuilding2(date, data_dir),
    ]


__all__ = [
    "Building",
    "ResidentialBuilding1",
    "ResidentialBuilding2",
    "LargeCommercialBuilding1",
    "LargeCommercialBuilding2",
    "default_buildings",
]
