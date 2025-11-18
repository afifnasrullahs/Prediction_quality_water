from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import AppConfig


@dataclass
class DatasetSummary:
    panel: str
    feature_summary: pd.DataFrame
    sample_rows: pd.DataFrame
    label_distribution_before: Dict[str, int]
    label_distribution_after: Dict[str, int]


class PanelDataset:
    """Lazy-loading wrapper around a panel-specific dataset."""

    def __init__(self, panel: str, csv_path: Path) -> None:
        self.panel = panel
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def filtered_df(self) -> pd.DataFrame:
        allowed = AppConfig.ALLOWED_LABELS
        return self.df[self.df[AppConfig.LABEL_COLUMN].isin(allowed)].copy()

    def get_summary(self, sample_size: int = 5) -> DatasetSummary:
        filtered = self.filtered_df().copy()
        sample_rows = filtered.head(sample_size)

        # Pastikan dataframe untuk ringkasan fitur memiliki semua kolom yang diharapkan
        for col in AppConfig.FEATURE_COLUMNS:
            if col not in filtered.columns:
                filtered[col] = pd.NA

        feature_summary = (
            filtered[AppConfig.FEATURE_COLUMNS]
            .describe()
            .transpose()
            .round(3)
        )

        def _count(series: pd.Series) -> Dict[str, int]:
            counts = series.value_counts().reindex(AppConfig.ALLOWED_LABELS, fill_value=0)
            return counts.to_dict()

        label_distribution_before = _count(self.df[AppConfig.LABEL_COLUMN])
        label_distribution_after = _count(filtered[AppConfig.LABEL_COLUMN])

        return DatasetSummary(
            panel=self.panel,
            feature_summary=feature_summary,
            sample_rows=sample_rows,
            label_distribution_before=label_distribution_before,
            label_distribution_after=label_distribution_after,
        )


class DataRepository:
    """Provides access to multiple panel datasets."""

    def __init__(self, dataset_paths: Dict[str, Path]) -> None:
        self._datasets: Dict[str, PanelDataset] = {
            panel: PanelDataset(panel, path)
            for panel, path in dataset_paths.items()
        }

    def get_dataset(self, panel: str) -> PanelDataset:
        try:
            return self._datasets[panel]
        except KeyError as exc:
            raise ValueError(f"Tidak ada dataset untuk panel '{panel}'") from exc

    def list_panels(self) -> List[str]:
        return list(self._datasets.keys())


__all__ = ["DataRepository", "DatasetSummary", "PanelDataset"]


