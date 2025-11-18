from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from .config import AppConfig
from .data_loader import PanelDataset


@dataclass
class ModelEvaluation:
    panel: str
    accuracy: float
    confusion_matrix: List[List[int]]
    labels: List[str]


class PanelModel:
    """Encapsulates model, scaler, and encoder for a given panel."""

    def __init__(self, panel: str, dataset: PanelDataset, artifacts: Dict[str, Path]) -> None:
        self.panel = panel
        self.dataset = dataset
        self.model = joblib.load(artifacts["model"])
        self.scaler = joblib.load(artifacts["scaler"])
        self.encoder = joblib.load(artifacts["encoder"])
        self.metadata: Dict[str, Any] | None = None
        if "metadata" in artifacts:
            self.metadata = joblib.load(artifacts["metadata"])
        if "feature_columns" in artifacts:
            self.feature_columns = joblib.load(artifacts["feature_columns"])
        elif self.metadata and "features" in self.metadata:
            self.feature_columns = list(self.metadata["features"])
        elif panel in AppConfig.MODEL_FEATURE_COLUMNS:
            self.feature_columns = AppConfig.MODEL_FEATURE_COLUMNS[panel]
        else:
            self.feature_columns = AppConfig.FEATURE_COLUMNS
        self._evaluation: ModelEvaluation | None = None

    def _prepare_features_for_prediction(self, payload: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare features for real-time prediction by aligning payload with the
        expected feature order.
        """
        feature_row = {feat: payload.get(feat, 0.0) for feat in self.feature_columns}

        return pd.DataFrame([feature_row], columns=self.feature_columns)

    def _prepare_features(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.scaler.transform(features)

    def _ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            for col in missing:
                df[col] = 0.0
        return df

    def evaluate(self) -> ModelEvaluation:
        if self._evaluation is not None:
            return self._evaluation

        df = self.dataset.filtered_df()
        df_with_features = self._ensure_feature_columns(df.copy())

        # Select features in the order expected by model
        X = df_with_features[self.feature_columns]
        y = df[AppConfig.LABEL_COLUMN].values
        y_encoded = self.encoder.transform(y)

        X_scaled = self._prepare_features(X)
        y_pred = self.model.predict(X_scaled)
        accuracy = float(accuracy_score(y_encoded, y_pred))

        cm = confusion_matrix(
            y_encoded,
            y_pred,
            labels=list(range(len(self.encoder.classes_))),
        ).tolist()

        self._evaluation = ModelEvaluation(
            panel=self.panel,
            accuracy=accuracy,
            confusion_matrix=cm,
            labels=list(self.encoder.classes_),
        )
        return self._evaluation

    def predict_label(self, payload: Dict[str, float]) -> str:
        # Prepare features with temporal features (using defaults for real-time prediction)
        ordered_values = self._prepare_features_for_prediction(payload)
        scaled = self._prepare_features(ordered_values)
        y_pred = self.model.predict(scaled)
        label = self.encoder.inverse_transform(y_pred)[0]
        return AppConfig.DISPLAY_LABELS.get(label, label)


class ModelRegistry:
    """Registry for all available panel models."""

    def __init__(self, datasets: Dict[str, PanelDataset], artifact_paths: Dict[str, Dict[str, Path]]) -> None:
        self._models: Dict[str, PanelModel] = {
            panel: PanelModel(panel, datasets[panel], paths)
            for panel, paths in artifact_paths.items()
        }

    def get_model(self, panel: str) -> PanelModel:
        try:
            return self._models[panel]
        except KeyError as exc:
            raise ValueError(f"Tidak ada model untuk panel '{panel}'") from exc

    def evaluations(self) -> List[ModelEvaluation]:
        return [model.evaluate() for model in self._models.values()]

    def best_model(self) -> PanelModel:
        return max(self._models.values(), key=lambda model: model.evaluate().accuracy)

    def list_panels(self) -> List[str]:
        return list(self._models.keys())


__all__ = ["ModelRegistry", "PanelModel", "ModelEvaluation"]


