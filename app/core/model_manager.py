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
        else:
            self.feature_columns = AppConfig.FEATURE_COLUMNS
        self.uses_temporal_features = "feature_columns" in artifacts
        self._evaluation: ModelEvaluation | None = None

    def _create_temporal_features(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Create temporal features from dataframe (for evaluation).
        Same logic as in notebook: lag, rolling stats, delta.
        """
        if features is None:
            features = AppConfig.FEATURE_COLUMNS
        
        df = df.copy()
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        for feature in features:
            # Lag features (previous values)
            df[f'{feature}_lag1'] = df[feature].shift(1)
            df[f'{feature}_lag2'] = df[feature].shift(2)
            df[f'{feature}_lag3'] = df[feature].shift(3)
            
            # Rolling statistics (window=5)
            df[f'{feature}_rolling_mean'] = df[feature].rolling(window=5, min_periods=1).mean()
            df[f'{feature}_rolling_std'] = df[feature].rolling(window=5, min_periods=1).std().fillna(0)
            
            # Rate of change (delta)
            df[f'{feature}_delta'] = df[feature].diff().fillna(0)
            
            # Rolling min/max
            df[f'{feature}_rolling_min'] = df[feature].rolling(window=5, min_periods=1).min()
            df[f'{feature}_rolling_max'] = df[feature].rolling(window=5, min_periods=1).max()
        
        # Fill NaN in lag features with forward fill, then backward fill
        lag_cols = [col for col in df.columns if 'lag' in col]
        if lag_cols:
            df[lag_cols] = df[lag_cols].ffill().bfill()
        
        # If still any NaN, fill with 0
        df = df.fillna(0)
        
        return df

    def _prepare_features_for_prediction(self, payload: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare features for real-time prediction.
        When temporal features are required, we synthesize stats; otherwise we
        just align payload values with expected feature ordering.
        """
        feature_row: Dict[str, float]
        if self.uses_temporal_features:
            base_features = AppConfig.FEATURE_COLUMNS
            feature_dict = {}

            for feature in base_features:
                value = payload.get(feature, 0.0)

                # Base feature
                feature_dict[feature] = value

                # Lag features (use current value as default since no history)
                feature_dict[f"{feature}_lag1"] = value
                feature_dict[f"{feature}_lag2"] = value
                feature_dict[f"{feature}_lag3"] = value

                # Rolling statistics (use current value)
                feature_dict[f"{feature}_rolling_mean"] = value
                feature_dict[f"{feature}_rolling_std"] = 0.0
                feature_dict[f"{feature}_rolling_min"] = value
                feature_dict[f"{feature}_rolling_max"] = value

                # Delta (no change)
                feature_dict[f"{feature}_delta"] = 0.0

            feature_row = {feat: feature_dict.get(feat, 0.0) for feat in self.feature_columns}
        else:
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

        if self.uses_temporal_features:
            df_with_features = self._create_temporal_features(df)
        else:
            df_with_features = df.copy()

        df_with_features = self._ensure_feature_columns(df_with_features)

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


