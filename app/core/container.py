from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import AppConfig
from .data_loader import DataRepository, PanelDataset
from .model_manager import ModelRegistry, PanelModel
from .recommendation import GroqRecommender


@dataclass
class ApplicationContainer:
    data_repository: DataRepository
    model_registry: ModelRegistry
    recommender: GroqRecommender

    @classmethod
    def build(cls) -> "ApplicationContainer":
        data_repo = DataRepository(AppConfig.DATASET_PATHS)
        datasets: Dict[str, PanelDataset] = {
            panel: data_repo.get_dataset(panel) for panel in data_repo.list_panels()
        }
        model_registry = ModelRegistry(datasets, AppConfig.MODEL_PATHS)
        recommender = GroqRecommender()
        return cls(
            data_repository=data_repo,
            model_registry=model_registry,
            recommender=recommender,
        )

    def get_best_model(self) -> PanelModel:
        return self.model_registry.best_model()


__all__ = ["ApplicationContainer"]


