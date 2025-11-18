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

        # Pastikan setiap model memiliki pasangan dataset yang konsisten
        model_datasets: Dict[str, PanelDataset] = {}
        fallback_dataset = next(iter(datasets.values())) if datasets else None
        for model_panel in AppConfig.MODEL_PATHS.keys():
            if model_panel in datasets:
                model_datasets[model_panel] = datasets[model_panel]
            elif fallback_dataset:
                model_datasets[model_panel] = fallback_dataset

        model_registry = ModelRegistry(model_datasets, AppConfig.MODEL_PATHS)
        recommender = GroqRecommender()
        return cls(
            data_repository=data_repo,
            model_registry=model_registry,
            recommender=recommender,
        )

    def get_best_model(self) -> PanelModel:
        return self.model_registry.best_model()


__all__ = ["ApplicationContainer"]


