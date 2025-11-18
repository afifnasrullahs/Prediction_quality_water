from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


# Tentukan ROOT_DIR project (folder utama aplikasi)
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

# Load environment variables dari file `.env` di ROOT_DIR (jika ada)
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)


class AppConfig:
    """Central place to store application-wide configuration."""

    FEATURE_COLUMNS: List[str] = ["flow1", "flow2", "turbidity", "tds", "ph"]
    LABEL_COLUMN: str = "quality_label"
    ALLOWED_LABELS: List[str] = ["Biru", "Coklat", "Orange", "putih"]
    DISPLAY_LABELS: Dict[str, str] = {
        "Biru": "Biru",
        "Coklat": "Cokelat",
        "Orange": "Orange",
        "putih": "Putih",
    }

    # LLM / Groq
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"
    GROQ_API_KEY_ENV: str = "GROQ_API_KEY"

    # Paths (dataset dan model)
    ROOT_DIR: Path = ROOT_DIR
    DATASET_PATHS: Dict[str, Path] = {
        "panelA": ROOT_DIR / "Dataset" / "processed" / "panelAs_labeled.csv",
        "panelB": ROOT_DIR / "Dataset" / "processed" / "panelBs_labeled.csv",
        # Model overall memakai dataset Panel B (hasil terbaik)
        "overall": ROOT_DIR / "Dataset" / "processed" / "panelBs_labeled.csv",
    }
    MODEL_PATHS: Dict[str, Dict[str, Path]] = {
        "overall": {
            "model": ROOT_DIR / "Model" / "best_model_overall.pkl",
            "scaler": ROOT_DIR / "Model" / "scaler_overall.pkl",
            "encoder": ROOT_DIR / "Model" / "label_encoder_overall.pkl",
            "metadata": ROOT_DIR / "Model" / "model_metadata_overall.pkl",
        },
    }

    MODEL_PERFORMANCE: Dict[str, Dict[str, Dict[str, float]]] = {
        "panelA": {
            "Logistic Regression": {"train": 0.8317, "test": 0.8044},
            "Random Forest": {"train": 1.0, "test": 1.0},
            "XGBoost": {"train": 0.9983, "test": 0.9981},
        },
        "panelB": {
            "Logistic Regression": {"train": 0.8416, "test": 0.9023},
            "Random Forest": {"train": 0.9440, "test": 0.9547},
            "XGBoost": {"train": 0.9347, "test": 0.9535},
        },
    }

    CONFUSION_MATRIX_IMAGES: Dict[str, Dict[str, Path]] = {
        "panelA": {
            "Logistic Regression": ROOT_DIR / "Model" / "cm_Panel_A_Logistic_Regression.png",
            "Random Forest": ROOT_DIR / "Model" / "cm_Panel_A_Random_Forest.png",
            "XGBoost": ROOT_DIR / "Model" / "cm_Panel_A_XGBoost.png",
        },
        "panelB": {
            "Logistic Regression": ROOT_DIR / "Model" / "cm_Panel_B_Logistic_Regression.png",
            "Random Forest": ROOT_DIR / "Model" / "cm_Panel_B_Random_Forest.png",
            "XGBoost": ROOT_DIR / "Model" / "cm_Panel_B_XGBoost.png",
        },
    }


__all__ = ["AppConfig"]
