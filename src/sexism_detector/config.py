"""
Central configuration for the sexism detector project.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# EDOS dataset files
EDOS_AGGREGATED_PATH = RAW_DATA_DIR / "edos_labelled_aggregated.csv"
EDOS_CLEANED_PATH = RAW_DATA_DIR / "edos_cleaned.csv"
EDOS_INDIVIDUAL_PATH = RAW_DATA_DIR / "edos_labelled_individual_annotations.csv"


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    remove_emojis: bool = True
    translate_emojis: str | None = None  # "en" | "alias" | None
    remove_urls: bool = True
    remove_users: bool = True
    lowercase: bool = False  # Keep False for BERT-based models
    remove_punctuation: bool = False
    min_length: int = 3
    max_length: int = 512


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_name: str = "FacebookAI/roberta-large"
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 512
    warmup_steps: int = 500


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True  # Development mode


# Category labels (from your original research)
CATEGORY_LABELS: List[str] = [
    "threat",       # Threats, plans to harm and incitement
    "derogation",   # Derogation
    "animosity",    # Animosity
    "prejudiced"    # Prejudiced discussions
]

# Binary labels
BINARY_LABELS: List[str] = ["not_sexist", "sexist"]

# Label mappings
LABEL_TO_ID = {label: idx for idx, label in enumerate(CATEGORY_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
