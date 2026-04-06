"""
Water Potability ML Training Module
Training scripts for machine learning models predicting drinking water potability.
"""

__version__ = "2.0.0"
__author__ = "ML Team"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODELS_DIR = PACKAGE_ROOT / "models"
LOGS_DIR = PACKAGE_ROOT / "logs"

__all__ = [
    "PACKAGE_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
]
