"""Fender Stratocaster origin classifier.

Classifies a Stratocaster listing image as one of:
    american | japanese | mexican

Public API:
    build_transforms, build_splits, make_dataloaders   (strat_classifier.data)
    build_model                                         (strat_classifier.model)
    load_model, predict                                 (strat_classifier.inference)
"""

from strat_classifier.data import build_splits, build_transforms, make_dataloaders
from strat_classifier.model import build_model

__version__ = "0.1.0"

CLASSES = ["american", "japanese", "mexican"]  # alphabetical — matches ImageFolder ordering

__all__ = [
    "build_transforms",
    "build_splits",
    "make_dataloaders",
    "build_model",
    "CLASSES",
    "__version__",
]
