"""Guitar make/model classifier.

Classifies an electric-guitar listing image into one of the trained make/model
classes. The current release is a 2-class set:
    stratocaster | telecaster

Public API:
    build_transforms, build_splits, make_dataloaders   (strat_classifier.data)
    build_model                                         (strat_classifier.model)
    load_model, predict                                 (strat_classifier.inference)

The authoritative class list for a trained model is ``models/class_names.json``
(written at train time); ``CLASSES`` below is just the current default set.
"""

from strat_classifier.data import build_splits, build_transforms, make_dataloaders
from strat_classifier.inference import load_model, predict
from strat_classifier.model import build_model

__version__ = "0.1.0"

CLASSES = ["stratocaster", "telecaster"]  # alphabetical — matches ImageFolder ordering

__all__ = [
    "build_transforms",
    "build_splits",
    "make_dataloaders",
    "build_model",
    "load_model",
    "predict",
    "CLASSES",
    "__version__",
]
