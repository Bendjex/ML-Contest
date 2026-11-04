"""Single decision tree on the baseline feature vector (see tree_train.py)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from baseline_model import features
from competition_api import Sample

MODEL_FILE = Path(__file__).with_name("tree_model.pkl")


class Model:
    def __init__(self):
        with open(MODEL_FILE, "rb") as fh:
            self.tree = pickle.load(fh)

    def predict(self, x: Sample) -> int:
        return int(self.tree.predict(features(x)[np.newaxis, :])[0])