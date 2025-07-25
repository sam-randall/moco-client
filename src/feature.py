from src.coloring import compute_categorical_coloring, compute_numerical_coloring
from typing import List
import numpy as np


class Feature:
    def __init__(self, values: np.ndarray, is_categorical: bool):
        self.values = values
        self.is_categorical = is_categorical

    def as_coloring(self, to_hex: bool = True) -> List:
        return (
            compute_categorical_coloring(self.values)
            if self.is_categorical
            else compute_numerical_coloring(self.values, to_hex=to_hex)
        )

    def to_dict(self):
        return {"values": self.values.tolist(), "is_categorical": self.is_categorical}
