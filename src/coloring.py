from colour import Color
import numpy as np
from typing import List
import matplotlib.pyplot as plt


def compute_categorical_coloring(f: np.ndarray, cmap: str = "tab20"):
    if cmap == "tab20":
        colormap = plt.cm.tab20.colors
    else:
        raise Exception(f"Colormap {cmap} not supported.")
    feature = np.array(f).astype(np.int64)
    colors = [colormap[p] for p in feature]
    return colors


def compute_numerical_coloring(
    f: np.ndarray, bins: int = 11, start_color: str = "red", end_color: str = "green"
) -> List:
    red = Color(start_color)
    colors = list(red.range_to(Color(end_color), bins))
    if f.min() == f.max():
        return [Color(start_color).rgb for r in f]
    else:
        min_ = f.min()
        feature = (f - min_) / (f.max() - min_)
        feature = np.floor(feature * (bins - 1)).astype(np.int64)
        colors = [colors[p].rgb for p in feature]
        return colors
