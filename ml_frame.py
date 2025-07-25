


from typing import Dict, Optional, List, Set, Union
import numpy as np
import os
import pandas as pd
import copy
from feature import Feature

class MLFrame:
    pass

class MLFrame:
    """
    The `MLFrame` is motivated by the problem that arrays are not naturally supported
    in `pd.DataFrame`, yet they are a quite common type, especially to be interpreted
    as embeddings.

    The `MLFrame` is a lightweight object to be considered
    as analagous to a `pd.DataFrame`, but in philosophy is to be seen as
    fundamentally an annotated set of data representations.

    """

    def __init__(self, data: Optional[Dict[str, np.ndarray]] = None):
        """Initializes object with data.

        If data is `None`, we create an empty `MLFrame`.
        """
        if data is not None and len(data) != 0:
            # Validation.
            min_length = float("inf")
            max_length = float("-inf")
            for key in data:
                min_length = min(min_length, len(data[key]))
                max_length = max(max_length, len(data[key]))
            if min_length != max_length:
                raise Exception("Difference in lengths. ")

            self.N = min_length
            self.univariate_columns = [col for col in data if len(data[col].shape) == 1]
            self.multivariate_columns = [
                col for col in data if len(data[col].shape) > 1
            ]
            self._data = data
        else:
            self._data = {}
            self.N = None
            self.univariate_columns = []
            self.multivariate_columns = []

    def to_dict(self) -> Dict:
        return copy.deepcopy(self._data)

    @property
    def columns(self) -> List[str]:
        return self.univariate_columns + self.multivariate_columns

    @property
    def targets(self) -> Optional[np.ndarray]:
        if "targets" in self._data:
            return self._data["targets"]
        return None

    def __getitem__(self, key: Union[str, List[str]]) -> Union[MLFrame, np.ndarray]:
        """Gets data associated with label(s) `key`"""
        if isinstance(key, str):
            return self._data[key]
        else:
            return MLFrame({k: self._data[k] for k in key})

    def __setitem__(self, key: str, data: np.ndarray):
        """Sets column with label `key` to be `data`."""
        if self.N is None or self.N == len(data):
            self._data[key] = data
            self.N = len(data)
            if len(data.shape) > 1:
                self.multivariate_columns.append(key)
            else:
                self.univariate_columns.append(key)
        else:
            raise Exception(f"len(data) == {len(data)} not {self.N}.")

    def get_features(
        self, categorical_set: List[str], only_tabular: bool = True
    ) -> Dict[str, Feature]:
        """Return Dictionary of Features.

        Args:
            categorical_set (List[str]): features to treat as categorical.

        Returns:
            features (Dict[str, Feature]): dictionary with featurized values.

        Notes:
            This feature exists to get features from raw data for visualization purposes.
        """
        data = self.get_tabular()._data if only_tabular else self._data
        return {key: Feature(self._data[key], key in categorical_set) for key in data}

    def get_tabular(self):
        """Return univariate columns as `MLFrame`"""
        d = {col: self._data[col] for col in self.univariate_columns}
        return MLFrame(d)

    def at(self, index: int) -> Dict:
        """Return values from each column at `index`."""
        return {col: self[col][index] for col in self.columns}

    def to_frame(self, path: str):
        """Save `MLFrame` to disk, splitting into a parquet and embeddding files.

        Args:
            path (str): File path to save to.

        """
        tabular_path = os.path.join(path, "tabular.parquet")
        if not os.path.exists(path):
            os.mkdir(path)

        df = self.as_tabular_data_frame()
        df.to_parquet(tabular_path)

        for key in self.multivariate_columns:
            emb_path = os.path.join(path, f"{key}.npy")
            np.save(emb_path, self._data[key])

    @staticmethod
    def load_frame(path: str) -> MLFrame:
        """Load from disk."""
        tabular_path = os.path.join(path, "tabular.parquet")
        df = pd.read_parquet(tabular_path)
        _data = {}
        for key in df:
            _data[key] = np.array(df[key])

        embedding_paths = [f for f in os.listdir(path) if f.endswith("npy")]

        for p in embedding_paths:
            _data[p[:-4]] = np.load(os.path.join(path, p))

        return MLFrame(_data)

    def get_arrays(self) -> MLFrame:
        """Return multivariate data."""
        d = {col: self._data[col] for col in self.multivariate_columns}
        return MLFrame(d)

    def subset(self, indices: List[int]):
        """Return frame only at indices."""
        d = {col: self._data[col][indices] for col in self._data}
        return MLFrame(d)

    def concatenate(self, other: MLFrame, axis=0) -> MLFrame:
        """Concatenation instance with another MLFrame."""
        if set(self.univariate_columns) != set(other.univariate_columns):
            raise Exception("Univariate columns don't match.")

        if set(self.multivariate_columns) != set(other.multivariate_columns):
            raise Exception("Multivariate columns don't match.")

        if axis == 1:
            raise NotImplementedError()

        d = {}
        for column in self.columns:
            d1 = self._data[column]
            d2 = other._data[column]
            combo = np.concatenate([d1, d2], axis=0)
            d[column] = combo

        return MLFrame(d)

    def __len__(self):
        """Return number of entities."""
        return self.N

    def as_tabular_data_frame(self) -> pd.DataFrame:
        """Return univariate columns only as DataFrame."""
        tabular = self.get_tabular()
        return pd.DataFrame(tabular._data)

    def __repr__(self) -> str:
        embedding_representation = {
            k: array.shape for k, array in self.get_arrays()._data.items()
        }
        return (
            "Tabular\n"
            + repr(self.as_tabular_data_frame())
            + "\nMultivariate\n"
            + repr(embedding_representation)
        )
    
    def grouped_by(self, sets: List[Set[int]]):
        subsets = []
        for s in sets:
            subsets.append(self.subset(list(s)))
        return subsets