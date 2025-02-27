import numpy as np          # To handle vector / matrix operations 
# ------------------- Label Encoder - assign number to string labels ----------------------------

class LabelEncoder:
    __slots__ = '_label_to_num', '_num_to_label', '_classes_'
    def __init__(self) -> None:
        self._label_to_num = {}
        self._num_to_label = {}
        self._classes_ = []
        return None

    def fit(self, labels:np.ndarray) -> None:
        unique_labels = set(labels)
        self._classes_ = sorted(unique_labels)
        # Map class names to numbers from 0 - n-1 for 'n' classes
        self._label_to_num = {label: index for index, label in enumerate(self._classes_)}
        self._num_to_label = {index: label for index, label in enumerate(self._classes_)}
        return self

    def transform(self, labels:np.ndarray) -> np.ndarray:
        return np.array([self._label_to_num[label] for label in labels])

    def fit_transform(self, labels:np.ndarray) -> np.ndarray:
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, labels:np.ndarray) -> np.ndarray:
        return np.array([self._num_to_label[idx] for idx in labels])    