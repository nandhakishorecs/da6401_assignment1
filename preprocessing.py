import numpy as np          # To handle vector / matrix operations 
# ------------------- Label Encoder - assign number to string labels ----------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

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
    
class MinMaxScaler: 
    __slots__ = '_array', '_min', '_max'
    def __init__(self, X:np.ndarray) -> None:
        self._array = X 

    def fit(self): 
        self._min = np.min(self._array, axis = 0)
        self._max = np.max(self._array, axis = 0)

    def transform(self): 
        return (self._array - self._min) / (self._max - self._min)

    def fit_transform(self): 
        self.fit()
        return self.fit_transform()