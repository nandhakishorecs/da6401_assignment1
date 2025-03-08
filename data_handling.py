import numpy as np          # To handle vector / matrix operations 
# ------------------- Label Encoder - assign number to string labels ----------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# One way to encode labels, but the model is more versatile with one hot encoding 
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
    
class OneHotEncoder:
    def __init__(self) -> None:
        pass
    
    def fit(self, y: np.ndarray, n_class: int) -> None:
        self._y = y 
        self._n_class = n_class

    def transform(self) -> np.ndarray:
        transformed = np.zeros((self._n_class, self._y.size))
        for column, row in enumerate(self._y):
            transformed[row, column] = 1
        return transformed

    def fit_transform(self, y: np.ndarray, n_class: int) -> np.ndarray:
        self.fit(y, n_class)
        return self.transform()

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        # Assumes direct correation between the position and class number
        y_class = np.argmax(y, axis=0)
        return y_class

class MinMaxScaler: 
    def __init__(self) -> None:
        pass

    def fit(self, array: np.ndarray) -> None: 
        self._min = np.min(array, axis = 0)
        self._max = np.max(array, axis = 0)

    def transform(self, array: np.ndarray) -> np.ndarray: 
        self.fit(array)
        transformed = (array - self._min) / (self._max - self._min)
        return transformed 

    def fit_transform(self, array: np.ndarray) -> np.ndarray: 
        self.fit(array)
        return self.transform(array)
    
# COMPLETED