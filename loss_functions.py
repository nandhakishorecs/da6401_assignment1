import numpy as np          # To handle vector / matrix operations 

# ------------------- Loss Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Used for classification
class CategoricalCrossEntropy():
    __slots__ = '_true', '_predicted'
    def __init__(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        self._true = y 
        self._predicted = y_pred

    def _loss(self) -> float: 
        return (-1) * np.sum(
            np.sum(self._true * np.log(self._predicted))
        )

    def _derivative(self) -> np.ndarray: 
        return (-1) * (self._true / (self._predicted))

# Used for regression
class MeanSquaredError(): 
    __slots__ = '_true', '_predicted', '_t_batch', '_predicted_batch'
    def __init__(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        self._true = y 
        self._predicted = y_pred

    def _loss(self) -> float:
        return np.sum((self._true - self._predicted)**2)

    def _derivative(self, true_batch, predicted_batch) -> np.ndarray: 
        self._t_batch = true_batch
        self._predicted_batch = predicted_batch
        return (-1) * (self._t_batch - self._predicted_batch)