import numpy as np          # To handle vector / matrix operations 

# ------------------- Loss Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Used for classification
class CategoricalCrossEntropy():
    def __init__(self) -> None:
        pass

    def _loss(self, true: np.ndarray, predicted: np.ndarray) -> float: 
        return (-1) * np.sum(
            np.sum(true * np.log(predicted))
        )

    def _derivative(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray: 
        return (-1) * (true / predicted)

# Used for regression
class MeanSquaredError(): 
    def __init__(self) -> None:
        pass

    def _loss(self, true: np.ndarray, predicted: np.ndarray) -> float:
        return np.sum((true - predicted)**2)

    def _derivative(self, true_batch, predicted_batch) : 
        return (-1) * (true_batch - predicted_batch)