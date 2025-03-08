import numpy as np          # To handle vector / matrix operations 

# ------------------- Loss Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Used for classification
class CategoricalCrossEntropy:
    
    def __init__(self) -> None:
        pass 

    def value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return (-1) * np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (-1) * y_true / y_pred
        
    def __repr__(self) -> str:
        return "CategoricalCrossEntropy"

# Used for regression
class MeanSquaredError:
    def __init__(self) -> None:
        pass
    
    def value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred) / y_true.shape[0]
    
    def __repr__(self) -> str:
        return "MeanSquaredError"