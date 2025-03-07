import numpy as np          # To handle vector / matrix operations 

# ------------------- Loss Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Used for classification
class CategoricalCrossEntropy:
    __slots__ = ['y_true', 'y_pred']
    
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
    
    def value(self):
        epsilon = 1e-12
        y_pred = np.clip(self.y_pred, epsilon, 1. - epsilon)
        return -np.sum(self.y_true * np.log(y_pred)) / self.y_true.shape[0]
    
    def _derivative(self):
        return -self.y_true / self.y_pred
    
    def __repr__(self):
        return "CategoricalCrossEntropy"

# Used for regression
class MeanSquaredError:
    __slots__ = ['y_true', 'y_pred']
    
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
    
    def value(self):
        return np.mean((self.y_true - self.y_pred) ** 2)
    
    def _derivative(self):
        return -2 * (self.y_true - self.y_pred) / self.y_true.shape[0]
    
    def __repr__(self):
        return "MeanSquaredError"