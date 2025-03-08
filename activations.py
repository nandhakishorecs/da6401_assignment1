import numpy as np          # To handle vector / matrix operations 

# ------------------- Activation Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# used for any hidden layer in a neural network 
class Sigmoid:
    def __init__(self, c: float = 1, b: float= 0) -> None:
        self._c = c
        self._b = b
    
    def value(self, X: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-self._c * (X + self._b)))
    
    def _derivative(self, X: np.ndarray) -> np.ndarray:
        sigmoid_x = self.value(X)
        return self._c * sigmoid_x * (1 - sigmoid_x)
    
    def __repr__(self) -> str:
        return "Sigmoid"

# used for any hidden layer in a neural network 
class Tanh:
    def __init__(self) -> None:
        pass
    
    def value(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X)
    
    def _derivative(self, X: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(X) ** 2
    
    def __repr__(self) -> str:
        return "Tanh"
    
# used for any hidden layer in a neural network 
class ReLU:
    def __init__(self) -> None:
        pass
    
    def value(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)
    
    def _derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X> 0, 1, 0)
    
    def __repr__(self) -> str:
        return "ReLU"

# used for the last layer in a neural network 
class Softmax:
    def __init__(self) -> None:
        pass
    
    def value(self, X: np.ndarray) -> np.ndarray:
        # exp_x = np.exp(X - np.max(X))  # Prevent overflow
        # return exp_x / np.sum(exp_x)
        val = np.exp(X) / np.sum(np.exp(X), axis = 0)
        return val
        
    
    def _derivative(self, X: np.ndarray) -> np.ndarray:
        # softmax_x = self.value(X).reshape(-1, 1)
        # return np.diagflat(softmax_x) - np.dot(softmax_x, softmax_x.T)
        y = self.value(X)
        mat = np.tile(y, y.shape[0])
        val = np.diag(y.reshape(-1,)) - (mat*mat.T)
        return val
        
    
    def __repr__(self) -> str:
        return "Softmax"