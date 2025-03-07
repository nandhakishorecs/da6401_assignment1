import numpy as np          # To handle vector / matrix operations 

# ------------------- Activation Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# used for any hidden layer in a neural network 
class sigmoid: 
    __slots__ = '_c', '_b'
    def __init__(self, c:float = 1, b:float  = 0) -> None:
        self._b = b
        self._c = c

    def value(self, X:np.ndarray) -> np.ndarray:
        obj = 1 + np.exp((-1) * self._c(X + self._b))
        return 1/obj

    def _derivative(self, X:np.ndarray, remove:bool = False) -> np.ndarray: 
        _value = self.value(X)
        if(remove == True): 
            _value = _value[: -1, :]
        return self._c * _value * (1 - _value)

# used for any hidden layer in a neural network 
class tanh: 
    def __init__(self) -> None:
        pass

    def value(self, X: np.ndarray) -> np.ndarray:
        numerator = np.exp(X) - np.exp((-1) * X)
        denominator = np.exp(X) + np.exp((-1) * X)
        return numerator/denominator

    def _derivative(self, X: np.ndarray) -> np.ndarray: 
        y = self.value(X)
        diff = 1 - (y**2)
        return diff

# used for any hidden layer in a neural network 
class relu: 
    def __init__(self) -> None:
        pass

    def value(self, X: np.ndarray) -> np.ndarray:
        r = X 
        r[r < 0] = 0 
        return r
    
    def _derivative(self, X: np.ndarray) -> np.ndarray: 
        diff = np.ones(X.shape)
        diff[diff < 0] = 0
        return diff

# used for final layer in a neural network 
class softmax: 
    def __init__(self) -> None:
        pass

    def value(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X)/np.sum(np.exp(X), axis = 0)

    def _derivative(self, X: np.ndarray) -> np.ndarray: 
        y = self.value(X)
        matrix = np.title(y, y.shape[0])
        diff = np.diag(y.reshape(-1, ) - (matrix * matrix.T))
        return diff