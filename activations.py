import numpy as np          # To handle vector / matrix operations 

# ------------------- Activation Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# used for any hidden layer in a neural network 
class sigmoid: 
    __slots__ = '_x', '_c', '_b'
    def __init__(self, x: np.ndarray, c:float = 1, b:float  = 0) -> None:
        self._x = x
        self._b = b
        self._c = c

    def value(self) -> np.ndarray:
        obj = 1 + np.exp((-1) * self._c(self._x + self._b))
        return 1/obj

    def derivative(self, remove:bool = False) -> np.ndarray: 
        _value = self.value(self._x)
        if(remove == True): 
            _value = _value[: -1, :]
        return self._c * _value * (1 - _value)

# used for any hidden layer in a neural network 
class tanh: 
    __slots__ = '_x'
    def __init__(self, x: np.ndarray) -> None:
        self._x = x 

    def value(self) -> np.ndarray:
        numerator = np.exp(self._x) - np.exp((-1) * self._x)
        denominator = np.exp(self._x) + np.exp((-1) * self._x)
        return numerator/denominator

    def derivative(self) -> np.ndarray: 
        y = self.value(self._x)
        diff = 1 - (y**2)
        return diff

# used for any hidden layer in a neural network 
class relu: 
    __slots__ = '_x'
    def __init__(self, x: np.ndarray) -> None:
        self._x = x

    def value(self) -> np.ndarray:
        r = self._x 
        r[r < 0] = 0 
        return r
    
    def derivative(self) -> np.ndarray: 
        diff = np.ones(self._x.shape)
        diff[diff < 0] = 0
        return diff

# used for final layer in a neural network 
class softmax: 
    __slots__ = '_x'
    def __init__(self, x: np.ndarray) -> None:
        self._x = x 

    def value(self) -> np.ndarray:
        return np.exp(self._x)/np.sum(np.exp(self._x), axis = 0)

    def derivative(self) -> np.ndarray: 
        y = self.value(self._x)
        matrix = np.title(y, y.shape[0])
        diff = np.diag(y.reshape(-1, ) - (matrix * matrix.T))
        return diff