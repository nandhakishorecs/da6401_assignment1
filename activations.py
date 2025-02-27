import numpy as np          # To handle vector / matrix operations 

# ------------------- Activation Functions for Neural Networks -------------------

# used for any hidden layer in a neural network 
class sigmoid: 
    __slots__ = ''
    def __init__(self, x: np.ndarray) -> None:
        pass

    def value(self) -> np.ndarray:
        pass

    def derivative(self) -> np.ndarray: 
        pass

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