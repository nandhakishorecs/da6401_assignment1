import numpy as np                  # for vector / matrix operations
from activations import *           # functions sigmoid, tanh, relu, softmax 

# ------------------- Layers for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

activation_functions_map = {
    'Sigmoid': sigmoid(), 
    'Tanh': tanh(), 
    'ReLU': relu(), 
    'Softmax': softmax()
}

# We have K layers - Kth layer is output, 1st layer is Input and we have L-1 Hidden layers 
# Input Layer 
class Input: 
    # __slots__ = '_name', '_input', '_size', '_a'
    def __init__(self, X:np.ndarray) -> None:
        self._name = 'Input',
        self._input = X, 
        self._size = X.shape[0]
        self._activation_val = self._input                # activated value

    # python special function to give a string representavle name to the class
    def __repr__(self) -> str: 
        return  self.__class__.__name__ + 'of size:' + str(self._size)

# Hidden Layers 
class Dense: 
    # __slots__ = '_size', '_activation', '_activation_func_name', '_name'
    def __init__(self, size:int, activation: str, name: str, isLastLayer:bool = False) -> None:
        self._name = name 
        self._size = size 
        self._activation = activation_functions_map[activation]
        self._activation_func_name = activation

    def __repr__(self) -> str: 
        return self.__class__.__name__ + 'of size:' + str(self._size) + '|' + self._activation_func_name + 'activation'