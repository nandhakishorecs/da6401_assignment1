import numpy as np                  # for vector / matrix operations
from activations import *           # functions sigmoid, tanh, relu, softmax 
from initialisers import *          # Random Normal, Xavier 

# ------------------- Layers for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

activation_functions_map = {
    'Sigmoid': Sigmoid(), 
    'Tanh': Tanh(), 
    'ReLU': ReLU(),
    'Softmax': Softmax()
}

initialiser_map = {
    'Random_Normal': RandomInit(), 
    'Xavier': XavierInit()
}

# We have K layers - Kth layer is output, 1st layer is Input and we have L-1 Hidden layers 
# Input Layer 
class InputLayer:
    __slots__ = ['_input_data']
    
    def __init__(self, input_data) -> None:
        self._input_data = np.array(input_data)
    
    def __repr__(self) -> str:
        return "InputLayer"

class Dense:
    __slots__ = ['_weights', '_biases', '_activation', '_input_data', '_output_data']
    
    def __init__(self, input_size: int, output_size: int, activation: str, initializer) -> None:
        self._weights = initializer.initialize(input_size, output_size)
        self._biases = np.zeros((1, output_size))
        self._activation = activation_functions_map[activation]
        self._input_data = None
        self._output_data = None
    
    def __repr__(self) -> str:
        return "Dense"