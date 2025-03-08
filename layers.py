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
class Input:
    def __init__(self, input_data) -> None:
        self._input_data = np.array(input_data)
        self._size = self._input_data.shape[0]
        self._a = self._input_data
    
    def __repr__(self) -> str:
        return f"Input Layer with size {self._size}"

# Dense Layer 
class Dense:
    def __init__(self, layer_size: int, activation: str, LastLayer: bool = False) -> None:
        self._activation = activation_functions_map[activation]
        self._size = layer_size

    def __repr__(self) -> str:
        return f"Dense Layer with size: {self._size} | Activation: {self._activation}"