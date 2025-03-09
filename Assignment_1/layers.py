import numpy as np
from activations import *
from initialisers import *

# ------------------- Layers for Neural Networks ---------------------------
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

# We have K layers - Kth layer is output, 1st layer is Input and we have L-1 Hidden layers 
# Input Layer 
class Input:
    def __init__(self, input_data: np.ndarray) -> None:
        self._input_data = input_data
        self._size = self._input_data.shape[0]
        self._a = self._input_data
    
    def __repr__(self) -> str:
        return f"Input Layer with size {self._size}\n"

# Dense Layer 
class Dense:
    def __init__(self, name: str, layer_size: int, activation: str = 'ReLU') -> None:
        self._name = name
        self._activation = activation_functions_map[activation]
        self._size = layer_size

    def __repr__(self) -> str:
        if(self._name == 'Last_Layer'):
            return f"{self._name} | Dense Layer with size: {self._size} | Activation: Softmax\n" 
        return f"Layer name: {self._name} | Dense Layer with size: {self._size} | Activation: {self._activation}\n"
    
# COMPLETED