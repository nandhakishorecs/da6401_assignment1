import numpy as np
from activations import *
from initialisers import *

# ------------------- Layers for Neural Networks -----------------------------------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
#   
#   This file contains the base classes for layers of a Artifical Neural Network 
#   This file has dependenices from other files to use the activation functions 
#
#   There are two types: 
#       - Input: takes the data and send it's to a fully connected layer, no activations 
#       - Dense: takes the ouput of the input layer and sends it to a next dense layer, 
#                if it's last layer , uses Softmax actiation, else uses other activation functions from the other files 
#       - Each layer has a __repr__ function to print their structure. Use print() to see it. 
# ----------------------------------------------------------------------------------------------------------------------------

activation_functions_map = {
    'sigmoid': Sigmoid(), 
    'tanh': Tanh(), 
    'relu': ReLU(),
    'softmax': Softmax(),
    'identity': Identity()
}

# We have K layers - Kth layer is output, 1st layer is Input and we have L-1 Hidden layers 
# Input Layer 
class Input:
    __slots__ = '_input_data', '_size', '_a', '_a_val', '_a_test'
    def __init__(self, input_data: np.ndarray) -> None:
        self._input_data = input_data
        self._size = self._input_data.shape[0]
        self._a = self._input_data
    
    def __repr__(self) -> str:
        return f'Input Layer | Size: {self._size}'

# Dense Layer 
class Dense:
    __slots__ = '_name', '_activation', '_size', '_W', '_W_size', '_W_optimiser', '_W_grad', '_W_update', '_b', '_b_optimiser', '_b_grad', '_b_update','_h', '_h_val', '_h_grad', '_h_test', '_a', '_a_grad', '_a_val', '_a_test','_y', '_y_val', '_y_test'
    def __init__(self, name: str, layer_size: int, activation: str = 'relu') -> None:
        self._name = name
        self._activation = activation_functions_map[activation]
        self._size = layer_size

    def __repr__(self) -> str:
        if(self._name == 'Last_Layer'):
            return f'''
        \t\t Layer name: {self._name} | Dense layer | Size: {self._size} | Activation: Softmax'''
        return f'''
        \t\t Layer name: {self._name} | Dense Layer | Size: {self._size} | Activation: {self._activation}'''
    
# COMPLETED