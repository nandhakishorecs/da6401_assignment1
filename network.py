import math 
import numpy as np                  
import wandb
from tqdm import tqdm 
from copy import deepcopy

from initialisers import * 
from activations import * 
from loss_functions import *
from preprocessing import * 
from optimisers import * 
from layers import *
from metrics import * 

# ------------------- A Complete Neural Network -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Basic skeleton - yet to be implemented
class NeuralNetwork: 
    __slots__ = '_layers', '_batch_size', '_optimiser', '_target', '_init', '_n_epochs', '_validation', '_X_val', '_y_val', '_log', '_loss_type', '_loss_function', '_a', '_optimised_parameters'
    def __init__(
            self, layers: list, batch_size: int, optimiser: str, initilaisation: str, n_epochs: int, 
            target: np.ndarray, 
            loss_function: str, 
            validation: bool = True,
            validation_features: np.ndarray = None, 
            validation_target: np.ndarray = None, 
            # learning rate
            optimised_parameters: list = None
        ) -> None:
        self._layers = layers
        self._batch_size = batch_size
        self._init = initilaisation
        self._optimiser = optimiser
        self._target = target
        if(self._validation == validation): 
            self._X_val = validation_features
            self._y_val = validation_target
            self.layers[0]._a = validation_features
        self._X_val = validation_features
        # self._loss_function = loss_functions_map[loss_function]
        self._loss_type = loss_function
        self._n_epochs = n_epochs
        self._optimised_parameters = optimised_parameters
        self._init_parameters()

    # initialising parameters
    def _init_parameters(self):
        previous_layer_size = self._layers[0].size
        for layer in self._layers[1: ]:
            layer.W_size = (layer.size, previous_layer_size)
            previous_layer_size = layer.size
            layer.W_optimiser = deepcopy(self._loss_function)
            layer.b_optimiser = deepcopy(self._loss_function)
            if(self._optimiser is not None): 
                layer.W_optimiser._set_parameters(self._optimised_parameters)
                layer.b_optimiser._set_parameters(self._optimised_parameters)
        pass
    
    
    # Python special function to describe the neural network with given configuration
    def __str__(self): 
        name = '\nNeural Network\n Configuration:\n'
        for layer in self._layers: 
            temp = str(layer) + ' '
        temp += '\n'
        epochs = f'Epochs:\t{self._n_epochs}\n'
        loss = f'Loss function:\t{self._loss_function}\n'
        batch_size = f'Batch size:\t{self._batch_size}\n'
        optimiser = f'Optimiser:\t{self._optimiser}\n'
        init = f'Initialisation type:\t{self._init}\n'
        return name + temp + epochs + loss + batch_size + optimiser + init
