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

# ------------------- A Complete Neural Network -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Basic skeleton - yet to be implemented
class NeuralNetwork: 
    def __init__(
            self, 
            layers: list, 
            batch_size: int, 
            optimiser: str, 
            initilaisation: str, 
            epochs: int, 
            target: np.ndarray, 
            loss_function: str, 
            validation_features: np.ndarray, 
            validation_target: np.ndarray, 
            optimiser_parameters = None
        ) -> None:
        pass
