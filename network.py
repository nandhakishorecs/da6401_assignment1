import math 
import numpy as np                  
import wandb
from tqdm import tqdm 
from copy import deepcopy
import tensorflow as tf

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

# optimiser map 
map_optimiser = {
    'Vannial_GD' : VannilaGradientDescent(),
    'Momentum_GD' : MomentumGD(), 
    'Nestorov': NesterovMomentumGD(), 
    'AdaGrad': AdaGrad(), 
    'RMSProp': RMSProp(),
    'AdaDelta': AdaDelta(),
    'Adam': Adam() 
}

# loss function map 
map_loss_function = {
    'CategoricalCrossEntropy' : CategoricalCrossEntropy(), 
    'MeanSquaredError' : MeanSquaredError()
}

# inictialiser map 
map_initialiser = {
    'RandomInit' : RandomInit(), 
    'Xavier': XavierInit()
}

# Encoder 
encoder = OneHotEncoder()

# Classification metrics calculator 
metrics = Metrics()

# Basic skeleton - partially implemented
class NeuralNetwork: 
    def __init__(self, layers, batch_size, optimiser, n_epochs, target, loss_function, initialisation: str,  validation:bool = False, val_X = None, val_target = None, wandb_log = False) -> None: 
        self._layers = layers
        self._batch_size = batch_size
        self._optimiser = optimiser
        self._n_epochs = n_epochs 
        self._target = target 
        self._loss_function = map_loss_function[loss_function]
        self._initialisation = initialisation
        self._n_batch = math.ceil(self._target.shape[1]/batch_size)
        if(validation): 
            self._val_X = val_X
            self._val_target = val_target
            self._layers[0]._a_val = val_X
        self._log = wandb_log
        self._init_parameters()
        pass

    def _init_parameters(self): 
        size_prev = self._layers[0]._size

        for layer in self._layers[1: ]: 
            
            layer._W_size = (layer._size, size_prev)
            size_prev = layer._size

            layer._W_optimiser = deepcopy(map_optimiser[self._optimiser])
            layer._b_optimiser = deepcopy(map_optimiser[self._optimiser])
        
        if(self._initialisation == 'RandomInit'):
            print('YES!') 
            for layer in self._layers[1: ]: 
                layer._W = RandomInit().initialize(layer_size = layer._W_size)
                layer._b = np.zeros((layer._W_size[0], 1))

        elif(self._initialisation == 'XavierInit'): 
            print('YES!') 
            for layer in self._layers[1: ]: 
                layer._W = XavierInit().initialize(layer_size = layer._W_size)
                layer._b = np.zeros((layer._W_size[0], 1))
    
    def forward_propagation(self):
        for i in range(1, len(self._layers)): 
            self._layers[i]._h = self._layers[i]._W @ self._layers[i-1]._a - self._layers[i]._b
            self._layers[i]._a = self._layers[i]._activation.value(self._layers[i]._h)
            self._layers[i]._h_val = self._layers[i]._W @ self._layers[i-1]._a_val - self._layers[i]._b
            self._layers[i]._a_val = self._layers[i]._activation.value(self._layers[i]._h_val)
        
        if(self._loss_function == 'CategoricalCrossEntropy'): 
            self._layers[-1]._y = Softmax().value(self._layers[-1]._a)
            self._layers[-1]._y_val = Softmax().value(self._layers[-1]._a_val)
        else: 
            self._layers[-1]._y = self._layers[-1]._a
            self._layers[-1]._y_val = self._layers[-1]._a_val

    def backward_propagation(self, verbose: bool = False): 
        lr_log = []
        train_loss_log = []
        train_accuracy_log = []

        val_loss_log = []
        val_accuracy_log = []

        flag = False

        for epoch in tqdm(range(self._n_epochs)):
            lr_log.append(self._layers[-1]._W_optimiser._lr)
            
            train_loss_log.append(self._loss_function.value(self._target, self._layers[-1]._y))
            val_loss_log.append(self._loss_function.value(self._val_target, self._layers[-1]._y_val))
        
            training_accuracy, validation_accuracy = self._get_accuracy(validation = True)
            train_accuracy_log.append(training_accuracy)
            val_accuracy_log.append(validation_accuracy)

            if(self._log): 
                # wandb
                pass
            
            for batch in range(self._n_batch):     
                pass
        pass

    def _get_accuracy(self, validation: bool = False, verbose: bool = False): 
        train_t = encoder.inverse_transform(self._target)
        train_y = encoder.inverse_transform(self._layers[-1]._y)
        training_accuracy = metrics.accuracy_score(train_t, train_y)

        if(validation): 
            val_t = encoder.inverse_transform(self._val_target)
            val_y = encoder.inverse_transform(self._layers[-1]._y_val)
            validation_accuracy = metrics.accuracy_score(val_t, val_y)
            if(verbose):
                print(f'Validation accuracy:\t{validation_accuracy}')
            return training_accuracy, validation_accuracy
        
        if(verbose):
            print(f'Training accuracy:\t{training_accuracy}')

        return training_accuracy