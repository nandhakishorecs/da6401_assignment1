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
    'Vannial_GD' : VanilaGradientDescent(),
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
        self._initialisation = initialisation
        self._n_epochs = n_epochs 
        self._optimiser = optimiser
        self._target = target 
        self._n_batch = math.ceil(self._target.shape[1]/batch_size)
        self._loss_type = loss_function
        self._loss_function = map_loss_function[loss_function]
        self._log = wandb_log
        if(validation): 
            self._val_X = val_X
            self._layers[0]._a_val = val_X
            self._val_target = val_target
        self._init_parameters()

    def _init_parameters(self): 
        size_prev = self._layers[0]._size

        for layer in self._layers[1: ]: 
            
            layer._W_size = (layer._size, size_prev)
            size_prev = layer._size

            layer._W_optimiser = deepcopy(map_optimiser[self._optimiser])
            layer._b_optimiser = deepcopy(map_optimiser[self._optimiser])
        
        if(self._initialisation == 'RandomInit'):
            for layer in self._layers[1: ]: 
                layer._W = RandomInit().initialize(layer_size = layer._W_size) # np.random.normal(loc = 0, scale = 1, size = layer._W_size)
                layer._b = np.zeros((layer._W_size[0], 1))

        elif(self._initialisation == 'XavierInit'): 
            for layer in self._layers[1: ]: 
                layer._W = XavierInit().initialize(layer_size = layer._W_size) 
                layer._b = np.zeros((layer._W_size[0], 1))
    
    def forward_propagation(self):
        for i in range(1, len(self._layers)): 
            self._layers[i]._h = self._layers[i]._W @ self._layers[i-1]._a - self._layers[i]._b
            self._layers[i]._a = self._layers[i]._activation.value(self._layers[i]._h)
            self._layers[i]._h_val = self._layers[i]._W @ self._layers[i-1]._a_val - self._layers[i]._b
            self._layers[i]._a_val = self._layers[i]._activation.value(self._layers[i]._h_val)
        
        if(self._loss_type == 'CategoricalCrossEntropy'):
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
                # wandb logging 
                wandb.log({
                    'Step': epoch, 
                    'Training loss': train_loss_log[-1] / self._target.shape[1], 
                    'Training Accuracy': train_accuracy_log[-1] / self._target.shape[1], 
                    'Validation Loss': val_loss_log[-1] / self._val_target.shape[1], 
                    'Validation Accuracy': val_accuracy_log[-1] / self._val_target.shape[1]
                })

            for batch in range(self._n_batch):     
                target_batch = self._target[:, batch * self._batch_size: (batch + 1) * self._batch_size]
                y_batch = self._layers[-1]._y[:, batch * self._batch_size: (batch + 1) * self._batch_size]

                # Parameter anealing 
                # try: 
                #     if(train_loss_log[-1] > train_loss_log[-2]):
                #         for layer in self._layers[1: ]: 
                #             pass
                #     pass
                # except: 
                #     pass

                if(flag): 
                    break

                self._layers[-1]._a_grad = self._loss_function.derivative(target_batch, y_batch)
                self._layers[-1]._h_grad = self._layers[-1]._a_grad * self._layers[-1]._activation._derivative(self._layers[-1]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size])

                self._layers[-1]._W_grad = self._layers[-1]._h_grad @ self._layers[-2]._a[:, batch * self._batch_size: (batch + 1) * self._batch_size].T
                self._layers[-1]._W_update = self._layers[-1]._W_optimiser.update(self._layers[-1]._W_grad)

                self._layers[-1]._b_grad = (-1) * np.sum(self._layers[-1]._h_grad, axis = 1).reshape(-1, 1)
                self._layers[-1]._b_update = self._layers[-1]._b_optimiser.update(self._layers[-1]._b_grad)

                assert self._layers[-1]._W_update.shape == self._layers[-1]._W.shape, 'Size Mismatch'

                for i in range(len(self._layers[: -2]), 0, -1): 
                    self._layers[i]._a_grad = self._layers[i + 1]._W.T @ self._layers[i + 1]._h_grad
                    self._layers[i]._h_grad = self._layers[i]._a_grad * self._layers[i]._activation._derivative(self._layers[i]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size])

                    self._layers[i]._b_grad = (-1) * np.sum(self._layers[i]._h_grad, axis = 1).reshape(-1, 1)
                    self._layers[i]._W_grad = self._layers[i]._a_grad * self._layers[i]._activation._derivative(self._layers[i]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size])
                    
                    self._layers[i]._W_update = self._layers[i]._W_optimiser.update(self._layers[i]._W_grad)
                    self._layers[i]._b_update = self._layers[i]._b_optimiser.update(self._layers[i]._b_grad)
                    
                for _, layer in enumerate(self._layers[1: ]): 
                    layer._W = layer._W - layer._W_update
                    layer._b = layer._b - layer._b_update

                self.forward_propagation()
            
            if(flag): 
                break

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