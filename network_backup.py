import math
import numpy as np
import wandb
from tqdm import tqdm
from copy import deepcopy

from initialisers import *
from activations import *
from loss_functions import *
from data_handling import *
from optimisers import *
from layers import *
from metrics import *

# ------------------- A Complete Neural Network ----------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# optimiser map 
map_optimiser = {
    'SGD' : GradientDescent(),
    'Momentum_GD' : MomentumGD(), 
    'Nestorov': NesterovMomentumGD(), 
    'AdaGrad': AdaGrad(), 
    'RMSProp': RMSProp(),
    'AdaDelta': AdaDelta(),
    'Adam': Adam(), 
    'Nadam': Nadam(), 
    'Eve': Eve()
}

# loss function map 
map_loss_function = {
    'CategoricalCrossEntropy' : CategoricalCrossEntropy(), 
    'MeanSquaredError' : MeanSquaredError()
}

# inictialiser map 
map_initialiser = {
    'RandomInit' : RandomInit(), 
    'Xavier': XavierInit(), 
    'HeInit': HeInit()
}

# Encoder 
encoder = OneHotEncoder()

# Classification metrics calculator 
metrics = Metrics()

# Basic skeleton 
class NeuralNetwork: 
    def __init__(self, layers: list, batch_size: int, optimiser: str, n_epochs: int, target: np.ndarray, loss_function: str, initialisation: str,  validation:bool = False, val_X: np.ndarray = None, val_target: np.ndarray = None, wandb_log: bool = False, verbose: bool = False, optimised_parameters = None) -> None: 
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
        self._verbose = verbose
        self._validation = validation
        if(self._validation):
            self._val_X = val_X
            self._layers[0]._a_val = val_X
            self._val_target = val_target
        self._init_parameters()

    def _init_parameters(self) -> None: 
        size_prev = self._layers[0]._size

        for layer in self._layers[1: ]: 
            layer._W_size = (layer._size, size_prev)
            size_prev = layer._size

            layer._W_optimiser = deepcopy(map_optimiser[self._optimiser])
            layer._b_optimiser = deepcopy(map_optimiser[self._optimiser])

        if(self._initialisation == 'RandomInit'):
            for layer in self._layers[1: ]: 
                layer._W = RandomInit().initialize(layer_size = layer._W_size) 
                # layer._W = np.random.normal(loc = 0, scale = 1, size = layer._W_size)
                layer._b = np.zeros((layer._W_size[0], 1))

        elif(self._initialisation == 'XavierInit'): 
            for layer in self._layers[1: ]: 
                layer._W = XavierInit().initialize(layer_size = layer._W_size) 
                layer._b = np.zeros((layer._W_size[0], 1))

        elif(self._initialisation == 'HeInit'): 
            for layer in self._layers[1: ]: 
                layer._W = HeInit().initialize(layer_size = layer._W_size) 
                layer._b = np.zeros((layer._W_size[0], 1))
    
    def forward_propagation(self) -> None:
        for i in range(1, len(self._layers)): 
            self._layers[i]._h = self._layers[i]._W @ self._layers[i-1]._a + self._layers[i]._b
            
            # testing
            # if np.isnan(self._layers[i]._h).any():
            #     print(f"NaN detected in h at Layer {i}")

            self._layers[i]._a = self._layers[i]._activation.value(self._layers[i]._h)

            # testing
            # if np.isnan(self._layers[i]._a).any():
            #     print(f"NaN detected in activations at Layer {i}")

            self._layers[i]._h_val = self._layers[i]._W @ self._layers[i-1]._a_val + self._layers[i]._b
            self._layers[i]._a_val = self._layers[i]._activation.value(self._layers[i]._h_val)
        
        if(self._loss_type == 'CategoricalCrossEntropy'):
            self._layers[-1]._y = Softmax().value(self._layers[-1]._a)
            self._layers[-1]._y_val = Softmax().value(self._layers[-1]._a_val)
        else: 
            self._layers[-1]._y = self._layers[-1]._a
            self._layers[-1]._y_val = self._layers[-1]._a_val

    def backward_propagation(self) -> None: 
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
        
            training_accuracy, validation_accuracy = self._get_accuracy()
            train_accuracy_log.append(training_accuracy)
            val_accuracy_log.append(validation_accuracy)

            if(self._log): 
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

                if(flag): 
                    break

                # Compute gradient for output layer
                self._layers[-1]._a_grad = self._loss_function.derivative(target_batch, y_batch)
                self._layers[-1]._h_grad = self._layers[-1]._a_grad * self._layers[-1]._activation._derivative(self._layers[-1]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size])

                self._layers[-1]._W_grad = self._layers[-1]._h_grad @ self._layers[-2]._a[:, batch * self._batch_size: (batch + 1) * self._batch_size].T
                self._layers[-1]._W_update = self._layers[-1]._W_optimiser.update(self._layers[-1]._W_grad)

                self._layers[-1]._b_grad = (1) * np.sum(self._layers[-1]._h_grad, axis=1).reshape(-1, 1)
                self._layers[-1]._b_update = self._layers[-1]._b_optimiser.update(self._layers[-1]._b_grad)

                assert self._layers[-1]._W_update.shape == self._layers[-1]._W.shape, 'Size Mismatch'

                # Backpropagate through hidden layers
                for i in range(len(self._layers) - 2, 0, -1):  
                    self._layers[i]._a_grad = self._layers[i+1]._W.T @ self._layers[i+1]._h_grad
                    self._layers[i]._h_grad = self._layers[i]._a_grad * self._layers[i]._activation._derivative(
                        self._layers[i]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size]
                    )

                    self._layers[i]._b_grad = (1) * np.sum(self._layers[i]._h_grad, axis=1).reshape(-1, 1)
                    self._layers[i]._W_grad = self._layers[i]._h_grad @ self._layers[i-1]._a[:, batch * self._batch_size: (batch + 1) * self._batch_size].T
                    
                    self._layers[i]._W_update = self._layers[i]._W_optimiser.update(self._layers[i]._W_grad)
                    self._layers[i]._b_update = self._layers[i]._b_optimiser.update(self._layers[i]._b_grad)

                # Update weights and biases
                for layer in self._layers[1:]:  
                    layer._W -= layer._W_update
                    layer._b -= layer._b_update

                self.forward_propagation()

            if(flag): 
                break

    def _get_accuracy(self):    # Returns tuple when validation, else returns a single float 
        train_t = encoder.inverse_transform(self._target)
        train_y = encoder.inverse_transform(self._layers[-1]._y)
        training_accuracy = metrics.accuracy_score(train_t, train_y)

        if(self._validation): 
            val_t = encoder.inverse_transform(self._val_target)
            val_y = encoder.inverse_transform(self._layers[-1]._y_val)
            validation_accuracy = metrics.accuracy_score(val_t, val_y)
            if(self._verbose):
                print(f'Validation accuracy:\t{validation_accuracy}')
            return training_accuracy, validation_accuracy
        
        if(self._verbose):
            print(f'Training accuracy:\t{training_accuracy}')

        return training_accuracy
    
    def predict(self, test_X: np.ndarray) -> np.ndarray: 
        a = test_X
        for i in range(1, len(self._layers)): 
            h = self._layers[i]._W @ a - self._layers[i]._b
            a = self._layers[i]._activation.value(h)

        if(self._loss_type == 'CategoricalCrossEntropy'): 
            pred_y = Softmax().value(a)
        else: 
            pred_y = a
        
        return np.argmax(pred_y, axis = 0)
        # return pred_y
