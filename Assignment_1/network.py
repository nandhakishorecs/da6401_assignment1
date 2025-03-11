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

# ------------------- A Complete Neural Network ------------------------------------------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
#
#   This file contains the code for implementing a artificial neural netwrok from scratch with numpy and math libraries in Python
#   This file uses activation functions, optimisers, loss functions, layers and initialisers from other Python files. 
#   
#   Description of the function in the class NeuralNetwork: 
#       - _init_parameters (self): 
#           * initialises the weights and biases of a neural network with the given initialisation 
#
#       - forward_propagation (self): 
#           * does a complete forward sweep and gives out the logits for the classes 
#           * takes the functions from actiavtions, optimisers, layeres and initialisers
#       
#       - backward_propagation (self): 
#           * computes gradient and updates the parameters using the optimiser function 
#           * takes the functions from actiavtions, optimisers, layeres and initialisers
#
#       - _get_accuracy(true, predicted): 
#           * private function, to calculate accuracy for validation and training - used for logging 
#
#       - _check(data):
#           * private function, to check for loss values for debugging 
#
#       - __repr__
#           * python magic function to describe the neural network, print the class to use it. 
#
#
# ----------------------------------------------------------------------------------------------------------------------------------

# optimiser map 
map_optimiser = {
    'sgd' : GradientDescent,
    'momentum' : MomentumGD, 
    'nag': NesterovMomentumGD, 
    'adagrad': AdaGrad, 
    'rmsprop': RMSProp,
    'adadelta': AdaDelta,
    'adam': Adam, 
    'nadam': Nadam, 
    'eve': Eve
}

# loss function map 
map_loss_function = {
    'cross_entropy' : CategoricalCrossEntropy(), 
    'mean_squared_error' : MeanSquaredError()
}

# inictialiser map 
map_initialiser = {
    'random' : RandomInit(), 
    'xavier': XavierInit(), 
    'he': HeInit()
}

# Encoder 
encoder = OneHotEncoder()

# Classification metrics calculator 
metrics = Metrics()

# Basic skeleton 
class NeuralNetwork: 
    __slots__ = '_layers', '_batch_size', '_initialisation', '_n_epochs', '_optimiser', '_target', '_n_batch', '_loss_type', '_loss_function', '_log', '_verbose', '_validation', '_optimised_parameters', '_weight_decay', '_lr', '_validation', '_val_X', '_val_target', '_target_batch', '_y_batch'
    def __init__(self, layers: list, batch_size: int, optimiser: str, n_epochs: int, target: np.ndarray, loss_function: str, initialisation: str, learning_rate: float, validation:bool, val_X: np.ndarray = None, val_target: np.ndarray = None, wandb_log: bool = False, verbose: bool = False, weight_decay: float = 0, optimised_parameters = None) -> None: 
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
        self._optimised_parameters = optimised_parameters
        self._weight_decay = weight_decay
        self._lr = learning_rate
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
            
            layer._W_optimiser = deepcopy(map_optimiser[self._optimiser](self._lr))
            layer._b_optimiser = deepcopy(map_optimiser[self._optimiser](self._lr))

        if(self._initialisation == 'random'):
            for layer in self._layers[1: ]: 
                layer._W = RandomInit().initialize(layer_size = layer._W_size) 
                # layer._W = np.random.normal(loc = 0, scale = 1, size = layer._W_size)
                layer._b = np.zeros((layer._W_size[0], 1))

        elif(self._initialisation == 'xavier'): 
            for layer in self._layers[1: ]: 
                layer._W = XavierInit().initialize(layer_size = layer._W_size) 
                layer._b = np.zeros((layer._W_size[0], 1))

        elif(self._initialisation == 'he'): 
            for layer in self._layers[1: ]: 
                layer._W = HeInit().initialize(layer_size = layer._W_size) 
                layer._b = np.zeros((layer._W_size[0], 1))

    # def __repr__(self) -> str: 
    #     return f'''
    #     Neural Network:
    #     Number of layers:\t {len(self._layers)},
    #     Layers: {self._layers},
    #     Optimiser:\t\t\t{map_optimiser[self._optimiser].__name__},
    #     Initialisation:\t\t{map_initialiser[self._initialisation]},
    #     Epochs:\t\t\t{self._n_epochs},
    #     Batch Size:\t\t\t{self._batch_size},
    #     Loss Function:\t\t{self._loss_function},
    #     Learning Rate:\t\t{self._lr},
    #     Weight Decay:\t\t{self._weight_decay},
    #     Optimised Parameters:\t{self._optimised_parameters}
    #     '''
    def __repr__(self) -> str:
        return f'''Neural Network:
        Number of layers:     {len(self._layers)}
        Layers:               {self._layers}
        Optimiser:            {map_optimiser[self._optimiser].__name__}
        Initialisation:       {map_initialiser[self._initialisation]}
        Epochs:               {self._n_epochs}
        Batch Size:           {self._batch_size}
        Loss Function:        {self._loss_function}
        Learning Rate:        {self._lr}
        Weight Decay:         {self._weight_decay}
        Optimised Parameters: {self._optimised_parameters}'''

    
    def forward_propagation(self) -> None:
        for i in range(1, len(self._layers)): 
            # # testing
            # if np.isnan(self._layers[i]._h).any():
            #     print(f"NaN detected in h at Layer {i}")

            # # testing
            # if np.isnan(self._layers[i]._a).any():
            #     print(f"NaN detected in activations at Layer {i}")
            self._layers[i]._h = self._layers[i]._W @ self._layers[i-1]._a - self._layers[i]._b
            self._layers[i]._a = self._layers[i]._activation.value(self._layers[i]._h)
            self._layers[i]._h_val = self._layers[i]._W @ self._layers[i-1]._a_val - self._layers[i]._b
            self._layers[i]._a_val = self._layers[i]._activation.value(self._layers[i]._h_val)
        
        if(self._loss_type == 'cross_entropy'):
            self._layers[-1]._y = Softmax().value(self._layers[-1]._a)
            self._layers[-1]._y_val = Softmax().value(self._layers[-1]._a_val)
        else: 
            self._layers[-1]._y = self._layers[-1]._a
            self._layers[-1]._y_val = self._layers[-1]._a_val

    def backward_propagation(self) -> None: 
        # Logging locally 
        lr_log = [] 
        train_loss_log = []
        train_accuracy_log = []

        val_loss_log = []
        val_accuracy_log = []

        # self._loss_function = MeanSquaredError()
        flag = False

        for epoch in tqdm(range(self._n_epochs)):
            # Logging locally
            lr_log.append(self._layers[-1]._W_optimiser._lr)
            
            train_loss_log.append(self._loss_function.value(self._target, self._layers[-1]._y))
            val_loss_log.append(self._loss_function.value(self._val_target, self._layers[-1]._y_val))
        
            training_accuracy, validation_accuracy = self._get_accuracy()
            train_accuracy_log.append(training_accuracy)
            val_accuracy_log.append(validation_accuracy)

            if(self._log): 
                # Wandb logging 
                wandb.log({
                    'Step': epoch, 
                    'Training_loss': train_loss_log[-1] / self._target.shape[1], 
                    'Training_Accuracy': train_accuracy_log[-1] / self._target.shape[1], 
                    'Validation_Loss': val_loss_log[-1] / self._val_target.shape[1], 
                    'Validation_Accuracy': val_accuracy_log[-1] / self._val_target.shape[1]
                })

            # Start of Backpropagation
            for batch in range(self._n_batch):     
                target_batch = self._target[:, batch * self._batch_size: (batch + 1) * self._batch_size]
                y_batch = self._layers[-1]._y[:, batch * self._batch_size: (batch + 1) * self._batch_size]
                
                self._target_batch = target_batch
                self._y_batch = y_batch

                if(flag): 
                    break

                # Compute gradient for output layer
                self._layers[-1]._a_grad = self._loss_function.derivative(self._target_batch, self._y_batch) 
                self._layers[-1]._h_grad = self._layers[-1]._a_grad * self._layers[-1]._activation._derivative(self._layers[-1]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size])

                self._layers[-1]._W_grad = self._layers[-1]._h_grad @ self._layers[-2]._a[:, batch * self._batch_size: (batch + 1) * self._batch_size].T 
                self._layers[-1]._W_update = self._layers[-1]._W_optimiser.update(self._layers[-1]._W_grad)

                self._layers[-1]._b_grad = (-1) * np.sum(self._layers[-1]._h_grad, axis=1).reshape(-1, 1)
                self._layers[-1]._b_update = self._layers[-1]._b_optimiser.update(self._layers[-1]._b_grad)

                assert self._layers[-1]._W_update.shape == self._layers[-1]._W.shape, 'Size Mismatch, previous weights and new weights should have same sizes'

                # Backpropagate through hidden layers
                for i in range(len(self._layers) - 2, 0, -1):  
                    self._layers[i]._a_grad = self._layers[i+1]._W.T @ self._layers[i+1]._h_grad
                    self._layers[i]._h_grad = self._layers[i]._a_grad * self._layers[i]._activation._derivative(self._layers[i]._h[:, batch * self._batch_size: (batch + 1) * self._batch_size]) 

                    self._layers[i]._b_grad = (-1) * np.sum(self._layers[i]._h_grad, axis=1).reshape(-1, 1)
                    self._layers[i]._W_grad = self._layers[i]._h_grad @ self._layers[i-1]._a[:, batch * self._batch_size: (batch + 1) * self._batch_size].T

                    self._layers[i]._W_update = self._layers[i]._W_optimiser.update(self._layers[i]._W_grad)
                    self._layers[i]._b_update = self._layers[i]._b_optimiser.update(self._layers[i]._b_grad)

                # Update weights and biases
                for _ , layer in enumerate(self._layers[1:]):
                    layer._W -= layer._W_update + (self._weight_decay * layer._W)
                    layer._b -= layer._b_update
                    
                self.forward_propagation()

            # print('w update',self._layers[-1]._W_update)
            # print('w grad',self._layers[-1]._W_grad)
            # print('b update',self._layers[-1]._b_update)
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
                print(f'Training accuracy: {training_accuracy}\tValidation accuracy: {validation_accuracy}')
            return training_accuracy, validation_accuracy
        
        if(self._verbose):
            print(f'Training accuracy:\t{training_accuracy}')

        return training_accuracy
    
    # used for debugging
    def _check(self, test_X: np.ndarray, test_t: np.ndarray) -> tuple: 
        self._layers[0]._a_test = test_X
        for i in range(1, len(self._layers)):
            self._layers[i]._h_test = self._layers[i]._W @ self._layers[i-1]._a_test - self._layers[i]._b
            self._layers[i]._a_test = self._layers[i]._activation.value(self._layers[i]._h_test)
        
        if(self._loss_type == 'cross_entropy'):
            self._layers[-1]._y_test = Softmax().value(self._layers[-1]._a_test)
        else: 
            self._layers[-1]._y_test = self._layers[-1]._a_test

        test_loss = self._loss_function.value(test_t, self._layers[-1]._a_test)

        encoder = OneHotEncoder()
        y_tmp = encoder.inverse_transform(self._layers[-1]._y_test)
        t_tmp = encoder.inverse_transform(test_t)
        loss_accuracy = metrics.accuracy_score(y_tmp, t_tmp)

        return test_loss, loss_accuracy
    
    # to do inference
    def predict(self, test_X: np.ndarray) -> np.ndarray: 
        a = test_X
        for i in range(1, len(self._layers)): 
            h = self._layers[i]._W @ a - self._layers[i]._b
            a = self._layers[i]._activation.value(h)

        if(self._loss_type == 'cross_entropy'): 
            pred_y = Softmax().value(a)
        else: 
            pred_y = a
        
        # y = encoder.inverse_transform(pred_y)
        return np.argmax(pred_y, axis = 0)