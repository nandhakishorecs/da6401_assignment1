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
    'Momentum_GD' : MomentumGradientDescent() 
}

# loss function map 
map_loss_function = {
    'Categorical_Cross_Entropy' : CategoricalCrossEntropy(), 
    'Mean_Squared_Error' : MeanSquaredError()
}

# Encoder 
encoder = LabelEncoder()

# Basic skeleton - yet to be implemented
class NeuralNetwork: 
    # ---------------- Class initialisation ----------------
    # __slots__ = '_layers', '_batch_size', '_optimiser', '_target', '_init', '_n_epochs', '_validation', '_X_val', '_y_val', '_log', '_loss_type', '_loss_function', '_a', '_optimised_parameters'
    def __init__(
            self, layers: list, batch_size: int, optimiser: str, initilaisation: str, n_epochs: int, 
            target: np.ndarray, 
            loss_function: str, 
            wandb: bool = False, 
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

        self._validation = validation
        if(self._validation == validation): 
            self._X_val = validation_features
            self._y_val = validation_target
            self._layers[0]._a = validation_features

        self._X_val = validation_features
        self._loss_function = map_loss_function[loss_function]
        self._loss_type = loss_function
        self._n_epochs = n_epochs
        self._optimised_parameters = optimised_parameters
        self._log = wandb
        self._init_parameters()

    # ---------------- Class representation ----------------
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

    # ---------------- Parameter initialisation for different layers ----------------
    def _init_parameters(self):
        previous_layer_size = self._layers[0]._size
        for layer in self._layers[1: ]:
            layer._W_size = (layer._size, previous_layer_size)
            previous_layer_size = layer._size
            layer._W_optimiser = deepcopy(self._optimiser)
            layer._b_optimiser = deepcopy(self._optimiser)
            if(self._optimised_parameters is not None): 
                layer._W_optimiser._set_parameters(self._optimised_parameters)
                layer._b_optimiser._set_parameters(self._optimised_parameters)
        
        if(self._init == 'Random_normal'):
            for layer in self._layers[1: ]: 
                layer._W = np.random.normal(loc = 0, scale = 1.0, size = layer._W_size)
                layer._b = np.zeros((layer._W_size[0], 1))
        elif(self._init == 'Xavier'): 
            for layer in self._layers[1: ]: 
                initialiser = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05)
                layer._W = np.array(initialiser(shape = layer._W_size))
                layer._b = np.array((layer._W_size[0], 1))
        elif(self._init == 'Zero'): 
            for layer in self._layers[1: ]: 
                layer._W = np.ones(layer._W_size) * 0.05
                layer._b = np.zeros((layer._W_size[0], 1))
    
    # ---------------- Forward Propagation ----------------
    def _forward_propagation(self) -> None: 
        for i in range(1, len(self._layers)): 
            # in first layer, no activation, in hidden layers, pre-activation
            self._layers[i]._h = self._layers[i]._W @ self._layers[i-1]._a - self._layers[i]._b
            # applying activation function (relu, tanh, sigmoid) 
            self._layers[i]._a = self._layers[i]._activation.value(self._layers[i]._h)
            # validation 
            self._layers[i]._h_val = self._layers[i]._W @ self._layers[i-1]._activation_val - self._layers[i]._b
            self._layers[i]._activation_val = self._layers[i]._activation.value(self._layers[i]._h_val)
        
        if(self._loss_type == 'Categorical_Cross_Entropy'): 
            # Final layer has softmax, others have a different activation function
            self._layers[-1]._y = softmax().value(self._layers[-1]._a)
            self._layers[-1]._y_val = softmax().value(self._layers[-1]._activation_val)
        else: 
            # getting values from the activation layer of the previous layer as the input to the current layer
            self._layers[-1]._y = self._layers[-1]._a
            self._layers[-1]._y_val = self._layers[-1]._activation_val

    # ---------------- Backward Propagation ----------------
    def _backward_propagation(self, validation: bool = False, verbose: bool = False) -> None: 
        # Flag variable to stop when converged
        flag = False
        
        # For Wandb
        lr_log = []
        training_loss_log = []
        training_accuracy_log = []
        validation_loss_log = []
        validation_accuracy_log = [] 

        # needed debugging in verbose, wandb
        for epoch in tqdm(range(self._n_epochs)):
            # accuracy calculation - START
            # train_target = encoder.inverse_transform(self._target)
            train_target = self._target
            # train_y = encoder.inverse_transform(self._layers[-1]._y)
            train_y = self._layers[-1]._y
            training_accuracy = Metrics().accuracy_score(train_target, train_y)
            if(verbose is True):
                print(f'Training accuracy: {training_accuracy}')
            if(validation is True): 
                # validation_target = encoder.inverse_transform(self._y_val)
                validation_target = self._y_val
                # validation_y = encoder.inverse_transform(self._layers[-1]._y_val)
                validation_y = self._layers[-1]._y_val
                validation_accuracy = Metrics().accuracy_score(validation_target, validation_y)
                if(verbose is True):
                    print(f'Validation accuracy: {validation_accuracy}')
            # accuracy calculation - END

            # Logging - START
            lr_log.append(self._layers[-1]._W_optimiser._eta)
            training_loss_log.append(self._loss_function._loss(self._target, self._layers[-1]._y))
            validation_loss_log.append(self._loss_function._loss(self._y_val, self._X_val))
            training_loss_log.append(training_accuracy)
            training_accuracy_log.append(training_accuracy)
            validation_accuracy_log.append(validation_accuracy)
            # Logging - END

            # Wandb logging
            if(self._log is True): 
                wandb.log({
                    'Epoch:': epoch, \
                    'Training Loss:': training_loss_log[-1]/self._target.shape[1], \
                    'Validation Loss': validation_loss_log[-1]/self._y_val.shape[1], \
                    'Training Accuracy:': training_accuracy[-1]/self._target.shape[1], \
                    'Validation Accuracy:': validation_accuracy_log[-1]/self._y_val.shape[1]
                })

            for batch in range(self._batch_size): 
                target_batch = self._target[:, batch  * self._batch_size:(batch + 1) * self._batch_size]
                y_batch = self._layers[-1]._y[:, batch * self._batch_size:(batch + 1) * self._batch_size]

                try:
                    if(training_loss_log[-1] > training_loss_log[-2]):
                        for layer in self._layers[1:]: 
                            layer._W_optimiser._set_parameters({'eta': self._optimiser._lr/2})
                            layer._b_optimiser._set_parameters({'eta': self._optimiser._lr/2})
                            flag = True
                except: 
                    pass

                if(flag):
                    break

                self._layers[-1]._a_grad = self._loss_function._derivative(target_batch, y_batch)
                self._layers[-1]._h_grad = self._layers[-1]._a_grad * self._layers[-1]._activation._derivative(self._layers[-1]._h[: batch * self._batch_size:(batch+1) * self._batch_size])

                self._layers[-1]._W_grad = self._layers[-1]._h_grad @ self._layers[-2]._a[:, batch * self._batch_size : (batch + 1) * self._batch_size].T
                self._layers[-1]._W_update = self._layers[-1]._W_optimiser._update_parameters(self._layers[-1]._W_grad)

                self._layers[-1]._b_grad = (-1)*np.sum(self._layers[-1]._h_grad, axis = 1).reshape(-1, 1)
                self._layers[-1]._b_update = self._layers[-1]._b_optimiser._update_parameters(self._layers[-1]._b_grad)

                assert self._layers[-1]._W_update.shape == self._layers[-1]._W.shape, 'Size mismatch in hidden layers'

                # do backpropagation in remianing layers
                for i in range(len(self._layers[: -2]), 0, -1): 
                    self._layers[i]._a_grad = self._layers[i+1]._W.T @ self._layers[i+1]._h_grad
                    self._layers[i]._h_grad = self._layers[i]._a_grad * self._layers[i]._activation._derivative(self._layers[i]._h[:, batch * self._batch_size: (batch+1)* self._batch_size])
                    
                    self._layers[i]._b_grad = (-1) * np.sum(self._layers[i]._h_grad, axis = 1).reshape(-1, 1)
                    self._layers[i]._W_grad = self._layers[i]._h_grad @ self._layers[i-1]._a[:, batch * self._batch_size: (batch + 1) * self._batch_size].T

                    self._layers[i]._W_update = self._layers[i]._W_optimiser._update_parameters(self._layers[i]._W_grad)
                    self._layers[i]._b_update = self._layers[i]._b_optimiser._update_parameters(self._layers[i]._b_grad)
                    pass
                

                # update weight
                for _, layer in enumerate(self._layers[1:]):
                    layer._W = layer._W - layer._W_update
                    layer._b = layer._b - layer._b_update

                self._forward_propagation()

            if(flag): 
                break