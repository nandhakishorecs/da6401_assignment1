import numpy as np          # To handle vector / matrix operations 

# ------------------- Activation Functions for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# used for any hidden layer in a neural network 
class Sigmoid:
    __slots__ = ['x']
    
    def __init__(self, x):
        self.x = np.array(x)
    
    def value(self):
        return 1 / (1 + np.exp(-self.x))
    
    def _derivative(self):
        sigmoid_x = self.value()
        return sigmoid_x * (1 - sigmoid_x)
    
    def __repr__(self):
        return "Sigmoid"

# used for any hidden layer in a neural network 
class Tanh:
    __slots__ = ['x']
    
    def __init__(self, x):
        self.x = np.array(x)
    
    def value(self):
        return np.tanh(self.x)
    
    def _derivative(self):
        return 1 - np.tanh(self.x) ** 2
    
    def __repr__(self):
        return "Tanh"
    
# used for any hidden layer in a neural network 
class ReLU:
    __slots__ = ['x']
    
    def __init__(self, x):
        self.x = np.array(x)
    
    def value(self):
        return np.maximum(0, self.x)
    
    def _derivative(self):
        return np.where(self.x > 0, 1, 0)
    
    def __repr__(self):
        return "ReLU"

# used for the last layer in a neural network 
class Softmax:
    __slots__ = ['x']
    
    def __init__(self, x):
        self.x = np.array(x)
    
    def value(self):
        exp_x = np.exp(self.x - np.max(self.x))  # Prevent overflow
        return exp_x / np.sum(exp_x)
    
    def _derivative(self):
        softmax_x = self.value().reshape(-1, 1)
        return np.diagflat(softmax_x) - np.dot(softmax_x, softmax_x.T)
    
    def __repr__(self):
        return "Softmax"