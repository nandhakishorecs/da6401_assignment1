import numpy as np 
import math 

# ------------------- Different optimising algorithms for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Vannila gradient descent (only step size update using gradients)
class VannilaGradientDescent: 
    __slots__ = '_lr', '_update'
    def __init__(self, eta: float = 0.01) -> None:
        self._lr = eta
        self._update = 0 

    def _set_parameters(self, parameters): 
        for i in parameters: 
            setattr(self, i, parameters[i])

    def _update_parameters(self, gradient: np.ndarray) -> np.ndarray: 
        self._update = (self._lr * gradient )
        return self._update

# Momentum gradient descent - introducing history 
class MomentumGradientDescent: 
    __slots__ = '_lr', '_m' , '_update'
    def __init__(self, eta: float = 0.001, beta: float = 0.9) -> None:
        self._lr = eta
        self._m = beta
        self._update = 0 

    def _set_parameters(self, parameters): 
        for i in parameters: 
            setattr(self, i, parameters[i])

    def _update_parameters(self, gradient: np.ndarray) -> np.ndarray: 
        self._update = (self._m * self._update) + (self._lr * gradient)
        return self._update