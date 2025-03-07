import numpy as np 

# ------------------- Different optimising algorithms for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Vannila gradient descent (only step size update using gradients)
class VannilaGradientDescent:
    __slots__ = ['_lr']
    
    def __init__(self, lr=0.01) -> None:
        self._lr = lr
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self._lr * grads
    
    def __repr__(self) -> str:
        return "GradientDescent"

# Momentum gradient descent - introducing history 
class MomentumGD:
    __slots__ = ['_lr', '_momentum', '_velocity']
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self._lr = lr
        self._momentum = momentum
        self._velocity = 0
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self._velocity = self._momentum * self._velocity - self._lr * grads
        return params + self._velocity
    
    def __repr__(self) -> str:
        return "MomentumGD"
    
# class StochasticGD:
#     __slots__ = ['_lr']
    
#     def __init__(self, lr: float = 0.01) -> None:
#         self._lr = lr
    
#     def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
#         return params - self.lr * grads
    
#     def __repr__(self) -> str:
#         return "StochasticGD"
class NesterovMomentumGD(MomentumGD):
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        lookahead_params = params + self._momentum * self._velocity
        self._velocity = self._momentum * self._velocity - self._lr * grads
        return lookahead_params + self._velocity
    
    def __repr__(self) -> str:
        return "NesterovMomentumGD"

class AdaGrad:
    __slots__ = ['_lr', '_epsilon', '_G']
    
    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        self._lr = lr
        self._epsilon = epsilon
        self._G = 0
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.G += (grads ** 2)
        return params - self._lr * grads / (np.sqrt(self._G) + self._epsilon)
    
    def __repr__(self) -> str:
        return "AdaGrad"

class RMSProp(AdaGrad):
    __slots__ = ['_decay_rate']
    def __init__(self, lr: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__(lr, epsilon)
        self._decay_rate = decay_rate
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self._G = self._decay_rate * self._G + (1 - self._decay_rate) * grads ** 2
        return params - self._lr * grads / (np.sqrt(self._G) + self._epsilon)
    
    def __repr__(self) -> str:
        return "RMSProp"
    
class AdaDelta:
    __slots__ = ['_rho', '_epsilon', '_Eg', '_Edelta']
    
    def __init__(self, rho: float = 0.95, epsilon: float = 1e-6) -> None:
        self._rho = rho
        self._epsilon = epsilon
        self._Eg = 0
        self._Edelta = 0
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self._Eg = self._rho * self._Eg + (1 - self._rho) * grads ** 2
        delta = (-1) * np.sqrt(self._Edelta + self._epsilon) / np.sqrt(self._Eg + self._epsilon) * grads
        self._Edelta = self._rho * self._Edelta + (1 - self._rho) * delta ** 2
        return params + delta
    
    def __repr__(self):
        return "AdaDelta"
class Adam:
    __slots__ = ['_lr', '_beta1', '_beta2', '_epsilon', '_m', '_v', '_t']
    
    def __init__(self, lr: float = 0.001, beta1:float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._m = 0
        self._v = 0
        self._t = 0
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self._t += 1
        # Momentum update
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        # Learning Rate decay 
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grads ** 2)
        # Bias Correction 
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        return params - self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
    
    def __repr__(self) -> str:
        return "Adam"