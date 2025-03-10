import numpy as np 

# ------------------- Different optimising algorithms for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Vannila gradient descent (only step size update using gradients)
import numpy as np

class GradientDescent:
    def __init__(self, lr: float = 0.01) -> None:
        self._lr = lr
        self._update = 0
    
    def set_parameters(self, params):
        for key in params: 
            setattr(self, key, params[key])
    
    def update(self, grad) -> np.ndarray:
        self._update = self._lr * grad
        return self._update

class MomentumGD(GradientDescent):
    def __init__(self, lr: float = 0.001, momentum: float = 0.9) -> None:
        super().__init__(lr)
        self._momentum = momentum                     
    
    def update(self, grads: np.ndarray) -> np.ndarray:
        self._update = (self._momentum * self._update) + (self._lr * grads)
        # self._params += self._velocity
        return self._update

class NesterovMomentumGD(GradientDescent):
    def __init__(self, lr: float = 0.001, momentum: float = 0.9) -> None:
        super().__init__(lr)
        self._momentum = momentum

    def update(self, grads: np.ndarray) -> np.ndarray:
        lookahead = self._momentum * self._update                   # Apply momentum first (lookahead step)
        grads_at_lookahead = grads - (self._momentum * grads)       # Approximate Nesterov update
        self._update = lookahead + (self._lr * grads_at_lookahead)
        return self._update

class AdaGrad:
    def __init__(self, lr: float = 1e-2, epsilon: float = 1e-7) -> None:
        self._lr = lr
        self._epsilon = epsilon
        self._G = 0
            
    def set_parameters(self, params: np.ndarray) -> None:
        for key in params: 
            setattr(self, key, params[key])
    
    def update(self, grads: np.ndarray) -> np.ndarray:
        self._G += (grads ** 2)
        effective_lr = self._lr/((self._G + self._epsilon) ** (1/2))
        return effective_lr * grads

class RMSProp(AdaGrad):
    def __init__(self, lr: float = 1e-2, decay_rate: float = 0.9, epsilon: float = 1e-7) -> None:
        super().__init__(lr, epsilon)
        self._decay_rate = decay_rate
    
    def update(self, grads: np.ndarray) -> np.ndarray:
        self._G = (self._decay_rate * self._G) + (1 - self._decay_rate) * (grads ** 2)
        effective_lr = ((self._lr) / (self._G + self._epsilon)**(1/2))
        return effective_lr * grads

# The implementation is working, but not used in sweep
class AdaDelta:
    def __init__(self, decay_rate: float = 0.95, epsilon: float = 1e-7) -> None:
        # decay rate is noted as lr for ease of coding
        self._lr = decay_rate
        self._epsilon = epsilon
        self._G = 0
        self._delta = 0

    def set_parameters(self, params: np.ndarray) -> None:
        for key in params: 
            setattr(self, key, params[key])
    
    def update(self, grads: np.ndarray) -> np.ndarray:
        self._G = (self._G * self._lr) + ((1 - self._lr) * (grads ** 2))
        # update_step = (-1)*(np.sqrt(self._delta) + self._epsilon) / (np.sqrt(self._G) + self._epsilon) * grads
        update_step = (-1) * ((np.sqrt(self._delta) + self._epsilon) / (np.sqrt(self._G) + self._epsilon)) * grads
        self._delta = (self._lr * self._delta) + ((1 - self._lr) * (update_step ** 2))
        return update_step

class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> None:
        self._lr = lr
        self._beta1 = beta1     # Decay rate for first moment (momentum)
        self._beta2 = beta2     # Decay rate for second moment (RMSProp-like)
        self._epsilon = epsilon
        self._m = 0             # First moment (mean of gradients)
        self._v = 0             # Second moment (uncentered variance of gradients)
        self._t = 1             # Time step (for bias correction)
    
    def set_parameters(self, params: np.ndarray) -> None:
        for key in params:
            setattr(self, key, params[key])

    def update(self, grads: np.ndarray) -> np.ndarray:
        self._t += 1  # Increment time step
        
        self._m = (self._beta1 * self._m) + ((1 - self._beta1) * grads)
        self._v = (self._beta2 * self._v) + ((1 - self._beta2) * (grads ** 2))
        
        # Bias Correction 
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        
        # Compute adaptive update
        update_val = (self._lr / (v_hat + self._epsilon) ** (1/2)) * m_hat

        return update_val

class Nadam(Adam):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> None:
        super().__init__(lr, beta1, beta2, epsilon)

    def update(self, grads: np.ndarray) -> np.ndarray:
        self._t += 1  # Increment time step
        
        # Compute biased first moment estimate (momentum)
        self._m = (self._beta1 * self._m) + ((1 - self._beta1) * grads)
        self._v = (self._beta2 * self._v) + ((1 - self._beta2) * (grads ** 2))
        
        # Bias correction
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        
        # Nesterov momentum correction
        m_nesterov = (self._beta1 * m_hat) + ((1 - self._beta1) * grads)  

        # Compute adaptive update
        update_val = (self._lr / (np.sqrt(v_hat) + self._epsilon)) * m_nesterov

        return update_val

class Eve(Adam):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, beta3: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(lr, beta1, beta2, epsilon)
        self._beta3 = beta3  # Decay rate for smoothing relative loss changes
        self._f_prev = None  # Stores previous loss for adaptive learning rate

    def set_loss(self, loss: float) -> None:
        if self._f_prev is None:
            self._f_prev = loss  # Initialize only once
        else:
            self._adjust_learning_rate(loss)

    def _adjust_learning_rate(self, loss: float) -> None:
        d = abs(loss - self._f_prev) / (min(loss, self._f_prev) + self._epsilon)  # Relative loss change
        d_hat = self._beta3 * d + (1 - self._beta3) * (self._f_prev / (loss + self._epsilon))  # Smoothed factor
        self._lr *= (1 + np.tanh(d_hat - 1))  # Update learning rate adaptively
        self._f_prev = loss  # Store loss for next step

    def update(self, grads: np.ndarray) -> np.ndarray:
        self._t += 1  # Increment time step

        # Compute biased first and second moment estimates
        self._m = (self._beta1 * self._m) + ((1 - self._beta1) * grads)
        self._v = (self._beta2 * self._v) + ((1 - self._beta2) * (grads ** 2))

        # Bias correction
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)

        # Compute adaptive update
        update_val = (self._lr / (np.sqrt(v_hat) + self._epsilon)) * m_hat

        return update_val

# COMPLETED