import numpy as np 

# ------------------- Different optimising algorithms for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Vannila gradient descent (only step size update using gradients)
import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01):
        self._lr = lr
        self._update = 0
    
    def set_parameters(self, params):
        for key in params: 
            setattr(self, key, params[key])
    
    def update(self, grad):
        self._update = self._lr * grad
        return self._update

class MomentumGD:
    def __init__(self, lr=0.001, momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        self._velocity = 0
        self._params = 0 
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._velocity = self._momentum * self._velocity - self._lr * grads
        self._params += self._velocity
        return self._params

class NesterovMomentumGD(MomentumGD):
    def update(self, grads):
        lookahead_params = self._params + self._momentum * self._velocity
        self._velocity = self._momentum * self._velocity - self._lr * grads
        self._params = lookahead_params + self._velocity
        return self._params

class AdaGrad:
    def __init__(self, lr=0.01, epsilon=1e-8):
        self._lr = lr
        self._epsilon = epsilon
        self._G = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._G += grads ** 2
        self._params -= self._lr * grads / (np.sqrt(self._G) + self._epsilon)
        return self._params

class RMSProp:
    def __init__(self, lr=0.001, decay_rate=0.9, epsilon=1e-8):
        self._lr = lr
        self._decay_rate = decay_rate
        self._epsilon = epsilon
        self._G = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._G = self._decay_rate * self._G + (1 - self._decay_rate) * grads ** 2
        self._params -= self._lr * grads / (np.sqrt(self._G) + self._epsilon)
        return self._params

class AdaDelta:
    def __init__(self, decay_rate=0.95, epsilon=1e-6):
        self._lr = decay_rate
        self._epsilon = epsilon
        self._G = 0
        self._delta = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._G = self._lr * self._G + (1 - self._lr) * grads ** 2
        update_step = - (np.sqrt(self._delta + self._epsilon) / np.sqrt(self._G + self._epsilon)) * grads
        self._delta = self._lr * self._delta + (1 - self._lr) * update_step ** 2
        self._params += update_step
        return self._params

class Adam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._m = 0
        self._v = 0
        self._t = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grads ** 2)
        # Bias correction 
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        self._params -= self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
        return self._params
import numpy as np

class Nadam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._m = 0
        self._v = 0
        self._t = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grads ** 2)
        # Bias correction 
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        # Nadam-specific Nesterov term
        nesterov_m = self._beta1 * m_hat + (1 - self._beta1) * grads / (1 - self._beta1 ** self._t)
        self._params -= self._lr * nesterov_m / (np.sqrt(v_hat) + self._epsilon)
        return self._params
    
import numpy as np

class Nadam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._m = 0
        self._v = 0
        self._t = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grads ** 2)
        # Bias correction 
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        # Nadam-specific Nesterov term
        nesterov_m = self._beta1 * m_hat + (1 - self._beta1) * grads / (1 - self._beta1 ** self._t)
        self._params -= self._lr * nesterov_m / (np.sqrt(v_hat) + self._epsilon)
        return self._params

class Eve:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, beta3=0.999, epsilon=1e-8):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta3 = beta3
        self._epsilon = epsilon
        self._m = 0
        self._v = 0
        self._d = 1
        self._t = 0
        self._params = 0
    
    def set_parameters(self, params):
        self._params = params
    
    def update(self, grads):
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        
        # Compute d_t for adaptive learning rate scaling
        d_t = 1 + np.log(v_hat + self._epsilon)
        self._d = self._beta3 * self._d + (1 - self._beta3) * d_t
        
        # Update parameters
        self._params -= (self._lr / self._d) * m_hat / (np.sqrt(v_hat) + self._epsilon)
        return self._params