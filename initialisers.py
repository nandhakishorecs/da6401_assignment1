import numpy as np          # To handle vector / matrix operations 

# ------------------- Functions / Methods to initialise weights for Neural Networks -------------------

# Reference: https://www.deeplearning.ai/ai-notes/initialization/index.html
class Random: 
    __slots__ = '_prev', '_curr', '_mu', '_sigma'
    def __init__(self, mean: float, std_dev: float, previous: float, current: float) -> None:
        self._mu = mean
        self._sigma = std_dev
        self._prev = previous
        self._curr = current

    def init_weights_biases(self):
        # returns Tuple 
        W = np.random.normal(
            loc = self._mu, 
            scale = self._sigma, 
            size = (self._prev, self._curr)
        )
        b = np.random.normal(
            loc = self._mu, 
            scale = self._sigma, 
            size = (self._curr, )
        )
        return W, b

# Reference: https://cs230.stanford.edu/section/4/#:~:text=The%20goal%20of%20Xavier%20Initialization,gradient%20from%20exploding%20or%20vanishing.
class Xavier: 
    __slots__ = '_prev', '_curr'
    def __init__(self, previous: float, current: float) -> None:
        self._curr = current
        self._prev = previous

    def init_weights_biases(self):
        # returns Tuple 
        # the variance remain same
        upper_bound = np.sqrt(6/(self._prev + self._curr))
        lower_bound = (-1) * upper_bound
        W = np.random.uniform(
            low = lower_bound, 
            high = upper_bound, 
            size = (self._prev, self._curr)
        )
        b = np.zeros(
            shape = (self._curr, ),
            dtype = np.float64
        )
        return W, b