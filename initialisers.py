import numpy as np          # To handle vector / matrix operations 

# ------------------- Functions / Methods to initialise weights for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Reference: https://www.deeplearning.ai/ai-notes/initialization/index.html
class RandomInit:
    @staticmethod
    def initialize(layer_size: int, mean: float = 0, std_dev: float = 1.0, epsilon: float = 1):
        return np.random.normal(loc = mean, scale = std_dev, size = layer_size) * epsilon

    def __repr__(self) -> str: 
        return 'Random'

# Reference: https://cs230.stanford.edu/section/4/#:~:text=The%20goal%20of%20Xavier%20Initialization,gradient%20from%20exploding%20or%20vanishing.
class XavierInit:
    @staticmethod
    def initialize(layer_size: int, mean: float = 0, std_dev: float = 0.005):
        return  np.random.normal(loc = mean, scale = std_dev, size = layer_size) * (1 / layer_size[0])
    
    def __repr__(self) -> str: 
        return 'Xavier'

class HeInit:
    @staticmethod
    def initialize(layer_size: int, mean: float = 0, std_dev: float = 0.05):
        return  np.random.normal(loc = mean, scale = std_dev, size = layer_size) * np.sqrt(2 / layer_size[0])
        
    def __repr__(self) -> str: 
        return 'He'
