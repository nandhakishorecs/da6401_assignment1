import numpy as np          # To handle vector / matrix operations 

# ------------------- Functions / Methods to initialise weights for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Reference: https://www.deeplearning.ai/ai-notes/initialization/index.html
class RandomInit:
    @staticmethod
    def initialize(layer_size: int, mean: int = 0, std_dev: int = 1.0, epsilon: float = 0.001):
        return np.random.normal(loc = mean, scale = std_dev, size = layer_size) * epsilon
    # randn(input_size, output_size) * epsilon

# Reference: https://cs230.stanford.edu/section/4/#:~:text=The%20goal%20of%20Xavier%20Initialization,gradient%20from%20exploding%20or%20vanishing.
class XavierInit:
    @staticmethod
    def initialize(layer_size: int, mean: int = 0, std_dev: int = 0.05):
        return  np.random.normal(loc = mean, scale = std_dev, size = layer_size) * np.sqrt(1 / layer_size[0])
        # return  np.random.randn(*layer_size) * np.sqrt(1 / layer_size[1])

class HeInit:
    @staticmethod
    def initialize(layer_size: int, mean: int = 0, std_dev: int = 0.05):
        return  np.random.normal(loc = mean, scale = std_dev, size = layer_size) * np.sqrt(2 / layer_size[0])
        # return  np.random.randn(*layer_size) * np.sqrt(2 / layer_size[1])
