import numpy as np          # To handle vector / matrix operations 

# ------------------- Functions / Methods to initialise weights for Neural Networks -------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
# --------------------------------------------------------------------------

# Reference: https://www.deeplearning.ai/ai-notes/initialization/index.html
class RandomInit:
    @staticmethod
    def initialize(input_size: int, output_size: int, epsilon: float = 0.01) -> float:
        return np.random.randn(input_size, output_size) * epsilon

# Reference: https://cs230.stanford.edu/section/4/#:~:text=The%20goal%20of%20Xavier%20Initialization,gradient%20from%20exploding%20or%20vanishing.
class XavierInit:
    @staticmethod
    def initialize(input_size: int, output_size: int) -> float:
        return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)