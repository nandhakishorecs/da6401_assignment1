# DA6401 Introduction to Deep Learning - Assignment 1
This repository contains all the code for Assignment 1 for the Introduction to Deep Learning Course (DA6401) offered at Wadhwani School of Data Science and AI, Indian Institute of Technology Madras. 

**Course Instructor**: Prof. Mitesh Khapra \
**Author**: Nandhakishore C S \
**Roll Number**: DA24M011 

**Last Commit on**: Mar 9, 2025 

## Environment setup: 

- Clone the repository using ssh using git clone and install the dependencies using requirements.txt
- For MacOS - to download dataset, change the 'tensorflow' library's version to 'tensorflow-macos' with Python version 3.9. For other versions, tensorflow is not supported. 
- For Linux machines, preferably debian, keep the tensofrlow library's version as the latest version. 

Do the follwing command at the root directory where the code is cloned. 
**$ # python3.9 -m pip install -r requirements.txt**

I ran the experiments using a main.py file for ease of coding and I have added separte files for questions which needed separate code. 

## Question 1
The code for the question 1 can be accessed [here](https://github.com/nandhakishorecs/da6401_assignment1/blob/main/question1.py). The python scripts downloads and extracts the data from the keras.datasets and saves it in the local run time and prints image from each class. \
The image is logged into wandb. 

## Question 2
The code for the question 2 can be access [here](). The **network.py** file contains look up tables for optimisers, initialisers and loss functions and implements a base class named **NeuralNetwok** which implements a neural network from scratch. 

## Question 3 
All the optimisers (SGD, Momentum based GD, Nestorov Accelerated GD, Adagrad, AdaDelta, RMSProp, Adam, Nadam and Eve) are implemented. Some optimisers are implmented using inherited clases to make the code more readable and easy. Code can be accessed [here](). 

## Question 4
There is a python script named as question4.py which does sweep for the given hyperparameter table. The hyperparameter table is saved as a .yml file for better readability. \ 
A total of 120 runs were conducted and the results are published in the wandb report. 

## Question 5
The top three validation accuracies were for the following configuraions: 
1.  Validation Accuracy          : 0.88675 
    Training Accuracy            : 0.9105
    Actiavtion Function          : Tanh
    Batch Size                   : 512            
    Initialisation               : Xavier 
    Number of Neurons in a Layer : 128
    Number of Layers             : 5
    Learning Rate                : 0.001 
    Number of epochs             : 10
    Weight Decay                 : 0 
    Optimiser                    : Nadam  

2.  Validation Accuracy          : 0.8854166666666666
    Training Accuracy            : 0.91725
    Actiavtion Function          : Tanh
    Batch Size                   : 512            
    Initialisation               : He
    Number of Neurons in a Layer : 128
    Number of Layers             : 5
    Learning Rate                : 0.001 
    Number of epochs             : 10
    Weight Decay                 : 0 
    Optimiser                    : Adam  

3.  Validation Accuracy          : 0.8850833333333333
    Training Accuracy            : 0.9063958333333332
    Actiavtion Function          : Sigmoid
    Batch Size                   : 512            
    Initialisation               : He
    Number of Neurons in a Layer : 128
    Number of Layers             : 5
    Learning Rate                : 0.001 
    Number of epochs             : 20
    Weight Decay                 : 0 
    Optimiser                    : Nadam  




