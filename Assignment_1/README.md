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
The code for the question 2 can be access [here](https://github.com/nandhakishorecs/DA6401_IDL_Assignments/blob/main/Assignment_1/network.py). The **network.py** file contains look up tables for optimisers, initialisers and loss functions and implements a base class named **NeuralNetwok** which implements a neural network from scratch. 

## Question 3 
All the optimisers (SGD, Momentum based GD, Nestorov Accelerated GD, Adagrad, AdaDelta, RMSProp, Adam, Nadam and Eve) are implemented. Some optimisers are implmented using inherited clases to make the code more readable and easy. Code can be accessed [here](https://github.com/nandhakishorecs/DA6401_IDL_Assignments/blob/main/Assignment_1/optimisers.py). 

## Question 4
There is a python script named as question4.py which does sweep for the given hyperparameter table. The hyperparameter table is saved as a .yml file for better readability. \
A total of 120 runs were conducted and the results are published in the wandb report. The code can be accessed [here](https://github.com/nandhakishorecs/DA6401_IDL_Assignments/blob/main/Assignment_1/question4.py)

## Question 5
The best Validation Accuracy for the given sweep was: 0.88675. 

## Question 6
From the parallel co-ordinates plot, we can see that, some of the runs has an accuracy of 10% and less. 
- The weight decay for these runs were close to 1, (i.e.) 0.5 and this caused the features to be more sparse than the actually needed sparsity. Also, these models were simple models with 2 layers and 64 or 32 neurons in each layer. 
- When the correctly contributing features' weights are purposefully diminished, they tend do converge to zero, causing very less accuracies. 

For the configurations with maximum accuracies, more than 75% in many configurations 
- For the models where the number of layers was more than 5 and each layer had more than 64 neurons, the complexity of the model was too high. Thus, weight decay of 0.0005 helped to reduced the model complexity and give a good trade off between training and validation accuracy. 
- For the large models (layers $\ge$ 4 , neurons per layer $\ge$ 128), the learning rate was 0.01 and for small models (layers $<$ 4 and neurons per layer $\le$ 64 ), the learning rate was 0.001 or less. 

Configurations of sweep which got close to 95% accuracy. 

1.  **Validation Accuracy**          : 0.88675 \
    **Training Accuracy**            : 0.9105   \
    **Actiavtion Function**          : Tanh \
    **Batch Size**                   : 512                 
    **Initialisation**               : Xavier   \
    **Number of Neurons in a Layer** : 128  \
    **Number of Layers**             : 5    \
    **Learning Rate**                : 0.001    \
    **Number of epochs**             : 10   \
    **Weight Decay**                 : 0    \
    **Optimiser**                    : Nadam    

2.  **Validation Accuracy**          : 0.8854166666666666   \
    **Training Accuracy**            : 0.91725  \
    **Actiavtion Function**          : Tanh \
    **Batch Size**                   : 512             
    **Initialisation**               : He   
    **Number of Neurons in a Layer** : 128  \
    **Number of Layers**             : 5    \
    **Learning Rate**                : 0.001    \
    **Number of epochs**             : 10   \
    **Weight Decay**                 : 0    \
    **Optimiser**                    : Adam     

3.  **Validation Accuracy**          : 0.8850833333333333   \
    **Training Accuracy**            : 0.9063958333333332   \
    **Actiavtion Function**          : Sigmoid  \
    **Batch Size**                   : 512            
    **Initialisation**               : He   \
    **Number of Neurons in a Layer** : 128  \
    **Number of Layers**             : 5    \
    **Learning Rate**                : 0.001    \
    **Number of epochs**             : 20   \
    **Weight Decay**                 : 0    \
    **Optimiser**               : Nadam     

## Question 7

Refer wandb report for the confusion matrix

## Question 8
When categorical cross entropy loss is replaced by mean squared error loss, the accuracy drops drastically and leads to zero. The follwing points might be the reason why ths behavior is observed. 
- In categorical cross entropy, the loss is calculates as the difference between two probability distributions. Thus the error value is more relevelant it is incorporating the nature of the probability that, sum of the distribution is one and all the values in the probability distribution (predicted & true output ) are positive and between zero and 1 
- In Mean Squared Error (MSE), the difference between the predicted and actual error is measured in terms of euclidian distance, which ignore the properties of the probaiblity distribuions which are the output. This leads to the fact that, for classification task, MSE is not a good choice and it is sensitive to distances. 

## Question 9
This repository contains the complete code for Assignment 1 in prescribed format. 

## Question 10
For MNIST dataset, the folowing three configurations were tried and the details are as follows: 

1.  **Validation Accuracy**          : 0.8798333333333334 \
    **Training Accuracy**            : 0.9159166666666668   \
    **Actiavtion Function**          : Tanh \
    **Batch Size**                   : 32                 
    **Initialisation**               : Xavier   \
    **Number of Neurons in a Layer** : 64  \
    **Number of Layers**             : 4    \
    **Learning Rate**                : 0.001    \
    **Number of epochs**             : 20   \
    **Weight Decay**                 : 0    \
    **Optimiser**                    : Nadam    

2.  **Validation Accuracy**          : 0.8770833333333333   \
    **Training Accuracy**            : 0.904375  \
    **Actiavtion Function**          : Tanh \
    **Batch Size**                   : 32
    **Initialisation**               : He   
    **Number of Neurons in a Layer** : 64  \
    **Number of Layers**             : 5    \
    **Learning Rate**                : 0.0001    \
    **Number of epochs**             : 20   \
    **Weight Decay**                 : 0    \
    **Optimiser**                    : Nadam     

3.  **Validation Accuracy**          : 0.8565\
    **Training Accuracy**            : 0.90925\
    **Actiavtion Function**          : Sigmoid  \
    **Batch Size**                   : 32            
    **Initialisation**               : Random   \
    **Number of Neurons in a Layer** : 128  \
    **Number of Layers**             : 5    \
    **Learning Rate**                : 0.001    \
    **Number of epochs**             : 20   \
    **Weight Decay**                 : 0    \
    **Optimiser**               : Eve    

MNIST and Fashion MNIST datasets share a lot of common structure: 
- both the datasets have 10 classess and same number of samples. 
- both datasets have images of size (28 $\times$ 28) and they are greyscale. 
- All of these images are low on resolution 
- With efficient optimisers, and corrcted batch size, we can get good accuracies. (Reason: MNIST had numerical digits written in a 28 $\times$ 28 canvas and thus the numbers occupy a small space in the canvas, mostly positioned at the center; whereas, fashion_mnist has data which is spread across the $28 \times 28$ canvas)