import wandb
import keras
import numpy as np
from network import NeuralNetwork  
from layers import Dense, Input
from data_handling import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


[(train_X, train_y), (test_X, test_y)] = keras.datasets.fashion_mnist.load_data()
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# Number of Unique classes 
n_classes = len(np.unique(test_y))

# logging
# print("Size of Training data:", train_X.shape)
# print("Size of Validation data:", val_X.shape)

# scaling data
scaler = MinMaxScaler() 

train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

scaled_train_X = train_X / 255
scaled_val_X = val_X / 255 
scaled_test_X = test_X / 255

scaled_train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
scaled_val_X = val_X.reshape(val_X.shape[0], val_X.shape[1] * val_X.shape[2]).T
scaled_test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T

# Encoding labels as one hot vectors 
encoder = OneHotEncoder() 
train_y = encoder.fit_transform(y = train_y, n_class = 10)
val_y = encoder.fit_transform(y = val_y, n_class = 10)
test_y = encoder.fit_transform(y = test_y, n_class = 10)

import yaml

with open("sweep_config.yml", "r") as file:
    sweep_config = yaml.safe_load(file)

# Define sweep configuration
# sweep_config = {
#     'name': 'complete_sweep_1',
#     'method': 'bayes',  # use 'grid', 'random' or 'bayes' for different search methods
#     'metric': {'name': 'Validation_Accuracy', 'goal': 'maximize'},
#     'parameters': {
#         'activation': {'values': ['Sigmoid', 'ReLU', 'Tanh']}, 
#         'layer_size': {'values': [32, 64, 128]}, 
#         'n_layers': {'values': [3, 4, 5]},
#         'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
#         'batch_size': {'values': [32, 64, 128, 512, 1024]}, 
#         'optimiser': {'values': ['SGD', 'Momentum_GD', 'Nestorov', 'RMSProp', 'Adam', 'Nadam', 'Eve']},
#         'n_epochs': {'values': [10, 20]},
#         'initialisation': {'values': ['RandomInit', 'XavierInit', 'HeInit']}, # 
#         'weight_decay': {'values': [0, 0.0005, 0.5]}
#         # 'loss_function': {'values': ['CategoricalCrossEntropy', 'MeanSquaredError']}
#     }
# }

sweep_id = wandb.sweep(sweep_config,project='da6401_assignment1')  # project name 

def train_sweep():
    wandb.init(
        entity='Trial_4',  # team name 
        project='da6401_assignment1'  # project name 
    )
    config = wandb.config
    wandb.run.name = f"act_{config.activation}_hl_{config.layer_size}_nlayers_{config.n_layers}_lr_{config.learning_rate}_bs_{config.batch_size}_opt_{config.optimiser}_e_{config.n_epochs}_init_{config.initialisation}_wd_{config.weight_decay}"#_loss_{config.loss_function}" 


    # Create input layer
    input_layer = Input(scaled_train_X)
    
    # Create hidden layers based on sweep config
    layers = [input_layer]
    for i in range(config.n_layers-1):
        layers.append(Dense(name=f"Hidden_Layer_{i+1}", layer_size=config.layer_size, activation=config.activation))
        
    layers.append(Dense(name='Last_Layer', layer_size=n_classes)) # default Softmax

    # Create neural network instance
    model = NeuralNetwork(
        # hyper parameters from config 
        layers=layers,
        batch_size=config.batch_size,
        optimiser=config.optimiser,
        n_epochs=config.n_epochs,
        initialisation=config.initialisation,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        # default values 
        loss_function='cross_entropy',
        target=train_y,
        validation=True,
        val_X=scaled_val_X,
        val_target=val_y,
        wandb_log=True,
    )
    
    # Train the model
    model.fit()
    
    wandb.finish()

# Run sweep
wandb.agent(sweep_id, function=train_sweep, count=1)