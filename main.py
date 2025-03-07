import keras 
from network import * 
from sklearn.model_selection import train_test_split
from preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

[(train_X, train_y), (temp_X, temp_y)] = keras.datasets.fashion_mnist.load_data()
test_X, val_X, test_y, val_y = train_test_split(temp_X, temp_y, test_size=0.8, random_state=42)

scaler = MinMaxScaler() 

train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
val_X = val_X.reshape(val_X.shape[0], val_X.shape[1] * val_X.shape[2]).T
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T

encoder = LabelEncoder() 

print(train_y)

# train_y = encoder.fit_transform(train_y)
# val_y = encoder.fit_transform(val_y)
# test_y = encoder.fit_transform(test_y)

# print(train_y)

layers = [
    Input(X = train_X),
    Dense(size = 32, activation = 'Tanh', name = 'test_layer')
]

model = NeuralNetwork(
    layers = layers, 
    batch_size = 32, 
    optimiser = 'Vannial_GD', 
    initilaisation='Random_normal', 
    loss_function='Categorical_Cross_Entropy', 
    n_epochs= int(1),
    target=train_y, 
    validation=True, 
    validation_features = val_X, 
    validation_target= val_y, 
    wandb = False,
    optimised_parameters=None
)

model._forward_propagation()
model._backward_propagation(verbose = True)