import keras 
from network import * 
from sklearn.model_selection import train_test_split
from preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

[(train_X, train_y), (temp_X, temp_y)] = keras.datasets.fashion_mnist.load_data()
test_X, val_X, test_y, val_y = train_test_split(temp_X, temp_y, test_size=0.8, random_state=42)

n_classes = 10 

scaler = MinMaxScaler() 

train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
val_X = val_X.reshape(val_X.shape[0], val_X.shape[1] * val_X.shape[2]).T
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T

encoder = OneHotEncoder() 
train_y = encoder.fit_transform(y = train_y, n_class = 10)
val_y = encoder.fit_transform(y = val_y, n_class = 10)
test_y = encoder.fit_transform(y = test_y, n_class = 10)

print(train_X.shape)
print(train_y.shape)

layers = [
    Input(input_data = train_X),
    Dense(layer_size = 32, activation = 'Tanh')
]

model = NeuralNetwork(
    layers = layers, 
    batch_size = 32, 
    optimiser = 'Vannial_GD', 
    initialisation='RandomInit', 
    loss_function='CategoricalCrossEntropy', 
    n_epochs= int(1),
    target=train_y, 
    validation=True, 
    val_X = val_X, 
    val_target = val_y, 
    # wandb_log = False,
)

model.forward_propagation()
model.backward_propagation(verbose = True)