import keras 
from network import * 
from sklearn.model_selection import train_test_split
from preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

[(train_X, train_y), (test_X, test_y)] = keras.datasets.fashion_mnist.load_data()
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

n_classes = 10 

print("Size of Training data:", train_X.shape)
print("Size of Validation data:", val_X.shape)

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

encoder = OneHotEncoder() 
train_y = encoder.fit_transform(y = train_y, n_class = 10)
val_y = encoder.fit_transform(y = val_y, n_class = 10)
test_y = encoder.fit_transform(y = test_y, n_class = 10)

# Start with any number of inputs and end with 10 (we have 10 classes)
layers = [
    Input(input_data = scaled_train_X),
    Dense(name = 'Trail layer1', layer_size = 10, activation = 'Sigmoid')
]


'''
'Vannial_GD' : VanilaGradientDescent(),
'Momentum_GD' : MomentumGD(), 
'Nestorov': NesterovMomentumGD(), 
'AdaGrad': AdaGrad(), 
'RMSProp': RMSProp(),
'AdaDelta': AdaDelta(),
'Adam': Adam() 
'''
model = NeuralNetwork(
    layers = layers, 
    batch_size = 2048, 
    optimiser = 'Adam', 
    initialisation='RandomInit', 
    loss_function='CategoricalCrossEntropy', 
    n_epochs= int(1),
    target=train_y, 
    validation=True, 
    val_X = scaled_val_X, 
    val_target = val_y, 
    wandb_log = False,
)

model.forward_propagation()
model.backward_propagation(verbose = False)