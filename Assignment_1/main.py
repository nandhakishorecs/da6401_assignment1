import tensorflow as tf # type: ignore
from network import * 
# from network_backup import *
from sklearn.model_selection import train_test_split
from data_handling import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# python3.9 -m pip install -r requirements.txt
if __name__ == '__main__':
    # -------------------------------- Dataset Loading --------------------------------
    [(train_X, train_y), (test_X, test_y)] = tf.keras.datasets.fashion_mnist.load_data()
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=2)

    # Number of Unique classes 
    n_classes = len(np.unique(test_y))

    print("Size of Training data:", train_X.shape)
    print("Size of Validation data:", val_X.shape)

    # scaling data
    scaler = MinMaxScaler() 

    scaled_train_X = scaler.transform(train_X)
    scaled_val_X = scaler.transform(val_X)
    scaled_test_X = scaler.transform(test_X)

    # scaled_train_X = train_X / 255
    # scaled_val_X = val_X / 255 
    # scaled_test_X = test_X / 255

    scaled_train_X = scaled_train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
    scaled_val_X = scaled_val_X.reshape(val_X.shape[0], val_X.shape[1] * val_X.shape[2]).T
    scaled_test_X = scaled_test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T

    # Encoding labels as one hot vectors 
    encoder = OneHotEncoder() 
    onehot_train_y = encoder.fit_transform(y = train_y, n_class = 10)
    onehot_val_y = encoder.fit_transform(y = val_y, n_class = 10)
    onehot_test_y = encoder.fit_transform(y = test_y, n_class = 10)

    # Start with any number of inputs and end with 10 (we have 10 classes)
    layers = [
        Input(input_data = scaled_train_X),
        Dense(name = 'Hidden Layer 1', layer_size = 32, activation = 'Sigmoid'),
        Dense(name = 'Hidden Layer 2', layer_size = 32, activation = 'Sigmoid'),
        Dense(name = 'Last_Layer', layer_size = 10) #, activation = 'Softmax')
    ]
    model = NeuralNetwork(
        layers = layers, 
        batch_size = 1024, 
        optimiser = 'Adam', 
        initialisation = 'RandomInit', 
        loss_function = 'CategoricalCrossEntropy', 
        n_epochs = 5,
        target = onehot_train_y,
        learning_rate = 1e-3, 
        validation = True, 
        val_X = scaled_val_X, 
        val_target = onehot_val_y, 
        
        weight_decay = 0.0005,
        wandb_log = False, 
        verbose = True
    )

    print(model)
    
    test = scaled_test_X
    # print(test.shape)
    # print(len(test))

    # model.forward_propagation()
    # model.backward_propagation()

    # # predictions - yet to be implemented
    # pred_y = model.predict(test_X = test)
    # print(pred_y)

    # # metrics = Metrics()
    # from sklearn import metrics
    # # metrics = sklearn.metrics()
    # # accuracy = np.sum(test_y == pred_y)
    # accuracy = metrics.accuracy_score(test_y, pred_y)
    # print(f'Accuracy:\t{accuracy}')

    # # train_acc, val_acc = model._get_accuracy()
    # # print(f'Accuracy:\t{train_acc} and {val_acc}')
