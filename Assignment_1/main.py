import tensorflow as tf # type: ignore
from network import * 
# from network_backup import *
from sklearn.model_selection import train_test_split
from data_handling import MinMaxScaler
from metrics import Metrics

import warnings
warnings.filterwarnings("ignore")

map_datasets = {
    "mnist": tf.datasets.MNIST, 
    "fashion_mnist": tf.datasets.FashionMNIST
}

# python3.9 -m pip install -r requirements.txt
if __name__ == '__main__':
    # wandb initialisation
    # wandb.init(
    #     project = 'da6401_assignment1',  # project name 
    #     name = 'Best Model: 1 - MNIST'
    # )

    # -------------------------------- Dataset Loading --------------------------------
    # print('\033[92m' + '\nLoading Fashion Mnist Data\n' + '\033[0m')
    # [(train_X, train_y), (test_X, test_y)] = tf.keras.datasets.fashion_mnist.load_data()
    [(train_X, train_y), (test_X, test_y)] = tf.keras.datasets.mnist.load_data()
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 2)

    # Number of Unique classes 
    print(np.unique(test_y))
    n_classes = len(np.unique(test_y))
    print(n_classes)
    # class_map = {
    #     0 : "T-shirt/top", 
    #     1 : "Trouser", 
    #     2 : "Pullover",
    #     3 : "Dress", 
    #     4 : "Coat", 
    #     5 : "Sandal", 
    #     6 : "Shirt",
    #     7 : "Sneaker", 
    #     8 : "Bag",
    #     9 : "Ankle boot"  
    # }
    # classes = np.array((class_map.values()))

    print("Size of Training data:", train_X.shape)
    print("Size of Validation data:", val_X.shape)
    # print('Labels present in fashion mnist dataset: ', ", ".join(class_map.values()))
    
    # # scaling data
    scaler = MinMaxScaler() 

    scaled_train_X = scaler.transform(train_X)
    scaled_val_X = scaler.transform(val_X)
    scaled_test_X = scaler.transform(test_X)

    scaled_train_X = train_X / 255
    scaled_val_X = val_X / 255 
    scaled_test_X = test_X / 255

    scaled_train_X = scaled_train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
    scaled_val_X = scaled_val_X.reshape(val_X.shape[0], val_X.shape[1] * val_X.shape[2]).T
    scaled_test_X = scaled_test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T

    # Encoding labels as one hot vectors 
    encoder = OneHotEncoder() 
    onehot_train_y = encoder.fit_transform(y = train_y, n_class = 10)
    onehot_val_y = encoder.fit_transform(y = val_y, n_class = 10)
    onehot_test_y = encoder.fit_transform(y = test_y, n_class = 10)

    print('\033[92m' + '\nScaling and flattening the features into vectors\n' + '\033[0m')
    
    print('Size of Training data:', scaled_train_X.shape)
    print("Size of Validation data:", scaled_val_X.shape)
    print("Size of Test data:", scaled_test_X.shape)

    print('\033[92m' + '\nThe labels are encoded as one hot vectors\n' + '\033[0m')
    

    # Start with any number of inputs and end with 10 (we have 10 classes)
    layers = [
        Input(input_data = scaled_train_X),
        Dense(name = 'Hidden Layer 1', layer_size = 128, activation = 'Sigmoid'),
        Dense(name = 'Hidden Layer 2', layer_size = 128, activation = 'Sigmoid'),
        Dense(name = 'Hidden Layer 3', layer_size = 128, activation = 'Sigmoid'),
        Dense(name = 'Last_Layer', layer_size = 10) #, activation = 'Softmax')
    ]
    model = NeuralNetwork(
        layers = layers, 
        batch_size = 1024, 
        optimiser = 'Nadam', 
        initialisation = 'HeInit', 
        learning_rate = 1e-2, 
        weight_decay = 0.0005,
        n_epochs = 5,
        loss_function = 'CategoricalCrossEntropy', 
        target = onehot_train_y,
        validation = True, 
        val_X = scaled_val_X, 
        val_target = onehot_val_y, 
        wandb_log = False,  
    )

    print('\033[92m' + '\nThe Neural Network formed from the given hyper parameter(s):\n' + '\033[0m' + '\n' + f'{model}')
    
    # Train model for the given configuration 
    model.fit()
    
    # metrics = Metrics()


    # # predictions - yet to be implemented
    # pred_y = model.predict(test_X = scaled_test_X)    

    
    # print('\033[92m' + '\nClassification on test data:\n' + '\033[0m')
    # accuracy = metrics.accuracy_score(test_y, pred_y)
    # print(f'Testing accuracy:\t{accuracy}')
    # print('\nClassification report:\n')
    # print(metrics.classification_report(pred_y, test_y))

    # # Log Confusion Matrix to wandb
    
    # # # Testing
    # # wandb.log({'Testing Confusion Matrix': wandb.plot.confusion_matrix(
    # #     title = 'Testing Confusion Matrix',
    # #     probs = None, 
    # #     y_true = test_y, 
    # #     preds = pred_y, 
    # #     class_names = list(class_map.values())
    # # )})

    # # # Training 
    # # pred_y = model.predict(test_X = scaled_train_X)    
    # # wandb.log({'Training Confusion Matrix': wandb.plot.confusion_matrix(
    # #     title = 'Training Confusion Matrix',
    # #     probs = None, 
    # #     y_true = train_y, 
    # #     preds = pred_y, 
    # #     class_names = list(class_map.values())
    # # )})

    # # # Validation 
    # # pred_y = model.predict(test_X = scaled_val_X)    
    # # wandb.log({'Validation Confusion Matrix': wandb.plot.confusion_matrix(
    # #     title = 'Training Confusion Matrix',
    # #     probs = None, 
    # #     y_true = val_y, 
    # #     preds = pred_y, 
    # #     class_names = list(class_map.values())
    # # )})
