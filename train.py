import tensorflow as tf # type: ignore
from network import * 
import argparse
from sklearn.model_selection import train_test_split
from data_handling import MinMaxScaler
# from metrics import Metrics

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description = '\033[92m' + '\nTrain a neural network on MNIST/Fashion-MNIST\n' + '\033[0m')
    
    # dataset 
    parser.add_argument('-d', '--dataset', type = str, default = 'fashion_mnist', choices = ['mnist', 'fashion_mnist'], help = 'Dataset to use')
    
    # Neural netwrok architecture 
    parser.add_argument('-sz', '--layer_size', type = int, default = 512, help = 'Number of neurons in hidden layer')
    parser.add_argument('-nh1', '--hidden_layers', type = int, default = 3, help = 'Number of hidden layers')
    parser.add_argument('-e', '--epochs', type = int, default = 10, help = 'Number of epochs')
    
    # layer parameters 
    parser.add_argument('-a', '--activation', type = str, default = 'sigmoid', choices=['identity', 'sigmoid', 'relu', 'tanh'], help = 'Activation function')
    parser.add_argument('-bs', '--batch_size', type = int, default = 2048, help = 'Batch size')
    parser.add_argument('-w_d', '--weight_decay', type = float, default = 0, help = 'Weight decay')
    parser.add_argument('-w_i', '--initialisation', type = str, default = 'random', help = 'Parameter initialisation')
    
    # optimiser parameters
    parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-2, help = 'Learning rate')
    parser.add_argument('-o', '--optimiser', type = str, default = 'sgd', help = 'Optimiser')
    parser.add_argument('-eps', '--epsilon', type = float, default = 1e-7 , help = 'Epsilon for adam, adagrad, rmsprop, nadam, eve optimiser')
    parser.add_argument('-m', '--momentum', type = float, default = 0.9 , help = 'Momentum for nag, momentum optimiser')
    parser.add_argument('-b', '--beta', type = float, default = 0.9 , help = 'Beta for RMSProp, AdaDelta optimiser')
    parser.add_argument('-b1', '--beta1', type = float, default = 0.9 , help = 'Beta1 for Adam, Eve and Nadam')
    parser.add_argument('-b2', '--beta2', type = float, default = 0.999 , help = 'Beta2 for Adam, Eve and Nadam')
    
    # model structure 
    parser.add_argument('-v', '--validation', type = bool, default = True, help = 'Use validation')
    parser.add_argument('-l', '--loss', type = str, default = 'cross_entropy', help = 'Loss Function')
    
    # wandb configuration
    parser.add_argument('-log', '--log', type = bool, default = False, help = 'Use wandb')
    parser.add_argument('-wp', '--wandb_project', type = str, default = 'da6401_assignment1', help = 'Use wandb')
    parser.add_argument('-we', '--wand_entity', type = str, default = 'trial1', help = 'Use wandb')
    
    return parser.parse_args()

# python3.9 -m pip install -r requirements.txt
if __name__ == '__main__':
    args = get_args()

    # wandb initialisation
    if(args.log):
        wandb.init(
            project = args.wandb_project,  # project name 
            name = args.wand_entity
        )

    # -------------------------------- Dataset Loading --------------------------------
    if(args.dataset == 'mnist'): 
        print('\033[92m' + '\nLoading Mnist Data\n' + '\033[0m')
    elif(args.dataset == 'fashion_mnist'): 
        print('\033[92m' + '\nLoading Fashion Mnist Data\n' + '\033[0m')

    dataset_loader = tf.keras.datasets.fashion_mnist if args.dataset == "fashion_mnist" else tf.keras.datasets.mnist
    [(train_X, train_y), (test_X, test_y)] = dataset_loader.load_data()
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 2)

    # Number of Unique classes 
    n_classes = len(np.unique(test_y))
    print(f'\nNumber of unique classes:\t{n_classes}\n')

    print('\nSize of Training data:\t', train_X.shape)
    print('Size of Validation data:\t', val_X.shape)
    
    # -------------------------------- Dataset scaling --------------------------------
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

    # -------------------------------- One hot encoding --------------------------------
    encoder = OneHotEncoder() 
    onehot_train_y = encoder.fit_transform(y = train_y, n_class = 10)
    onehot_val_y = encoder.fit_transform(y = val_y, n_class = 10)
    onehot_test_y = encoder.fit_transform(y = test_y, n_class = 10)

    print('\033[92m' + '\nScaling and flattening the features into vectors\n' + '\033[0m')
    
    print('Size of Training data:', scaled_train_X.shape)
    print("Size of Validation data:", scaled_val_X.shape)
    print("Size of Test data:", scaled_test_X.shape)

    print('\033[92m' + '\nThe labels are encoded as one hot vectors\n' + '\033[0m')
    
    # -------------------------------- Neural network construction --------------------------------
    
    # Start with any number of inputs and end with 10 (we have 10 classes)
    layers = [Input(input_data=scaled_train_X)]
    for i in range(1, args.hidden_layers):
        layers.append(Dense(name=f'Hidden Layer: {i}', layer_size = args.layer_size, activation = args.activation))
    layers.append(Dense(name='Last_Layer', layer_size = n_classes))

    model = NeuralNetwork(
        layers = layers, 
        batch_size = args.batch_size, 
        optimiser = args.optimiser, 
        initialisation = str(args.initialisation), 
        learning_rate = args.learning_rate, 
        weight_decay = args.weight_decay,
        n_epochs = args.epochs,
        loss_function = args.loss,
        validation = args.validation, 
        val_X = scaled_val_X, 
        target = onehot_train_y,
        val_target = onehot_val_y, 
        wandb_log = args.log,  
    )

    print('\033[92m' + '\nThe Neural Network formed from the given hyper parameter(s):\n' + '\033[0m' + '\n' + f'{model}')
    
    # -------------------------------- Model Training --------------------------------
    # Train model for the given configuration 
    model.fit()

    # print model parameters
    # model.parameters
    
    metrics = Metrics()

    # -------------------------------- Model inference --------------------------------
    # predictions - yet to be implemented
    pred_y = model.predict(test_X = scaled_test_X)    

    
    print('\033[92m' + '\nClassification on test data:\n' + '\033[0m')
    accuracy = metrics.accuracy_score(test_y, pred_y)
    print(f'Testing accuracy:\t{accuracy}')
    print('\nClassification report:\n')
    print(metrics.classification_report(pred_y, test_y))

    # Log Confusion Matrix to wandb
    if(args.log):
        # Testing
        wandb.log({'Testing Confusion Matrix': wandb.plot.confusion_matrix(
            title = 'Testing Confusion Matrix',
            probs = None, 
            y_true = test_y, 
            preds = pred_y, 
            class_names = list(np.unique(test_y))
        )})

        # Training 
        pred_y = model.predict(test_X = scaled_train_X)    
        wandb.log({'Training Confusion Matrix': wandb.plot.confusion_matrix(
            title = 'Training Confusion Matrix',
            probs = None, 
            y_true = train_y, 
            preds = pred_y, 
            class_names = list(np.unique(test_y))
        )})

        # Validation 
        pred_y = model.predict(test_X = scaled_val_X)    
        wandb.log({'Validation Confusion Matrix': wandb.plot.confusion_matrix(
            title = 'Training Confusion Matrix',
            probs = None, 
            y_true = val_y, 
            preds = pred_y, 
            class_names = list(np.unique(test_y))
        )})