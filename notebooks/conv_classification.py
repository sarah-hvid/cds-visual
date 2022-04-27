"""
A script that performs a neural network classification.
As input, provide either the mnist_784 or cifar10 dataset after the flag: -d.

Usage example:
    For cifar10 dataset:
    $ python nn_classifier.py -d cifar10
    
    For mnist_784 dataset:
    $ python nn_classifier.py -d mnist_784
    
Requirements:
The folder structure must be the same as in the GitHub repository.
The current working directory when running the script must be the one that contains the data, output and src folder. 
The output folder must be named output.
"""

# path tools
import sys, os
import argparse

# image processing
import cv2

# data tools
import os
import numpy as np
import matplotlib.pyplot as plt

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD


# function that specifies the required arguments
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-d", "--data", required = True, help = "The dataset we want to work with, mnist_784 or cifar10")

    args = vars(ap.parse_args())
    return args


def load_mnist():
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def minmax_scaling(data):
    X_norm = (data - data.min()) / (data.max() - data.min())
    return X_norm


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        random_state=9,
                                                        train_size=7500, 
                                                        test_size=2500)
    
    #scaling the features
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def plot_report(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plt.savefig('conv.png', bbox_inches='tight') # not in output folder
    plt.show()

def load_cifar():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
     
    # scaling
    X_train = minmax_scaling(X_train)
    X_test = minmax_scaling(X_test)
 
    #create one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    return X_train, X_test, y_train, y_test


def init_model():
    # initialising model
    model = Sequential()

    # define conv layer and relu layer. always defined together. CONV => ReLU
    model.add(Conv2D(32,                       # we're using conv2d because we're only using height and width.
                     (3,3),                    # kernel size
                     padding = 'same',         # using 0-padding. 
                     input_shape = (32, 32, 3)))# input shape of each image

    model.add(Activation('relu'))              #adding activation layer

    # FC classifier
    model.add(Flatten())
    model.add(Dense(10)) # number of predictons available
    model.add(Activation('softmax'))
    
    return model

def fit_model(model, X_train, X_test, y_train, y_test):
    # define the gradient descent
    sgd = SGD(0.01)
    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=sgd,
                  metrics=["accuracy"])
    
    history = model.fit(X_train, y_train,
                    validation_data = (X_test, y_test),
                    epochs = 10,
                    batch_size = 32)
    
    # get predictions
    predictions = model.predict(X_test, batch_size=32)   
    
    # initialize label names
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # get classification report
    report = classification_report(y_test.argmax(axis=1), 
                                predictions.argmax(axis=1), 
                                target_names=label_names)
    # save plot
    plot_report(history, 10) 
    
    # save report
    with open("conv_report_cifar10.txt", "w") as f: # not in output for now
            print(report, file=f)


def main():
    
    # check if arguments are provided
    script = sys.argv[0]
    if len(sys.argv) == 1: # no arguments, so print help message
        print("""Error: an input is required\nUsage: input -d flag followed by the dataset""")
        return
    
    # parse arguments
    args = parse_args()
    data_set = args['data']
    
    # model process
    if data_set == 'mnist_784':
        X, y = load_mnist()
        X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, y)
        fit_model(X_train_scaled, y_train, X_test_scaled, y_test)
     
    elif data_set == 'cifar10':
        X_train, X_test, y_train, y_test = load_cifar()
        model = init_model()
        fit_model(model, X_train, X_test, y_train, y_test)
    
    else:
        print('Input is in the wrong format. Write mnist_784 for digits dataset and cifar10 for animal/vehicles dataset.')
    
    
    
if __name__ == '__main__':
    main()