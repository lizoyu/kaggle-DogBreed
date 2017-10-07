import csv
import numpy as np
from random import shuffle

def load_MNIST(filename):
    """
    load all the data from MNIST

    Input:
        - filename: path to the csv file
    Output:
        - data: list of data
    """
    with open(filename, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        # split labels and data and save in dictionary
        data= []
        for row in fileToRead:
            data.append(row)

        return data

def get_MNIST_data(num_training=41000, num_validation=1000, num_test=1000, subtract_mean=True, fit=False):
    """
    Load the MNIST dataset(42,000 in total) from disk and perform preprocessing to prepare
    it for classifiers.

    Inputs:
        - num_training: number of data used in training
        - num_validation: number of data used in validation
        - num_test: number of data used in test
        - subtract_mean: indicate whether to normalize the data

    Outputs:
        - datadict: prepared data dictionary with 'X_train', 'y_train', 'X_val', 'y_val', 
            'X_test' and 'y_test'
    """
    # load MNIST data
    mnist_path = '../data/train.csv'
    mnist_data = np.array(load_MNIST(mnist_path), dtype=np.float32)

    # shuffle and split data into training, validation and test sets
    shuffle(mnist_data)
    X_train = mnist_data[:num_training,1:].reshape((-1,28,28,1))
    y_train = mnist_data[:num_training,0]
    if num_validation:
        X_val = X_train[:num_validation,:].reshape((-1,28,28,1))
        y_val = y_train[:num_validation]
    if num_test:
        X_test = mnist_data[num_training:num_training+num_test,1:].reshape((-1,28,28,1))
        y_test = mnist_data[num_training:num_training+num_test,0]

    # extra preprocess if to fit the pretrained model
    if fit:
        new_X = np.zeros((num_training,28,28,3))
        for i in range(num_training):
            new_X[i,:,:,0] = X_train[i,:,:,0]
            new_X[i,:,:,1] = new_X[i,:,:,0]
            new_X[i,:,:,2] = new_X[i,:,:,0]
        X_train = new_X.copy()

        if num_validation:
            X_val = X_train[:num_validation]

        if num_test:
            new_X = np.zeros((num_test,28,28,3))
            for i in range(num_test):
                new_X[i,:,:,0] = X_test[i,:,:,0]
                new_X[i,:,:,1] = new_X[i,:,:,0]
                new_X[i,:,:,2] = new_X[i,:,:,0]
            X_test = new_X.copy()

    # normalize the data: subtract the mean from images
    if subtract_mean:
        mean_img = np.mean(X_train, axis=0)
        X_train -= mean_img
        if num_validation:
            X_val -= mean_img
        if num_test:
            X_test -= mean_img

    # merge into a dictionary
    datadict = {'X_train': X_train, 'y_train': y_train}
    if num_test:
        datadict['X_test'] = X_test
        datadict['y_test'] = y_test
    if num_validation:
        datadict['X_val'] = X_val
        datadict['y_val'] = y_val

    return datadict

def create_submission(model, test_path, save_path, batch_size=32, subtract_mean=True, fit=False):
    """
    use Keras trained model to create submission for Kaggle competition

    Inputs:
        - model: trained Keras model
        - test_path: the path to the test data
        - save_path: the path to save the submission with file name
        - fit: transform the test data to fit the model
    """
    # open the test file
    X_test = []
    with open(test_path, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        for x in fileToRead:
            X_test.append(x)

    X_test = np.array(X_test, dtype=np.float32).reshape((-1,28,28,1))

    # transform the data to fit the model
    if fit:
        new_X = np.zeros((X_test.shape[0],28,28,3))
        for i in range(X_test.shape[0]):
            new_X[i,:,:,0] = X_test[i,:,:,0]
            new_X[i,:,:,1] = new_X[i,:,:,0]
            new_X[i,:,:,2] = new_X[i,:,:,0]
        X_test = new_X.copy()

    # normalize the data by mean subtraction
    if subtract_mean:
        mean_img = np.mean(X_test, axis=0)
        X_test -= mean_img

    # predict and save
    predictions = np.argmax(model.predict(X_test, verbose=1, batch_size=batch_size), axis=1)
    with open(save_path, 'wt') as csvfile:
        fileToWrite = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        # write the header
        fileToWrite.writerow(['ImageID', 'Label'])
        # write the predictions
        for i in range(len(predictions)):
            fileToWrite.writerow([i+1, predictions[i]])