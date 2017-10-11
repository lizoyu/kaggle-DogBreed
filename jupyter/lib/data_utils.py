import os
import csv
import json
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from random import shuffle

def create_npy(trainpath='../../data/train/', 
                testpath='../../data/test/', 
                labelpath='../../data/labels.csv', 
                savepath='../../data/all.npy'):
    """
    Load all images, and related labels for train data, and save in an npy file.
    """
    # save the labels in a dictionary
    with open(labelpath, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        # 'id' as key and 'breed' as value
        labels = {}
        for row in fileToRead:
            key, val = row
            labels[key] = val
    
    data = {'X_train':[], 'y_train':[], 'X_test':[]}
    
    # save train data and related labels
    for file in os.listdir(trainpath):
        img = imresize(imread(trainpath+file), (32,32))
        data['X_train'].append(img)
        data['y_train'].append(labels[file[:-4]])
    print('Train data loading complete.')

    # save test data
    for file in os.listdir(testpath):
        img = imresize(imread(testpath+file), (32,32))
        data['X_test'].append(img)
    print('Test data loading complete.')

    # save to npy
    with open(savepath, 'wb') as file:
        np.save(file, data)
    print('Saved to npy file.')

#create_npy()

def tester():
    with open('../../data/all.npy', 'rb') as file:
        data = np.load(file)
    data = data[()]
    print(len(data['X_test']), len(data['X_train']), len(data['y_train']), data['X_test'][0].shape)

tester()