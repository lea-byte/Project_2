# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:28:00 2021

@author: leagr
"""


import numpy as np
import random
from PIL import Image 

def rotate_images(X, Y, degrees):
    """
     Used for data augmentation by rotating the images in the input data set
     X: list of images in train set 
     Y: list of groundtruth images in train set
    """

    # X = np.asarray(X)
    # Y = np.asarray(Y)
    # X_rot = np.zeros(X.shape) #returns shape: [image indeX, rows, cols, depth]
    # Y_rot = np.zeros(Y.shape)
    
    X_rot = X
    Y_rot = Y
    X_temp = X #because one is going to modify the original input several times
    Y_temp = Y


    #rotate of some degree all images and add them to the image vector
    for deg in degrees:
        for i in range(len(X_temp)):
            X_rot[i] = X_temp[i].rotate(deg, reshape=False, mode='grid-mirror')
            Y_rot[i] = Y_temp[i].rotate(deg, reshape=False, mode='grid-mirror')
        
        X = np.concatenate([X, X_rot])
        Y = np.concatenate([Y, Y_rot])
        
    
    return X, Y

def add_gaussian_noise(X, Y, mean = 0, sigma = 0.1):
    '''
    Add gaussian noise to the train data set
    '''
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    X_noisy = np.zeros(X.shape) 
    Y_noisy = np.zeros(Y.shape)
    
    
    noise = np.random.normal(mean, sigma, X[1].shape) #all the images have the same size
    
    for i in range(len(X)): 
        X_noisy[i] =  X[i] + noise
        Y_noisy[i] =  Y[i] + noise
        
    X = np.concatenate([X, X_noisy])
    Y = np.concatenate([Y, Y_noisy])
           
    return X, Y



