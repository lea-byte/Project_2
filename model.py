# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:50:31 2021

@author: leagr
"""
import numpy as np
import tensorflow as tf
from tf import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import add, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from metrics import f1

def Down_layer(data, nbr_filters, filter_size, batch=True, training=False, max_pool=True):
       # first layer
       #Convolution of filters with the layer input 
       cv1 = Conv2D(nbr_filters, kernel_size=filter_size, strides=1, padding="same", 
               kernel_initializer="glorot_uniform")(data)
       
           #Call argument: 4D tensor with shape: [batch size, rows, cols, channels] or [image index, y, x, depth]
           #nbr_filters= the number of output filters in the convolution
           #kernel_size: an integer specifying the height and width of the 2D convolution window.
           #strides=1 by defaults 
           #padding='same' i.e. the output feature map has the same size as the input
           #kernel_initializer: Initializer for the kernel weights matrix, by defaults 'glorot_uniform', try "he_normal"
       
       if batch:
           cv1 = BatchNormalization()(cv1, training)
               #training=boolean indicating whether the layer should behave in training mode or in inference mode
           
       cv1 = Activation('relu')(cv1) #rectified linear non-linearity.
       
       # second layer
       cv2 = Conv2D(filters = n_filters,  kernel_size=filter_size, strides=1, padding="same", 
               kernel_initializer="glorot_uniform")(cv1)
           
       if batch:
           cv2 = BatchNormalization()(cv2, training)
       
       cv2 = Activation('relu')(cv2)
       
       # Max pooling
       if max_pool:
           cv_2 = MaxPooling2D(pool_size=2, strides=2, padding="same")(cv2)
           #pool_size=2 i.e. will take the max value over a 2x2 pooling window.       
 
       return cv_2
        
   
    
def Up_layer(data, down_layer, nbr_filters, filter_size, dropout_rate=0.5, batch=True):
       
    cv_T1 = Conv2DTranspose(nbr_filters, kernel_size=filter_size, strides=2, padding = 'same', 
                         kernel_initializer='glorot_uniform')(data)
    cv_T1 = concatenate([cv_T1, down_layer])
    cv_T1 = Dropout(dropout_rate)(cv_T1)
    cv_T1 = Down_layer(cv_T1, nbr_filters, filter_size, batch=True, training=False)
   
    return cv_T1    
       
   
def unet(input_img = (None,None,3), nbr_filters=16, dropout_rate = 0.1,
         batch = True, loss = binary_crossentropy, optimizer = Adam(learning_rate=1e-3), training = False): #(learning_rate=1e-4)
   
   filter_size = 3
    
   #larger combinations of patterns to capture so the number of filters increases with the number of hidden layers   
   layer1 = Down_layer(input_img, nbr_filters, filter_size, batch, training)

   layer2 = Down_layer(layer1, nbr_filters*2, filter_size, batch, training)
   
   layer3 = Down_layer(layer2, nbr_filters*4, filter_size, batch, training)
    
   layer4 = Down_layer(layer3, nbr_filters*8, filter_size, batch, training) 
    
   layer5 = Down_layer(layer4, nbr_filters*16, filter_size, batch, training, max_pool=False)

   filter_size = 2

   layer6 = Up_layer(layer5, layer4, nbr_filters*8, filter_size, dropout_rate, batch)
   
   layer7 = Up_layer(layer6, layer3, nbr_filters*4, filter_size, dropout_rate, batch)
   
   layer8 = Up_layer(layer7, layer2, nbr_filters*2, filter_size, dropout_rate, batch)
   
   layer9 = Up_layer(layer8, layer1, nbr_filters, filter_size, dropout_rate, batch)
   
   #classifier
   layer10 = Conv2D(1, 1, activation='sigmoid')(layer9)

   model = Model(inputs = [input_img], outputs = [layer10])

   model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
    

   return model
   
   
   
   
   