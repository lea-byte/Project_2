import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
import tensorflow 
import keras
from skimage.transform import rotate
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPool2D, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Add, Concatenate
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizers import Adam

# Convolution part of the encoder

def convolution_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding = "same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

# Enconder block - output is a maxpool and the skip connection

def encoder_block(input, num_filters):
    x = convolution_block(input, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p

# decoder block
def decoder_block(input, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides = 2, padding = "same")(input)
    x = Concatenate()([x,skip])
    return x
# Entire U-net Algorithm    
def build_unet(input_shape):
    inputs = Input(input_shape)
    
    #Encoding part of the U-net
    
    # s's are the skip connections and p's are the maxpool results
    s1, p1 = encoder_block(inputs,64)
    s2, p2 = encoder_block(p1,128)
    s3, p3 = encoder_block(p2,256)
    s4, p4 = encoder_block(p3,512)

    # Bottle-neck:
    
    b1 = convolution_block(p4,1024)
    
    #Deconding part of the U-net
    
    d1 = decoder_block(b1,s4,512)
    d2 = decoder_block(d1,s3,256)
    d3 = decoder_block(d2,s2,128)
    d4 = decoder_block(d3,s1,64)
    
    outputs = Conv2D (1, 1, padding = "same", activation = "sigmoid")(d4)
    model = Model(inputs, outputs, name = "U-Net")
    return model


#input_shape = (400, 400, 3)
#model = build_unet(input_shape)
#model.summary()
