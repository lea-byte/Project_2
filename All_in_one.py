import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K





def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)





smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)




# MAYBE USE BINARY CROSS ENTROPY
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)




##### DATA ######




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize

import os,sys
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split




# creates a directory if it does not exist
def Directory_maker(path):
        if not os.path.exists(path):
            os.makedirs(path)





# Loading_images
def load_data(path, split = 0.2):
    new_path_image = os.path.join(f"{path}",os.path.join("images","")+"*.png")
    new_path_groundtruth = os.path.join(f"{path}",os.path.join("groundtruths","")+"*.png")
    images = sorted(glob(new_path_image))    
    print(len(images))
    groundtruths = sorted(glob(new_path_groundtruth))
    split_size = int(len(images)*split)
    train_x, valid_x = train_test_split(images, test_size = split_size, random_state = 42)
    train_y, valid_y = train_test_split(groundtruths, test_size = split_size, random_state = 42)
    return (train_x, train_y), (valid_x, valid_y)










def augment_data(images, groundtruths, save_path, specials , flips = False, rotations = False):
        
        for idx, (x,y) in tqdm(enumerate(zip(images, groundtruths)), total = len(images)):
            #print("this is the idx:")
            #print(idx)
            #Getting the image name
            temporary_string = os.path.join("a","a")
            forward_or_backslash = temporary_string[-2]
            name = x.split(forward_or_backslash)[-1].split(".")[0]
            Number = (int(name.split("_")[-1]))
            
            ## Read the images and groundtruths
            
            x = mpimg.imread(x)
            y = mpimg.imread(y)
            
            if ((flips == True) or (Number in specials)):
                # flip
                #vertical
                x1 = x[:, ::-1,:]
                y1 = y[:, ::-1]
                
                #Horizontal
                x2 = x[::-1,:,:]
                y2 = y[::-1, :]
                
                if rotations or (Number in specials):
                    # rotations
                    x3 = rotate(x,90)
                    y3 = rotate(y,90)
                    x4 = rotate(x,180)
                    y4 = rotate(y,180)
                    x5 = rotate(x,270)
                    y5 = rotate(y,270)

                    x6 = rotate(x1,90)
                    y6 = rotate(y1,90)
                    x7 = rotate(x1,180)
                    y7 = rotate(y1,180)
                    x8 = rotate(x1,270)
                    y8 = rotate(y1,270)

                    x9 = rotate(x2,90)
                    y9 = rotate(y2,90)
                    x10 = rotate(x2,180)
                    y10 = rotate(y2,180)
                    x11 = rotate(x2,270)
                    y11 = rotate(y2,270)

                    # Merging all transformations
                    X = [x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]
                    Y = [y,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11]
                else:
                    X = [x,x1,x2]
                    Y = [y,y1,y2]
            else:
                X = [x]
                Y = [y]
                
            idx = 0
            for i, gt in zip(X,Y):
                gt = gt
                gt = (gt > 0.5)*255
                if len(X) == 1:
                    tmp_image_name = f"{name}.png"
                    tmp_groundtruth_name = f"{name}.png"
                else:
                    tmp_image_name = f"{name}_{idx}.png"
                    tmp_groundtruth_name = f"{name}_{idx}.png"
                #print("the temp image name is " + tmp_image_name)
                image_path = os.path.join(save_path,os.path.join("images", tmp_image_name))
                groundtruth_path = os.path.join(save_path, os.path.join("groundtruths", tmp_groundtruth_name))
                mpimg.imsave(image_path, i)
                #print(gt.shape)
                cv2.imwrite(groundtruth_path, gt)
                
                idx += 1
                
                
                
                





dataset_path = os.path.join("data","train")
(train_x, train_y), (valid_x, valid_y) = load_data(dataset_path,split = 0.2)

# list of special figures
#special = []
special = [15,20,21,26,27,28,30,31,37,42,64,65,67,68,72,73,83,87,92,97]
#special = [11,15,16,20,21,22,24,28,32,51,52,54,55,58,59,65,68,72,77]

print("Train: ", len(train_x))
print("Valid: ", len(valid_x))
Directory_maker(os.path.join("new_data","train","images",""))
Directory_maker(os.path.join("new_data","train","groundtruths",""))
Directory_maker(os.path.join("new_data","valid","images",""))
Directory_maker(os.path.join("new_data","valid","groundtruths",""))

augment_data(train_x, train_y, os.path.join("new_data","train",""), special, flips = False, rotations = False)
augment_data(valid_x, valid_y,os.path.join("new_data","valid",""), [], flips = False, rotations = False)




##### MODEL ######





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





#
input_shape = (400, 400, 3)
model = build_unet(input_shape)
model.summary()





##### TRAINING ######





#Functions necessary for training

H = 400
W = 400

def shuffling(x,y):
    x, y = shuffle(x,y, random_state = 42)
    return x,y

def load_data(path):
    
    new_path_image = os.path.join(f"{path}",os.path.join("images","")+"*.png")
    new_path_groundtruths = os.path.join(f"{path}",os.path.join("groundtruths","")+"*.png")
    x = sorted(glob(new_path_image))
    y = sorted(glob(new_path_groundtruths))
    return x, y

## TF pipeline

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_groundtruth(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1) # makes it into a 3D image
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_groundtruth(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


# In[18]:


import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
#from model import build_unet
#from metrics import dice_loss, dice_coef, iou





H = 400
W = 400

## seeding - to be reproducable (time random is not)

np.random.seed(42)
tf.random.set_seed(42)

Directory_maker("files")

## Hyperparameters

batch_size = 2
learning_rate = 0.0001
epochs = 30

## files for saving information of simulation
model_path = os.path.join("files", "model.h5")
csv_path = os.path.join("files", "data.csv")

## Data sets

augmented_dataset_path = os.path.join("new_data")
train_path = os.path.join(augmented_dataset_path, "train")
valid_path = os.path.join(augmented_dataset_path, "valid")

train_x, train_y = load_data(train_path)
train_x, train_y = shuffling(train_x, train_y)
valid_x, valid_y = load_data(valid_path)

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")

train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

### Model training
model = build_unet((H, W, 3))
metrics = [dice_coef, iou, Recall(), Precision()]
model.compile(loss=dice_loss, optimizer=Adam(learning_rate), metrics=metrics)

### various callbacks for optimization

callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
]
model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=callbacks,
    shuffle=False)





import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def save_results(image, groundtruth, y_pred, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    ## groundtruth Dimension expansion
    groundtruth = np.expand_dims(groundtruth, axis=-1)
    groundtruth = np.concatenate([groundtruth, groundtruth, groundtruth], axis=-1)

    ## Predicted groundtruth Dimension expansion
    y_pred = np.expand_dims(y_pred, axis=-1)    ## (400, 400, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (400, 400, 3)
    y_pred = y_pred * 255 

    cat_images = np.concatenate([image, line, groundtruth, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

## Seeding for reproducability
np.random.seed(42)
tf.random.set_seed(42)

## Creating directory for results
Directory_maker("results")

##  Loading the model
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("files/model.h5")

##  Loading the model
test_x = sorted(glob(os.path.join("new_data", "valid", "images", "*")))
test_y = sorted(glob(os.path.join("new_data", "valid", "groundtruths", "*")))
print(f"Test: {len(test_x)} - {len(test_y)}")

##  Loading the model
SCORE = []
for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
    ##  Extracting the name
    temporary_string = os.path.join("a","a")
    forward_or_backslash = temporary_string[-2]
    name = x.split(forward_or_backslash)[-1].split(".")[0]

     ##  Reading the image
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    x = image/255.0
    x = np.expand_dims(x, axis=0)

    ##  Reading the groundtruth
    groundtruth = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = groundtruth/255.0
    y = y > 0.5
    y = y.astype(np.int32)

    ##  Predictions for validation
    y_pred = model.predict(x)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)

    ##  Saving the predictions for validation
    
    save_image_path = os.path.join("results",f"{name}.png")
    #save_image_path = f"results/{name}.png"
    save_results(image, groundtruth, y_pred, save_image_path)

    ##  Flattening the array
    y = y.flatten()
    y_pred = y_pred.flatten()

##  Metrics 
    acc_value = accuracy_score(y, y_pred)
    f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

score = [s[1:]for s in SCORE]
score = np.mean(score, axis=0)
print(f"Accuracy: {score[0]:0.5f}")
print(f"F1: {score[1]:0.5f}")
print(f"Jaccard: {score[2]:0.5f}")
print(f"Recall: {score[3]:0.5f}")
print(f"Precision: {score[4]:0.5f}")

df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
df.to_csv(os.path.join("files","score.csv"))




######## Useful functions

# Finds the nearest neighbor for a given point (i,j) in an N*N matrix (at the edges, we do not do periodic boundary)

def Nearest_neighbor(i,j,N):
    pm = [-1,0,1]
    list_of_indices = []
    for a in pm:
        for b in pm:
            if ((i+a >= 0) and (j+b >= 0)) and ((i+a <= N) and (j+b <= N)):
                if ([a,b] != [0,0]):
                    list_of_indices.append([i+a,j+b])
    return list_of_indices

# Turns a matrix into a ground truth (using rounding and nearest neighbor voting if = 0.5) 
def makebinary (Matrix):
    N = np.size(Matrix,0)
    for i in range (0,N):
        for j in range (0,N):
            if Matrix[i][j] == 0.5:
                #print("ambigous")
                Matrix[i][j] = 0
                Nearest_Neighbors = Nearest_neighbor(i,j,N-1)
                #print(Nearest_Neighbors)
                M = np.size(Nearest_Neighbors,0)
                for index in Nearest_neighbor(i,j,N-1):
                    Matrix[i][j] += (Matrix[index[0]][index[1]])/M
                if (Matrix[i][j] == 0.5):
                    #print("last chance")
                    Matrix[i][j] = np.random.rand()
            Matrix[i][j] = round(Matrix[i][j])
            #print(Matrix[i][j])
    return Matrix

# Cutting test images (actual images and ground truths) to appropriate sizes (608 -> 400)

def cut_image(image, size):    
    N = np.size(image,0)
        
    image_1 = image[0:size, 0:size,:]

    image_2 = image[0: size, N - size : N,:]

    image_3 = image[N - size : N , 0: size,:]

    image_4 = image[N - size : N, N - size : N,:]

    return image_1, image_2, image_3, image_4


def cut_images(image_list , size):
    
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    
    N = np.size(image_list[0],0)
    
    for image in image_list:
        
        image_1, image_2, image_3, image_4 = cut_image(image,size)
        list_1.append(image_1)
        list_2.append(image_2)
        list_3.append(image_3)
        list_4.append(image_4)

    return list_1, list_2, list_3, list_4

# Fusing ground truths back to orignial size 

def Fuse_image_output(image1,image2,image3,image4, size):
    output_matrix = np.zeros((608,608,1))
    output_matrix [0: 208, 0: 208,:] = image1[0:208, 0:208,:]
    output_matrix [0: 208, 208: 400,:] = (image1[0: 208, 208: 400,:]+image2[0: 208, 0: 192,:])/2
    output_matrix [0: 208, 400: 608,:] = image2[0:208, 192:400,:]
    
    output_matrix [208: 400, 0: 208,:] = (image1[208: 400, 0:208,:]+image3[0: 192, 0:208,:])/2
    output_matrix [208: 400, 208: 400,:] = (image1[208: 400, 208: 400,:]+image2[208: 400, 0: 192,:]+image3[0: 192, 208: 400,:]+image4[0: 192, 0: 192,:])/4
    output_matrix [208: 400, 400: 608,:] = image2[208: 400, 192:400,:]
    
    output_matrix [400: 608 , 0: 208,:] = image3[192:400,0:208,:]
    output_matrix [400: 608 , 208: 400,:] = (image3[192: 400, 208: 400,:]+image4[192: 400, 0: 192,:])/2
    output_matrix [400: 608 , 400: 608:] = image4[192:400,192:400,:]
    
    output_matrix[:,:,0] = makebinary(output_matrix[:,:,0])
       
    return output_matrix

#def Fuse_images_output(list_1,list_2,list_3,list_4, size):

    #N = np.size(list_1,0)
    #output_list = []
    #for i in range(0,N):
        #output_matrix = np.zeros(3,608,608)
        
        #output_matrix [:, 0: 208, 0: 208] = list_1[i][:, 0:208, 0:208]
        #output_matrix [:, 0: 208, 208: 400] = (list_1[i][:, 0: 208, 208: 400]+list_2[i][:, 0: 208, 0: 192])/2
        #output_matrix [:, 0: 208, 400: 608] = list_2[i][:, 0:208, 192:400]
        
        #output_matrix [:, 208: 399, 0: 207] = (list_1[i][:, 208: 399, 0:207] + list_3[i][:, 0: 191, 0:207])/2
        #output_matrix [:, 0: 207, 208: 399] = (list_1[i][:, 208: 399, 208: 399]+list_2[i][:, 208: 399, 0: 191]+list_3[i][:, 0: 191, 208: 399]+list_4[i][:, 0: 191, 0: 191])/4
        #output_matrix [:, 208: 399, 0: 207] = (list_2[i][:, 208: 399, 192:399] + list_4[i][:, 0: 191, 193:399])/2
        
        #output_matrix [:, 400: 607 , 0: 207] = list_3[i][:, 193:400,0:207]
        #output_matrix [:, 400: 607 , 208: 399] = (list_3[i][:, 192: 399, 208: 399]+list_4[i][:, 192: 399, 0: 191])/2
        #output_matrix [:, 400: 607 ,400: 607] = list_4[i][:, 193:400,193:400]
        
        #makebinary(output_matrix)
        
        #output_list.append(output_matrix)
       
    #return output_list





import os
import numpy as np
import cv2
#import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


## Seeding - again the use of 42 is for reproducability
np.random.seed(42)
tf.random.set_seed(42)

## Final Directory 
Directory_maker("test")

## loading model from training in the h5 file
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model( os.path.join("files","model.h5"))
    #model = tf.keras.models.load_model("files/model.h5")
    
## loading images for the test

test_x = glob(os.path.join("data/test_set_images", "*", "*.png"))
#test_x = glob("data/test_set_images/*/*.png")
print(f"Test: {len(test_x)}")





## loading images for the test
for x in tqdm(test_x):
    ## Getting the name of the images
    #dir_name = x.split("/")[-2]
    #print(dir_name)
    #name = dir_name + "_" + x.split("/")[-1].split(".")[0]
    temporary_string = os.path.join("a","a")
    forward_or_backslash = temporary_string[-2]
    name = x.split(forward_or_backslash)[-1].split(".")[0]
    #print(name)
    ##  obtaining the actual image data
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    x = image/255.0
    #print(x.shape)
    
    x1,x2,x3,x4 = cut_image(x,400)
    
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x3 = np.expand_dims(x3, axis=0)
    x4 = np.expand_dims(x4, axis=0)
    
    #print(x1.shape,x2.shape,x3.shape,x4.shape)
    
    ## model prediction
    groundtruth1 = model.predict(x1)[0]
    groundtruth1 = groundtruth1 > 0.5
    groundtruth1 = groundtruth1.astype(np.int32)
    groundtruth1 = groundtruth1
    
    groundtruth2 = model.predict(x2)[0]
    groundtruth2 = groundtruth2 > 0.5
    groundtruth2 = groundtruth2.astype(np.int32)
    groundtruth2 = groundtruth2
    
    groundtruth3 = model.predict(x3)[0]
    groundtruth3 = groundtruth3 > 0.5
    groundtruth3 = groundtruth3.astype(np.int32)
    groundtruth3 = groundtruth3
    
    groundtruth4 = model.predict(x4)[0]
    groundtruth4 = groundtruth4 > 0.5
    groundtruth4 = groundtruth4.astype(np.int32)
    groundtruth4 = groundtruth4

    groundtruth = Fuse_image_output(groundtruth1,groundtruth2,groundtruth3,groundtruth4,400) * 255
    groundtruth = np.concatenate([groundtruth, groundtruth, groundtruth], axis=-1)  ## (400, 400, 3)
    cat_images = np.concatenate([image, groundtruth], axis=1)
    
    #cv2.imwrite(os.path.join("test",f"{name}.png"), cat_images)
    cv2.imwrite(os.path.join("test",f"{name}.png"), groundtruth)


