import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from Data import Directory_maker

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

