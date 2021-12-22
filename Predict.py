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

## Seeding - again the use of 42 is for reproducability
np.random.seed(42)
tf.random.set_seed(42)

## Final Directory 
Directory_maker("test")

## loading model from training in the h5 file
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("files/model.h5")
    
## loading images for the test
test_x = glob("data/test_set_images/*/*.png")
print(f"Test: {len(test_x)}")

## loading images for the test
for x in tqdm(test_x):
    ## Getting the name of the images
    #dir_name = x.split("/")[-2]
    #print(dir_name)
    #name = dir_name + "_" + x.split("/")[-1].split(".")[0]
    name = x.split("/")[-1].split(".")[0]
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
    cv2.imwrite(f"test/{name}.png", cat_images)
    
