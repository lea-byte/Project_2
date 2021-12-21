import os
import numpy as np
import cv2
import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from Metrics import dice_loss, dice_coef, iou

## Creating a directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

## Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("test")

## Loading the model
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

## Loading the dataset
    test_x = glob("data/test/*/*/*.dcm")
    print(f"Test: {len(test_x)}")

## Loop on data
    for x in tqdm(test_x):
        ## obtaining the name
        dir_name = x.split("/")[-3]
        name = dir_name + "_" + x.split("/")[-1].split(".")[0]

        ## image reading
        image = dicom.dcmread(x).pixel_array
        image = np.expand_dims(image, axis=-1)
        image = image/np.max(image) * 255.0
        x = image/255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)

        ## Predictions
        groundtruth = model.predict(x)[0]
        groundtruth = groundtruth > 0.5
        groundtruth = groundtruth.astype(np.int32)
        groundtruth = groundtruth * 255

        cat_images = np.concatenate([image, groundtruth], axis=1)
        cv2.imwrite(f"test/{name}.png", cat_images)
