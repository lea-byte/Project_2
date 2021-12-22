import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from Train import load_data
from Metrics import dice_loss, dice_coef, iou
from Data import Directory_maker


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
    name = x.split("/")[-1].split(".")[0]

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
    save_image_path = f"results/{name}.png"
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
df.to_csv("files/score.csv")
