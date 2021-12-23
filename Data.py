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
