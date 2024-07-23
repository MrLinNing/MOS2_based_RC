#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import cv2
import glob
from typing import List
from pandas import read_excel


def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "/home/rram/RC_Computing_YPJJ/eye_seg/LadderNet/DRIVE/training/images/"
groundTruth_imgs_train = "/home/rram/RC_Computing_YPJJ/eye_seg/LadderNet/DRIVE/training/1st_manual/"
borderMasks_imgs_train = "/home/rram/RC_Computing_YPJJ/eye_seg/LadderNet/DRIVE/training/mask/"
#test
original_imgs_test = "/home/rram/RC_Computing_YPJJ/eye_seg/LadderNet/DRIVE/test/images/"
groundTruth_imgs_test = "/home/rram/RC_Computing_YPJJ/eye_seg/LadderNet/DRIVE/test/1st_manual/"
borderMasks_imgs_test = "/home/rram/RC_Computing_YPJJ/eye_seg/LadderNet/DRIVE/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_datasets_training_testing/"

if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)


# Load and normalize quantization values
def load_normalized_quantization_values(path: str) -> List[float]:
    quantization_values = read_excel(path, usecols=[1], skiprows=1, nrows=64, header=None).values.flatten()
    print(f"quantization_values is {quantization_values}")
    return normalize_quantization_values(quantization_values)

# Normalize quantization values to [0, 1] range
def normalize_quantization_values(quantization_values):
    min_value = np.min(quantization_values)
    max_value = np.max(quantization_values)
    normalized_values = (quantization_values - min_value) / (max_value - min_value)
    return normalized_values

# Find the nearest quantization value
def find_nearest_quantization_value(value, quantization_values):
    idx = np.argmin(np.abs(quantization_values - value))
    return quantization_values[idx]

def process_image(img, normalized_quantization_values):
    # Split the image into R, G, B channels
    r, g, b = img.split()

    # Convert channels to NumPy arrays
    r_array = np.array(r, dtype=np.float32)
    g_array = np.array(g, dtype=np.float32)
    b_array = np.array(b, dtype=np.float32)

    # Normalize using cv2.normalize
    r_array = cv2.normalize(r_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    g_array = cv2.normalize(g_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    b_array = cv2.normalize(b_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    quantized_r_array = np.vectorize(find_nearest_quantization_value, excluded=[1])(r_array, normalized_quantization_values)
    quantized_g_array = np.vectorize(find_nearest_quantization_value, excluded=[1])(g_array, normalized_quantization_values)
    quantized_b_array = np.vectorize(find_nearest_quantization_value, excluded=[1])(b_array, normalized_quantization_values)

    # Combine quantized R, G, B arrays into a single image
    quantized_image = np.stack((quantized_r_array, quantized_g_array, quantized_b_array), axis=-1)

    # Convert the combined array to uint8 format
    quantized_image_uint8 = (quantized_image * 255).astype(np.uint8)

    return quantized_image_uint8

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,normalized_quantization_values,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])

            quantized_image = process_image(img, normalized_quantization_values)
            imgs[i] = np.asarray(quantized_image)


            # imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print("specify if train or test!!")
                exit()
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks


quantization_values_path = './Slope_of_64.xlsx'
normalized_quantization_values = load_normalized_quantization_values(quantization_values_path)

imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train, borderMasks_imgs_train, normalized_quantization_values, "train")

imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test, borderMasks_imgs_test, normalized_quantization_values, "test")

#getting the training datasets
# imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train_quant.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
# imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test_quant.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
