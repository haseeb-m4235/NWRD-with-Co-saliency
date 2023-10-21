from functools import total_ordering
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import math



patch_size = (128, 128)

def get_patch_position(img_shape, patch_size, patch_no):
    # Calculate the number of patches that fit horizontally and vertically
    num_patches_horizontally = img_shape[1] // patch_size[1]
    num_patches_vertically = img_shape[0] // patch_size[0]
    
    # Ensure the given patch number is valid
    if patch_no > num_patches_horizontally * num_patches_vertically:
        raise ValueError("Invalid patch number for the given image size and patch size.")
    
    # Calculate row and column based on the patch number
    row = (patch_no - 1) // num_patches_horizontally + 1
    col = (patch_no - 1) % num_patches_horizontally + 1
    
    return row, col

def replace_with_patch(matrix, patch, row, col, patch_size):
    start_y = (row - 1) * patch_size[0]
    start_x = (col - 1) * patch_size[1]
    matrix[start_y:start_y+patch_size[0], start_x:start_x+patch_size[1]] = patch
    return matrix






patch_size = (128, 128)
masks = glob.glob('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\rust128\\results\\testing with best CoCA\\*')
#print(masks)
for mask in masks:
    patches = glob.glob(mask+"\\*")
    imgNo = mask.split('\\')[-1]
    img = cv2.imread("C:\\Users\\hasee\\Desktop\\Semester Internship\\FineLine\\NWRD\\test\\images\\"+imgNo[4:]+".jpg", 0)
    print("image shape:", img.shape)
    pred = np.zeros(img.shape)
    for patch_path in patches:
        patchNo = patch_path.split('\\')[-1][:-4]
        row, col = get_patch_position(img.shape, patch_size, int(patchNo))
        patch = cv2.imread(patch_path, 0)
        print("pathcNo:",patchNo)
        print(f"Row: {row}, Column: {col}")  
        pred = replace_with_patch(pred, patch, row, col, patch_size)
    save_dir = f"C:\\Users\\hasee\\Desktop\\Semester Internship\\FineLine\\first_results\\{imgNo}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, f"{imgNo}.png"), pred)




