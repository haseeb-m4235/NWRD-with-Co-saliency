from torch.utils.data import  Dataset
import numpy as np
import glob
import cv2
import random
from PIL import Image



np.set_printoptions(threshold=np.inf)

class BinaryDataset(Dataset):
    def __init__(self, image_dict, transform=None):
        self.image_dict = image_dict
        self.image_paths = list(self.image_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_dict)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.image_dict[image_path]

        # Load image using PIL
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label

def get_label(patches_root):
    rust = []
    non_rust = []
    masks_paths = glob.glob(patches_root+'\\masks\\*')
    for mask_path in masks_paths:
        mask_patches = glob.glob(mask_path+'\\*')
        #print(mask_patches)
        image_name = mask_path.split("\\")[-1]
        for mask_patch in mask_patches:
            patch_name =  mask_patch.split("\\")[-1].split(".")[0]
            mask = cv2.imread(mask_patch, 0) 
            condition = (mask > 200)
            count = np.sum(condition)
            if count > 500:
                rust.append(mask_patch.replace("masks", "images"))
                #print(image_name, patch_name, "rust")
            else:
                non_rust.append(mask_patch.replace("masks", "images"))
                #print(image_name, patch_name, "non-rust")
    data = [(path, 1) for path in rust[:2500]] + [(path, 0) for path in non_rust[:2500]]
    data = dict(data)
    print("Labels loaded")
    return data




