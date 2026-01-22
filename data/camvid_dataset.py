import os
import torch
import numpy as np
import random as rd
from glob import glob
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# directories containing images and their masks
# for train, val and test
init_dir = './data/CamVid/'  # iitial directory of the CamVid dataset

# Palette CamVid
CAMVID_COLORS = {rgb:i for i, rgb in enumerate(torch.tensor([
                                                        [128,128,128],  # Sky
                                                        [128,0,0],      # Building
                                                        [192,192,128],  # Pole
                                                        [128,64,128],   # Road
                                                        [60,40,222],    # Pavement
                                                        [128,128,0],    # Tree
                                                        [192,128,128],  # SignSymbol
                                                        [64,64,128],    # Fence
                                                        [64,0,128],     # Car
                                                        [64,64,0],      # Pedestrian
                                                        [0,128,192]     # Bicyclist
                                                    ], dtype=torch.uint8))  # doit être uint8 pour matcher les labels
                }

class CamVidDataset(Dataset):
    def __init__(self,
                 file: str = "train"):
        """
        Params
        -------
        file : str
            File containing the list of data samples to be used.
            "val", "train" or "test". Defaults to "train".
        """

        assert file in ["train", "val", "test"], "file argument must be one of 'train', 'val', or 'test'"

        self.file = file
        self.imgs_dir = os.path.join(init_dir + self.file + '/images')
        self.mask_dir = os.path.join(init_dir + self.file + '/masks')

        # load the list of images names
        self.imgs_names = sorted(os.listdir(self.imgs_dir))

        # Image transformation
        self.imgs_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # mean and std of ImageNet
        ])
        self.masks_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256))
            ]
        )

    def __len__(self):
        return len(self.imgs_names)
    
    def __getitem__(self, id: int):
        img_name = self.imgs_names[id]
        mask_name = img_name.replace('.png', '_L.png')

        img_path = os.path.join(self.imgs_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load images and masks in RGB
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        img = self.imgs_transforms(img)
        mask = self.masks_transforms(mask)

        # Convert RGB values to indexes
        mask_array = np.array(mask)
        h, w, _ = mask_array.shape
        mask_index = np.zeros((h, w), dtype=np.int64)

        for rgb, index in CAMVID_COLORS.items():
            matches = np.all(mask_array == rgb.numpy(), axis=-1)
            mask_index[matches] = index
        
        mask = torch.from_numpy(mask_index).long()

        return img, mask  