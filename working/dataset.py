import glob
import re

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from utils import load_image


class BrainRSNADataset(Dataset):
    def __init__(
        self, patient_path, paths, targets, transform=None, mri_type="FLAIR", is_train=True, ds_type="forgot", do_load=True
    ):
        
        self.patient_path = patient_path
        self.paths = paths   
        self.targets = targets
        self.type = mri_type

        self.transform = transform
        self.is_train = is_train
        self.folder = "train" if self.is_train else "test"
        self.do_load = do_load
        self.ds_type = ds_type  

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _id = self.paths[index]
        target = self.targets[index]
        _3d_images = self.load_images_3d(_id)
        _3d_images = torch.tensor(_3d_images).float()
        if self.is_train:
            return {"image": _3d_images, "target": target}
        else:
            return {"image": _3d_images, "target": target}


    def get_middle(self, files):
        image_numbers = []
        for image_path in files:
            match = re.search(r"Image-(\d+).png", image_path)
            if match:
                image_numbers.append(int(match.group(1)))

        # Sort the image numbers and find the median.
        image_numbers.sort()
        return image_numbers[len(image_numbers) // 2]

    def load_images_3d(
        self,
        case_id,
        num_imgs=config.NUM_IMAGES_3D,
        img_size=config.IMAGE_SIZE,
        rotate=0,
    ):
        case_id = str(case_id).zfill(5)

        path = f"./input/reduced_dataset/{case_id}/{self.type}/*.png"
        files = sorted(
            glob.glob(path),
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ],
        )

        middle = len(files) // 2
        #print('middle', middle)
        if len(files) <= 64:
            image_stack = [load_image(f) for f in files]
            #print('image-stack shape', len(image_stack))
 
        else:
            p1 = middle - 32 #max(0, middle - num_imgs2)
            p2 = middle + 32 #min(len(files), middle + num_imgs2)
            image_stack = [load_image(f) for f in files[p1:p2]]
            
            
        
        img3d = np.stack(image_stack).T
        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d, n_zero), axis=-1)

        if np.min(img3d) < np.max(img3d):
            img3d = img3d - np.min(img3d)
            img3d = img3d / np.max(img3d)

        return np.expand_dims(img3d, 0)
