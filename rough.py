import os
import glob
import random
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils import load_image

class RSNAdataset(Dataset):
    def __init__(self, patient_path, paths, targets, n_slices, img_size, type, transform=None):
        self.patient_path = patient_path
        self.paths = paths
        self.targets = targets
        self.n_slices = n_slices
        self.img_size = img_size
        self.transform = transform
        self.type = type
          
    def __len__(self):
        return len(self.paths)
    
    def padding(self, paths): 
        images = [load_image(path) for path in paths]
        org_size = len(images)
        
        # Apply transformations if provided
        if self.transform:
            seed = random.randint(0, 99999)
            for i in range(len(images)):
                random.seed(seed)
                images[i] = self.transform(image=images[i])["image"]

        dup_len = self.n_slices - len(images)
        if org_size == 0:
            dup = np.zeros((1, self.img_size, self.img_size))
        else:
            dup = images[-1]

        # Ensure the duplicate image has the correct shape
        if len(dup.shape) == 2:
            dup = dup.reshape(1, *dup.shape)

        images.extend([dup] * dup_len)
        
        images = torch.tensor(images, dtype=torch.float32)
        return images, org_size
    
    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = os.path.join(self.patient_path, f'{str(_id).zfill(5)}/')

        data = []
        org = []
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, self.type, "*")), 
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
        image, org_size = self.padding(t_paths)

        data.append(image)
        org.append(org_size)
            
        data = torch.stack(data).transpose(0,1)
        y = torch.tensor(self.targets[index])
        
        return {"X": data.float(), "y": y, 'org': org}