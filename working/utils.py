import numpy as np
import pandas as pd
import config
import cv2
import os
import json

import torch

def load_image(path, size=(config.IMAGE_SIZE, config.IMAGE_SIZE)):
    image = cv2.imread(path, 0)
    if image is None:
        return np.zeros(config.IMAGE_SIZE)
    
    image = cv2.resize(image, size) / 255
    return image.astype('f')


def save_nested_dict_to_csv(metrics_dict, csv_path):
    """
    Flattens a nested dictionary of metrics and saves it to a CSV file.

    Parameters:
    - metrics_dict: Nested dictionary containing metric values.
    - csv_path: Path to save the CSV file.
    """
    rows = []
    
    for fold, datasets in metrics_dict.items():
        for dataset_type, metrics in datasets.items():
            for metric_name, values in metrics.items():
                for epoch, value in enumerate(values, start=1):
                    rows.append({
                        'fold': fold,
                        'dataset': dataset_type,
                        'epoch': epoch,
                        'metric': metric_name,
                        'value': value
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def update_metrics(metrics, fold, dataset_type, metric_name, value):
    if fold not in metrics:
        metrics[fold] = {}
    
    if dataset_type not in metrics[fold]:
        metrics[fold][dataset_type] = {}
    
    if metric_name not in metrics[fold][dataset_type]:
        metrics[fold][dataset_type][metric_name] = []

    metrics[fold][dataset_type][metric_name].append(value)

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


def save_metrics_to_json(metrics, model_name, encoder=TensorEncoder):
    base_dir = './plots'
    save_path = os.path.join(base_dir, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filename = f"metrics_{model_name}.json"
    full_path = os.path.join(save_path, filename)
    with open(full_path, "w") as file:
        json.dump(metrics, file, cls=encoder)
    
    print(f'Saving {filename}')
    return full_path










'''
def load_dicom_image(path, img_size=config.IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size))
    data = data - np.min(data)
    if np.min(data) < np.max(data):
        data = data / np.max(data)
    return data'''

'''
def crop_img(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    c1, c2 = False, False
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
    except:
        rmin, rmax = 0, img.shape[0]
        c1 = True

    try:
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        cmin, cmax = 0, img.shape[1]
        c2 = True
    bb = (rmin, rmax, cmin, cmax)
    
    if c1 and c2:
        return img[0:0, 0:0]
    else:
        return img[bb[0] : bb[1], bb[2] : bb[3]]


def extract_cropped_image_size(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    cropped_data = crop_img(data)
    resolution = cropped_data.shape[0]*cropped_data.shape[1]  
    return resolution'''