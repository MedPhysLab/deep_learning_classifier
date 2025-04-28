from multiprocessing import Pool, cpu_count
import logging
from scipy.ndimage import zoom
import numpy as np
import os
from joblib import Parallel,delayed
import fcntl
import time
import gc
from utils import dataset_perpaziente as dp

def resize_vlm(arr, factor):
    #new_shape = np.array((length_side,)*2)
    #print(new_shape)

    #zoom_factors12 = np.array(arr.shape[1:])/new_shape
    #print(zoom_factors12)
    resize_arr =  zoom(arr, (1/factor,1/factor,1/factor))
    

    return resize_arr



def normalize_array(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val) 
    return normalized_tensor





def save_image_with_lock(output_path, pixel_array):
    with open(output_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        #image.save(f, format='TIFF', compression='tiff_deflate')
        np.save(f, pixel_array)
        fcntl.flock(f, fcntl.LOCK_UN)


def save_images(root_output_folder, data_generator, split_type="train", factor=2, classe=None):
    #classe_dati= dp.DataProcessor("blablabla")
    #decompress_and_flip_image = classe_dati.decompress_and_flip_image
    def process_image(row):
        pixel_array, row = row
        patient_name = row['PatientID']
        class_id = row['Malata']
        view = row["View"]
        image_name = str(patient_name) + "-" + str(view).upper()
        pixel_array = classe.decompress_and_flip_image(row)
        
        # Normalize the pixel array
        normalized_array = normalize_array(pixel_array)
        normalized_array = (normalized_array).astype(np.uint8)

        # Resize the pixel array
        resized_array = resize_vlm(normalized_array, 3)

        # Create output directory if it doesn't exist
        if split_type == "train":
            output_dir = os.path.join(root_output_folder, "train", str(class_id))
        elif split_type == "val":
            output_dir = os.path.join(root_output_folder, "val", str(class_id))
        os.makedirs(output_dir, exist_ok=True)

        # Save the tensor
        output_path = os.path.join(output_dir, image_name + ".npy")
        save_image_with_lock(output_path, resized_array)

    Parallel(n_jobs=4)(delayed(process_image)(row) for row in data_generator)
    return 0