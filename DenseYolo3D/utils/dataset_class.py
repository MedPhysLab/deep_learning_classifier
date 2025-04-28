import torch
import numpy as np
import os
import pickle
import glob2
from torch.utils.data import DataLoader, Dataset
from functools import reduce
import torch.nn.functional as F
from joblib import Parallel, delayed
import hashlib
from scipy.ndimage import label, find_objects
import tifffile as tiff
import json


def process_volume(volume_file, grid_size):
    """
    Process a single volume file and returns the normalized padded volume and cell info.

    Args:
        volume_file (str): Path to the volume file to process.
        grid_size (tuple): The grid size to use for padding and calculating cell info.

    Returns:
        tuple: A tuple containing the normalized padded volume and the number of cells.
    """
    # Load the volume data
    volume = torch.tensor(tiff.imread(volume_file), dtype=torch.float32)
    
    # Extract largest connected component
    largest_mask, labeled_tensor = extract_largest_component(volume)
    volume = volume * largest_mask
    
    # Pad the volume
    padded_volume = pad_image(volume, grid_size)
    
    # Normalize the padded volume
    normalized_padded_volume = (padded_volume - padded_volume.min()) / (padded_volume.max() - padded_volume.min() + 1e-8)
    
    # Calculate the number of cells
    num_cells = tuple(max(1, v // g) for v, g in zip(padded_volume.shape, grid_size))
    
    return normalized_padded_volume, num_cells




def extract_largest_component(tensor):
    # Step 1: Threshold to create a binary mask (non-zero -> 1, zero -> 0)
    binary_mask = tensor > 0  # Create a binary mask for non-zero pixels

    # Step 2: Label the connected components
    labeled_tensor, num_components = label(binary_mask)

    # Step 3: Calculate the sizes of each component
    component_sizes = [np.sum(labeled_tensor == i) for i in range(1, num_components + 1)]
    
    # Step 4: Find the index of the largest connected component
    largest_component_index = np.argmax(component_sizes) + 1  # +1 because labels start from 1

    # Step 5: Extract the largest component (mask it out)
    largest_component_mask = labeled_tensor == largest_component_index

    # Optionally, you can return the largest component's mask, or apply it to the original tensor.
    return largest_component_mask, labeled_tensor

# Example 3D tensor (for testing)
#tensor = np.random.randint(0, 2, size=(10, 10, 10))  # Binary tensor with some non-zero pixels
#largest_mask, labeled_tensor = extract_largest_component(tensor)

#print("Largest connected component mask:\n", largest_mask)


def pad_image(image_tensor,patch_size, padding_value=0):
    """
    Pad the image and mask tensors to avoid going out of bounds during patch extraction.
    Pads in all dimensions (depth, height, width) based on the patch size.
    """
    D, H, W = image_tensor.shape  # Get the image dimensions

    pad_depth = max(0, patch_size[0] - D % patch_size[0]) if D % patch_size[0] != 0 else 0
    pad_height = max(0, patch_size[1] - H % patch_size[1]) if H % patch_size[1] != 0 else 0
    pad_width = max(0, patch_size[2] - W % patch_size[2]) if W % patch_size[2] != 0 else 0

    # Pad the image tensor and mask tensor with zeros (or another value if desired)
    padded_image_tensor = F.pad(image_tensor,  (0, pad_width, 0, pad_height, 0, pad_depth), "constant", padding_value)


    return padded_image_tensor

def return_cell(volume, cell_x, cell_y, cell_z,grid_size):
    volume_shape = volume.shape
    #print(f"Volume shape: {volume_shape}")
    # Calculate the start and end indices of the cell
    start_x, start_y, start_z = cell_x * grid_size[2], cell_y * grid_size[1], cell_z * grid_size[0]
    end_x = min(start_x + grid_size[2], volume_shape[2])
    end_y = min(start_y + grid_size[1], volume_shape[1])
    end_z = min(start_z + grid_size[0], volume_shape[0])
        
    # Extract the corresponding cell from the volume
    cell = volume[start_z:end_z, start_y:end_y, start_x:end_x]

    return cell

def get_3d_volume_vertices(centroid, h, w, d):
    """
    Calculate the coordinates of the vertices of a 3D volume given the centroid and dimensions.

    Parameters:
    centroid (tuple): The (x, y, z) coordinates of the centroid.
    h (float): The height of the volume.
    w (float): The width of the volume.
    d (float): The depth of the volume.

    Returns:
    list: A list of tuples representing the coordinates of the vertices.
    """
    x, y, z = centroid

    vertices = [
        (x - w / 2, y - h / 2, z - d / 2),
        (x + w / 2, y - h / 2, z - d / 2),
        (x - w / 2, y + h / 2, z - d / 2),
        (x + w / 2, y + h / 2, z - d / 2),
        (x - w / 2, y - h / 2, z + d / 2),
        (x + w / 2, y - h / 2, z + d / 2),
        (x - w / 2, y + h / 2, z + d / 2),
        (x + w / 2, y + h / 2, z + d / 2)
    ]

    return vertices

def window_leveling_volume(volume, rows):
    volume_windowed = volume.clone()
    min_vals, max_vals = [], []
    for _, row in rows.iterrows():
        if row["Malata"] == 1:
            x, y, z = row["X"]/3, row["Y"]/3, row["Slice"]/3
            w, h = row["Width"]/3, row["Height"]/3
            d = np.sqrt(w * h)
            z_inf = max(int(z - d/2), 0)
            z_sup = min(int(z + d/2), volume.shape[0])
            y_inf = max(int(y - h/2), 0)
            y_sup = min(int(y + h/2), volume.shape[1])
            x_inf = max(int(x - w/2), 0)
            x_sup = min(int(x + w/2), volume.shape[2])
            region = volume[z_inf:z_sup, y_inf:y_sup, x_inf:x_sup]
            if region.numel() > 0:
                min_vals.append(torch.min(region))
                max_vals.append(torch.max(region))
    if min_vals and max_vals:
        min_val = torch.min(torch.stack(min_vals))
        max_val = torch.max(torch.stack(max_vals))
        volume_windowed[volume_windowed > max_val] = 0
        volume_windowed[volume_windowed < min_val] = 0
    return volume_windowed

def window_leveling(cell, volume, rows, cell_x, cell_y, cell_z, grid_size):
    cell_dimensions = cell.shape  # [16, 128, 128] -> [z, y, x]
    volume_dimensions = volume.shape  # e.g., [83, 256, 256] -> [z, y, x]
    cell_windowed = cell.clone()

    # Cell’s bounds in volume space (grid_size = [z, x, y])
    cell_start_z = cell_z * grid_size[0]
    cell_end_z = cell_start_z + grid_size[0]
    cell_start_x = cell_x * grid_size[1]
    cell_end_x = cell_start_x + grid_size[1]
    cell_start_y = cell_y * grid_size[2]
    cell_end_y = cell_start_y + grid_size[2]

    intersecting_min_vals = []
    intersecting_max_vals = []

    for _, row in rows.iterrows():
        if row["Malata"] == 1:
            try:
                centroid = (float(row["X"]), float(row["Y"]), float(row["Slice"]))
                h = float(row["Height"])
                w = float(row["Width"])
                if not all(torch.isfinite(torch.tensor([h, w]))) or h <= 0 or w <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            depth = torch.sqrt(torch.tensor(h * w, dtype=torch.float32))
            if not torch.isfinite(depth):
                continue

            # Total ROI bounds in volume space
            z_inf_vol = max(int(centroid[2] - depth/2), 0)
            z_sup_vol = min(int(centroid[2] + depth/2), volume_dimensions[0])
            y_inf_vol = max(int(centroid[1] - h/2), 0)
            y_sup_vol = min(int(centroid[1] + h/2), volume_dimensions[1])
            x_inf_vol = max(int(centroid[0] - w/2), 0)
            x_sup_vol = min(int(centroid[0] + w/2), volume_dimensions[2])

            # Check intersection with cell
            intersects = (z_sup_vol > cell_start_z and z_inf_vol < cell_end_z and
                          y_sup_vol > cell_start_y and y_inf_vol < cell_end_y and
                          x_sup_vol > cell_start_x and x_inf_vol < cell_end_x)

            if intersects:
                # Get min and max from full ROI in volume
                selected_region_vol = volume[z_inf_vol:z_sup_vol, y_inf_vol:y_sup_vol, x_inf_vol:x_sup_vol]
                if selected_region_vol.numel() > 0:
                    intersecting_min_vals.append(torch.min(selected_region_vol))
                    intersecting_max_vals.append(torch.max(selected_region_vol))

    if intersecting_min_vals and intersecting_max_vals:
        min_val = torch.min(torch.stack(intersecting_min_vals))  # Min of mins from intersecting ROIs
        max_val = torch.max(torch.stack(intersecting_max_vals))  # Max of maxes from intersecting ROIs
        cell_windowed[cell_windowed > max_val] = 0
        cell_windowed[cell_windowed < min_val] = 0

    return cell_windowed





def is_intersection_above_threshold(intersection_volume, box_volume, threshold_percentage):
    threshold_volume = box_volume * threshold_percentage
    return intersection_volume >= threshold_volume



class LesionDataset3D(Dataset):
    def __init__(self, volume_dir, df, grid_size=(16  ,128,128) , anchor_size=(16  ,128,128) ,split_type="train"):
        self.volume_dir = volume_dir
        self.df = df
        self.grid_size = grid_size
        self.anchor_size = anchor_size
        self.tot_num_cells=0
        self.info_cells_pervolume={}
        self.tot_pos_cells=0
        self.pos_weights=0
        self.dizio={}
        self.split_type= split_type
        self.volumes= {}

       
        # List all volume files
        self.volume_files = glob2.glob(f"{volume_dir}/**/*.tiff")

        # cache directory
        cache_dir = os.path.join("/home/pcaterino/DenseYolo3D/model_data/", "cache")
        os.makedirs(cache_dir,exist_ok= True)

                
        # Cache file paths
        cache_key = self._get_cache_key()
        volumes_cache_file = os.path.join(cache_dir, f"volumes_{cache_key}.pt")
        info_cache_file = os.path.join(cache_dir, f"info_cells_{cache_key}.json")

        # Try loading from cache
        if os.path.exists(volumes_cache_file) and os.path.exists(info_cache_file):
            print(f"Loading cached volumes and info from {cache_dir}")
            try:
                self.volumes = torch.load(volumes_cache_file)
                with open(info_cache_file, 'r') as f:
                    self.info_cells_pervolume = {k: tuple(v) for k, v in json.load(f).items()}
                print(f"Loaded {len(self.volumes)} volumes from cache")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing volumes...")
                self._compute_volumes_and_info(volumes_cache_file, info_cache_file)
        else:
            print("Cache not found. Computing volumes...")
            self._compute_volumes_and_info(volumes_cache_file, info_cache_file)



        # Use joblib for parallel processing
        #results = Parallel(n_jobs=-1)(delayed(process_volume)(vf, grid_size) for vf in self.volume_files)

        #for volume_file, (normalized_padded_volume, num_cells) in zip(self.volume_files, results):
            #self.volumes[volume_file] = normalized_padded_volume
            #self.info_cells_pervolume[volume_file] = num_cells
            #print(self.volumes.keys())
        
         # Load or compute dictionary and volumes
        dict_path = os.path.join("/home/pcaterino/DenseYolo3D/model_data", f"saved_dictionary_{split_type}.pkl")
        if os.path.exists(dict_path):
            self.load_dictionary("/home/pcaterino/DenseYolo3D/model_data")
            #print(self.volumes.keys())
        else:    
            self.set_dictionary(factor=3, path="/home/pcaterino/DenseYolo3D/model_data")

        
    def __len__(self):
        return len(self.volume_files)
    
    def __getitem__(self, idx):
        volume_file = self.volume_files[idx]
        volume = self.volumes[volume_file].float()
        
        # Compute number of grid cells
        volume_shape = volume.shape  # [D, H, W]
        num_cells = tuple(max(1, v // g) for v, g in zip(volume_shape, self.grid_size))  # [D/16, H/128, W/128]
        
        # Initialize label tensor
        label = torch.zeros((7, *num_cells), dtype=torch.float32)  # [7, D/16, H/128, W/128]
        
        # Load annotations
        file_name = os.path.basename(volume_file).replace('.tiff', '.dcm')
        rows = self.df[self.df["name_file"] == file_name]
        
        for tumor_id, row in rows.iterrows():
            if row["Malata"] == 1:
                x, y, z = row["X"]/3, row["Y"]/3, row["Slice"]/3
                cell_x = int(x // self.grid_size[2])
                cell_y = int(y // self.grid_size[1])
                cell_z = int(z // self.grid_size[0])
                key = (volume_file, cell_x, cell_y, cell_z)
                
                if key in self.dizio:
                    confidence_gt, bbox_gt = self.dizio[key]
                    if 0 <= cell_z < num_cells[0] and 0 <= cell_y < num_cells[1] and 0 <= cell_x < num_cells[2]:
                        label[0, cell_z, cell_y, cell_x] = confidence_gt
                        label[1:7, cell_z, cell_y, cell_x] = bbox_gt
        volume = volume.unsqueeze(0)  # [1, D, H, W]
        #print(volume.shape)
        #print(label.shape)
        

        
        return volume, label
        
    

    def set_dictionary(self, factor=3, path=""):
        positive_cells = set()
        cell_to_tumor = {}
        
        for volume_file in self.volume_files:
            volume_shape = self.volumes[volume_file].shape
            num_cells = tuple(max(1, v // g) for v, g in zip(volume_shape, self.grid_size))
            
            name_file = os.path.basename(volume_file).replace('.tiff', '.dcm')
            bboxes_df = self.df[self.df["name_file"] == name_file]
            
            for tumor_id, row in bboxes_df.iterrows():
                if row["Malata"] == 1:
                    x, y, z, w, h = (
                        row["X"]/factor, row["Y"]/factor, row["Slice"]/factor,
                        row["Width"]/factor, row["Height"]/factor
                    )
                    d = np.sqrt(w * h)
                    cell_x = int(x // self.grid_size[2])
                    cell_y = int(y // self.grid_size[1])
                    cell_z = int(z // self.grid_size[0])
                    
                    if 0 <= cell_z < num_cells[0] and 0 <= cell_y < num_cells[1] and 0 <= cell_x < num_cells[2]:
                        start_x = cell_x * self.grid_size[2]
                        start_y = cell_y * self.grid_size[1]
                        start_z = cell_z * self.grid_size[0]
                        offset_x = (x - start_x) / self.grid_size[2]
                        offset_y = (y - start_y) / self.grid_size[1]
                        offset_z = (z - start_z) / self.grid_size[0]
                        scale_w = w / self.anchor_size[2]
                        scale_h = h / self.anchor_size[1]
                        scale_d = d / self.anchor_size[0]
                        
                        confidence_gt = torch.ones(1, dtype=torch.float32)
                        bbox_gt = torch.tensor(
                            [offset_x, offset_y, offset_z, scale_w, scale_h, scale_d],
                            dtype=torch.float32
                        )
                        
                        cell_key = (volume_file, cell_x, cell_y, cell_z)
                        if cell_key not in cell_to_tumor:
                            cell_to_tumor[cell_key] = (confidence_gt, bbox_gt)
                            positive_cells.add(cell_key)
        
        self.dizio = cell_to_tumor
        num_positive = len(positive_cells)
        num_negative = sum(
            reduce(lambda x, y: x * y, tuple(max(1, v // g) for v, g in zip(self.volumes[f].shape, self.grid_size)))
            for f in self.volume_files
        ) - num_positive
        self.pos_weights = num_negative / num_positive if num_positive > 0 else 1.0
        
        with open(os.path.join(path, f'saved_dictionary_{self.split_type}.pkl'), 'wb') as f:
            pickle.dump(self.dizio, f)
        with open(os.path.join(path, f'pos_weights_{self.split_type}.txt'), 'w') as f:
            f.write(str(self.pos_weights))

    def load_dictionary(self, path_file):
        with open(os.path.join(path_file, f'saved_dictionary_{self.split_type}.pkl'), 'rb') as f:
            self.dizio = pickle.load(f)
            self.pos_weights = float(open(os.path.join(path_file, f'pos_weights_{self.split_type}.txt'), 'r').read())


    def _get_cache_key(self):
        """
        Generate a unique key based on volume_dir, grid_size, anchor_size, and file list.
        """
        file_list = sorted(self.volume_files)
        hash_input = f"{self.volume_dir}{self.grid_size}{self.anchor_size}{file_list}"
        return hashlib.md5(hash_input.encode()).hexdigest()


    def _compute_volumes_and_info(self, volumes_cache_file, info_cache_file):
        """
        Compute volumes and info_cells_pervolume, then save to cache.
        """
        # Use joblib for parallel processing
        results = Parallel(n_jobs=-1)(delayed(process_volume)(vf, self.grid_size) for vf in self.volume_files)

        for volume_file, (normalized_padded_volume, num_cells) in zip(self.volume_files, results):
            self.volumes[volume_file] = normalized_padded_volume
            self.info_cells_pervolume[volume_file] = num_cells
        # Save to cache
        print(f"Saving volumes and info to {volumes_cache_file} and {info_cache_file}")
        try:
            torch.save(self.volumes, volumes_cache_file)
            with open(info_cache_file, 'w') as f:
                json.dump({k: list(v) for k, v in self.info_cells_pervolume.items()}, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
