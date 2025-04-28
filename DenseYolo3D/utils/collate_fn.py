import torch
import math

import torch
import math

def custom_collate_fn(batch):
    """
    Args:
        batch: List of (volume, label) tuples
            - volume: [1, D, H, W]
            - label: [7, ceil(D/16), ceil(H/128), ceil(W/128)]
    Returns:
        volumes: [batch_size, 1, max_D, max_H, max_W]
        labels: [batch_size, 7, max_ceil(D/16), max_ceil(H/128), max_ceil(W/128)]
        shapes: List of (D, H, W)
    """
    volumes, labels = zip(*batch)
    
    # Original shapes
    shapes = [(v.shape[1], v.shape[2], v.shape[3]) for v in volumes]
    
    # Max dimensions
    max_D = max(D for D, _, _ in shapes)
    max_H = max(H for _, H, _ in shapes)
    max_W = max(W for _, _, W in shapes)
    max_grid_d = max(math.ceil(D / 16) for D, _, _ in shapes)
    max_grid_h = max(math.ceil(H / 128) for _, H, _ in shapes)
    max_grid_w = max(math.ceil(W / 128) for _, _, W in shapes)

    # Pad
    padded_volumes = []
    padded_labels = []
    for vol, lab in zip(volumes, labels):
        D, H, W = vol.shape[1:]
        grid_d, grid_h, grid_w = lab.shape[1:]
        
        # Volume
        pad_d = max_D - D
        pad_h = max_H - H
        pad_w = max_W - W
        padded_vol = torch.nn.functional.pad(
            vol, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0
        )
        
        # Label
        pad_grid_d = max_grid_d - grid_d
        pad_grid_h = max_grid_h - grid_h
        pad_grid_w = max_grid_w - grid_w
        padded_lab = torch.nn.functional.pad(
            lab, (0, pad_grid_w, 0, pad_grid_h, 0, pad_grid_d), mode='constant', value=0
        )
        
        padded_volumes.append(padded_vol)
        padded_labels.append(padded_lab)
    
    return torch.stack(padded_volumes), torch.stack(padded_labels), shapes