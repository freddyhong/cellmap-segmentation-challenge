import torch
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from multi_separator import GreedySeparatorGrowing3D
from cellmap_data import CellMapImage
from pathlib import Path

# ===================================================================
#                      1. CONFIGURATION
# ===================================================================

# Tell the data loader to load a single 3D volume at a time.
# The `process_func` will manually load the second channel.
batch_size = 2
input_array_info = {"shape": (96, 96, 96), "scale": (8, 8, 8)}
target_array_info = {"shape": (96, 96, 96), "scale": (8, 8, 8)}

classes = ["lyso"] 

# CONNECTIVITY_3D = np.array([
#     [1, 0, 0], [0, 1, 0], [0, 0, 1],
#     [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
#     [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
#     [2, 0, 0], [0, 2, 0], [0, 0, 2],
#     [2, 2, 0], [2, -2, 0], [2, 0, 2], [2, 0, -2], [0, 2, 2], [0, 2, -2],
#     [2, 2, 2], [2, 2, -2], [2, -2, 2], [2, -2, -2]
# ], dtype=np.int32)

CONNECTIVITY_3D = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
], dtype=np.int32)

SIGMA = 0.5
AFFINITY_SCALE = 1.1
MEM_SCALE = 1.2
BIAS = 0.0# 0.25 for weak signal, -1.5 for strong signal      
# MIN_SIZE = 300      

def run_gsg_3d(lum: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """
    Runs the Greedy Separator Growing algorithm on a 3D volume,
    correctly preparing the cost array.
    """
    D, H, W = lum.shape
    # Smooth inputs
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    # combine = np.maximum(lum_s, mem_s)
    # g = 1 - combine
    g = 1 - lum_s
    eps = 1e-8
    
    v_cost = np.log((1 - g + eps) / (g + eps)).astype(np.float64)
    v_cost += BIAS

    edge_cost_volumes = []
    for offset in CONNECTIVITY_3D:
        s1, s2 = [slice(None)] * 3, [slice(None)] * 3
        pad_width = []
        for i in range(3):
            if offset[i] > 0:
                s1[i] = slice(None, -offset[i])
                s2[i] = slice(offset[i], None)
                pad_width.append((0, offset[i]))
            elif offset[i] < 0:
                s1[i] = slice(-offset[i], None)
                s2[i] = slice(None, offset[i])
                pad_width.append((abs(offset[i]), 0))
            else: # offset[i] == 0
                s1[i] = slice(None)
                s2[i] = slice(None)
                pad_width.append((0, 0))
        
        affinity = (lum_s[tuple(s1)] + lum_s[tuple(s2)]) / 2
        membrane = np.maximum(g[tuple(s1)], g[tuple(s2)])
        edge_cost = affinity * AFFINITY_SCALE - MEM_SCALE * membrane
        edge_cost_padded = np.pad(edge_cost, pad_width, 'constant').astype(np.float64)
        edge_cost_volumes.append(edge_cost_padded)

    # The rest of the function remains the same...
    all_costs_stacked = np.stack([v_cost] + edge_cost_volumes, axis=0)
    all_costs_flat = all_costs_stacked.ravel(order='F')
    shape = np.array([D, H, W], dtype=np.uint64)

    gsg = GreedySeparatorGrowing3D(shape, CONNECTIVITY_3D, all_costs_flat)
    gsg.run()
    labels = np.asarray(gsg.vertex_labels(), np.int32).reshape((D, H, W), order='F')
    return labels

def get_sibling_path(primary_path: str, new_class_name: str) -> str:
    return str(Path(primary_path).parent / new_class_name)

def process_func(batch_dict: dict) -> torch.Tensor:
    lum_batch = batch_dict["input"]
    B, _, Z, H, W = lum_batch.shape
    
    primary_path = batch_dict["lumen_path"]
    centers = batch_dict["centers"]

    lumen_class = "lyso"
    membrane_class = "lyso_mem"

    lumen_path = get_sibling_path(primary_path, lumen_class)
    membrane_path = get_sibling_path(primary_path, membrane_class)

    lum_image = CellMapImage(lumen_path, target_class=lumen_class, target_voxel_shape=(Z, H, W), target_scale=(8, 8, 8))
    mem_image = CellMapImage(membrane_path, target_class=membrane_class, target_voxel_shape=(Z, H, W), target_scale=(8, 8, 8)) 
    
    lum_patches = [lum_image[center] for center in centers]
    mem_patches = [mem_image[center] for center in centers]

    lum_batch = torch.stack(lum_patches)
    mem_batch = torch.stack(mem_patches)
    
    if lum_batch.ndim == 4: lum_batch = lum_batch.unsqueeze(1)
    if mem_batch.ndim == 4: mem_batch = mem_batch.unsqueeze(1)
        
    x = torch.cat([lum_batch, mem_batch], dim=1)

    print(f"Lumen Batch Stats -> Min: {lum_batch.min():.2f}, Max: {lum_batch.max():.2f}, Mean: {lum_batch.mean():.2f}")
    print(f"Membrane Batch Stats -> Min: {mem_batch.min():.2f}, Max: {mem_batch.max():.2f}, Mean: {mem_batch.mean():.2f}")

    # print("Successfully loaded both channels. Final input shape:", x.shape)

    out = torch.zeros((B, 1, Z, H, W), dtype=torch.float32, device=x.device)
    for b in range(B):
        # lum = torch.sigmoid(x[b,0]).cpu().numpy()
        # mem = torch.sigmoid(x[b,1]).cpu().numpy()
        lum = 1 / (1 + np.exp(-torch.clamp(x[b, 0].cpu(), -5, 5).numpy()))
        mem = 1 / (1 + np.exp(-torch.clamp(x[b, 1].cpu(), -5, 5).numpy()))
        
        labels = run_gsg_3d(np.clip(lum, 0, 1), np.clip(mem, 0, 1))
        # processed_mask = (labels > 0)
        
        # processed_mask = binary_fill_holes(labels > 0)

        # structure = np.ones((3,3,3))
        # processed_mask = binary_closing(processed_mask, structure=structure, iterations=2) 
        # processed_mask = remove_small_objects(labels, min_size=MIN_SIZE)
        out[b, 0] = torch.from_numpy(labels.astype(np.float32))
        
    return out

# This block allows your script to be run with `csc process`
if __name__ == "__main__":
    from cellmap_segmentation_challenge.process import process
    process()