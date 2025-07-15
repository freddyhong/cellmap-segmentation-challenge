import torch
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from multi_separator import GreedySeparatorGrowing3D
from cellmap_data import CellMapImage

# ===================================================================
#                      1. CONFIGURATION
# ===================================================================

# Tell the data loader to load a single 3D volume at a time.
# The `process_func` will manually load the second channel.
batch_size = 8
input_array_info = {"shape": (96, 96, 96), "scale": (8, 8, 8)}
target_array_info = {"shape": (96, 96, 96), "scale": (8, 8, 8)}

classes = ["mito"] 

CONNECTIVITY_3D = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [2, 0, 0], [0, 2, 0], [0, 0, 2],
    [2, 2, 0], [2, -2, 0], [2, 0, 2], [2, 0, -2], [0, 2, 2], [0, 2, -2],
    [2, 2, 2], [2, 2, -2], [2, -2, 2], [2, -2, -2]
], dtype=np.int32)

SIGMA = 0.2
AFFINITY_SCALE = 1.1 
MEM_SCALE = 1.3     
BIAS = 0.0          
MIN_SIZE = 100      

def run_gsg_3d(lum: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """
    Runs the Greedy Separator Growing algorithm on a 3D volume,
    correctly preparing the cost array.
    """
    D, H, W = lum.shape
    # Smooth inputs
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    g = 1 - lum_s
    eps = 1e-8
    
    # --- START OF THE FIX ---

    # 1. Ensure v_cost is explicitly float64 from the start.
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
        membrane = np.maximum(mem_s[tuple(s1)], mem_s[tuple(s2)])
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


def process_func(batch_dict: dict) -> torch.Tensor:
    """
    Receives a batch dictionary, including 'lumen_path' and 'centers',
    and correctly processes it.
    """
    lum_batch = batch_dict["input"]
    B, _, Z, H, W = lum_batch.shape

    # --- START OF THE FIX ---
    # Use the 'lumen_path' and 'centers' keys we just added in process.py
    lumen_path = batch_dict["lumen_path"]
    centers = batch_dict["centers"]
    # --- END OF THE FIX ---

    # Manually construct the path to the corresponding membrane data
    membrane_path = lumen_path.replace("mito", "mito_mem")
    target_scale = (8, 8, 8)

    # Create a CellMapImage object for the membrane data
    mem_image = CellMapImage(
        membrane_path,
        target_class = "mito",
        target_voxel_shape=(Z, H, W),
        target_scale=target_scale
    )

    # --- START OF THE FIX ---
    # Load the corresponding membrane data for each item by its center coordinate
    mem_patches = [mem_image[center] for center in centers]
    # --- END OF THE FIX ---
    mem_batch = torch.stack(mem_patches).to(lum_batch.device)
    mem_batch = mem_batch.unsqueeze(1)

    x = torch.cat([lum_batch, mem_batch], dim=1)

    print("Successfully loaded both channels. Final input shape:", x.shape)

    out = torch.zeros((B, 1, Z, H, W), dtype=torch.float32, device=x.device)
    for b in range(B):
        lum = 1 / (1 + np.exp(-torch.clamp(x[b, 0].cpu(), -5, 5).numpy()))
        mem = 1 / (1 + np.exp(-torch.clamp(x[b, 1].cpu(), -5, 5).numpy()))
        
        labels = run_gsg_3d(np.clip(lum, 0, 1), np.clip(mem, 0, 1))
        
        processed_mask = binary_closing(binary_fill_holes(labels > 0), np.ones((3,3,3)))
        processed_mask = remove_small_objects(processed_mask, min_size=MIN_SIZE)
        out[b, 0] = torch.from_numpy(processed_mask.astype(np.float32))
        
    return out

# This block allows your script to be run with `csc process`
if __name__ == "__main__":
    from cellmap_segmentation_challenge.process import process_cli
    process_cli()