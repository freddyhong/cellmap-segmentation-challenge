from cellmap_segmentation_challenge.utils import load_safe_config
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from multi_separator import GreedySeparatorShrinking, GreedySeparatorGrowing3D, GreedySeparatorGrowing2D
import time
import matplotlib.pyplot as plt

# Load training config
config_path = "train_3D.py"
config = load_safe_config(config_path)

# Required by process.py
batch_size = getattr(config, "batch_size", 8)
input_array_info = {"shape": (2, 32, 32, 32), "scale": (8, 8, 8)}
target_array_info = input_array_info

# lumen channel first, membrane channel second
label_groups = {
    "ld": ["ld_lum", "ld_mem"],
}

classes = ["ld"]

# Tunable parameters
SIGMA = 0.4
LUM_SCALE = 1.1  # Attraction force for lumen (object)
MEM_SCALE = 0.3  # Repulsive force for membranes (separator)
BG_SCALE  = 0.4 
MIN_SIZE  = 100          # 2D: 100, 3D: 500

def run_gss_2d(lum, mem):
    H, W = lum.shape
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    v_cost = (LUM_SCALE * lum_s) - (MEM_SCALE * mem_s) - (BG_SCALE * (1 - lum_s))
    v_cost = v_cost.ravel(order='F').astype(np.float64)

    affinity_R = (lum_s[:, 1:] + lum_s[:, :-1]) / 2
    membrane_R = np.maximum(mem_s[:, 1:], mem_s[:, :-1])
    eR = affinity_R - MEM_SCALE * membrane_R
    eR_full = np.pad(eR, ((0, 0), (0, 1)), constant_values=0.)

    affinity_D = (lum_s[1:, :] + lum_s[:-1, :]) / 2
    membrane_D = np.maximum(mem_s[1:, :], mem_s[:-1, :])
    eD = affinity_D - MEM_SCALE * membrane_D
    eD_full = np.pad(eD, ((0, 1), (0, 0)), constant_values=0.)
    i_cost = np.concatenate(
        [eR_full.ravel(order='F'),
         eD_full.ravel(order='F')]
    ).astype(np.float64)

    # ── 3. ASSEMBLE COSTS ─────────────────────────────────────────


    shape = np.array([H, W], dtype=np.uint64)
    connectivity = np.array([0, 1, 1, 0], dtype=np.int32)

    gss = GreedySeparatorShrinking()
    gss.setup_grid(shape, connectivity, v_cost, i_cost)
    gss.run()

    labels = np.asarray(gss.vertex_labels(), np.int32).reshape((H, W), order='F')
    labels[mem > 0.3] = 1

    return labels


def run_gsg_2d(lum: np.ndarray, mem: np.ndarray) -> np.ndarray:
    H, W = lum.shape
    # Smooth inputs
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    v_cost = (LUM_SCALE * lum_s) + (MEM_SCALE * mem_s) - (BG_SCALE * (1 - lum_s))

    # ── 2. INTERACTION COSTS (This model is good, we'll keep it) ────
    affinity_R = (lum_s[:, 1:] + lum_s[:, :-1]) / 2
    membrane_R = np.maximum(mem_s[:, 1:], mem_s[:, :-1])
    eR = affinity_R - MEM_SCALE * membrane_R
    eR_full = np.pad(eR, ((0, 0), (0, 1)), constant_values=0.)

    affinity_D = (lum_s[1:, :] + lum_s[:-1, :]) / 2
    membrane_D = np.maximum(mem_s[1:, :], mem_s[:-1, :])
    eD = affinity_D - MEM_SCALE * membrane_D
    eD_full = np.pad(eD, ((0, 1), (0, 0)), constant_values=0.)

    # ── 3. ASSEMBLE COSTS ─────────────────────────────────────────
    costs = np.stack([
        v_cost, 
        eR_full, 
        eD_full
    ], axis=0).ravel(order='F').astype(np.float64)

    shape        = np.array([H, W], dtype=np.uint64)
    connectivity = [[0, 1], [1, 0]]

    gsg = GreedySeparatorGrowing2D(shape, connectivity, costs)
    gsg.run()

    labels = np.asarray(gsg.vertex_labels(), np.int32).reshape((H, W), order='F')
    labels[mem > 0.3] = 1

    return labels

# def run_gsg_3d(lum: np.ndarray, mem: np.ndarray) -> np.ndarray:
#     """Run GSG for 3D ER segmentation."""
#     D, H, W = lum.shape
    
#     # Smooth inputs
#     lum_smooth = gaussian_filter(lum, sigma=SIGMA)
#     mem_smooth = gaussian_filter(mem, sigma=SIGMA)
    
#     # Define 3D offsets
#     offsets = [
#         [-1, 0, 0], [1, 0, 0],
#         [0, -1, 0], [0, 1, 0],
#         [0, 0, -1], [0, 0, 1]
#     ]
    
#     # Build costs array for 3D
#     costs = []
#     for z in range(D):
#         for y in range(H):
#             for x in range(W):
#                 # Vertex cost
#                 cost = -lum_smooth[z, y, x] * LUM_SCALE + mem_smooth[z, y, x] * MEM_SCALE + BIAS
#                 costs.append(cost)
                
#                 # Interaction costs
#                 for dz, dy, dx in offsets:
#                     nz, ny, nx = z + dz, y + dy, x + dx
#                     if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
#                         lum_diff = abs(lum_smooth[z, y, x] - lum_smooth[nz, ny, nx])
#                         mem_barrier = max(mem_smooth[z, y, x], mem_smooth[nz, ny, nx])
#                         interaction_cost = (1.0 - lum_diff) - MEM_SCALE * mem_barrier
#                         costs.append(interaction_cost)
#                     else:
#                         costs.append(0.0)
    
#     print(f"3D GSG: {D}x{H}x{W} volume, {len(costs)} total costs")
    
#     # Create and run 3D GSG
#     gsg = GreedySeparatorGrowing3D([D, H, W], offsets, costs)
#     gsg.run()
#     labels = np.asarray(gsg.vertex_labels(), np.int32).reshape(D, H, W)
#     return labels

def process_func(x: torch.Tensor) -> torch.Tensor:
    """Process function handling both 2D and 3D data."""
    device = x.device
    
    # Handle 2D/3D inputs
    if x.ndim == 4:  # (B, C, H, W) - 2D
        B, C, H, W = x.shape
        x = x.unsqueeze(2)  # Add Z: (B, C, 1, H, W)
        Z = 1
        squeeze_output = True
        is_2d = True
    else:  # (B, C, Z, H, W) - 3D
        B, C, Z, H, W = x.shape
        squeeze_output = False
        is_2d = (Z == 1)
    
    assert C == 2, f"Expected 2 channels (lumen, membrane), got {C}"
    
    out = torch.zeros((B, 1, Z, H, W), dtype=torch.float32, device=device)
    
    for b in range(B):
        lum_t = x[b, 0].cpu()           
        mem_t = x[b, 1].cpu()
        print(f"\nBatch {b}: Lum range [{lum_t.min():.2f}, {lum_t.max():.2f}], "
                    f"Mem range [{mem_t.min():.2f}, {mem_t.max():.2f}]")
        
        lum = 1 / (1 + np.exp(-lum_t.numpy()))  # Sigmoid to [0,1]
        mem = 1 / (1 + np.exp(-mem_t.numpy()))  #

        print(f"After normalized: Lum [{lum.min():.3f}, {lum.max():.3f}], "
                f"Mem [{mem.min():.3f}, {mem.max():.3f}]")
        
        # Ensure in [0,1] range
        lum = np.clip(lum, 0, 1)
        mem = np.clip(mem, 0, 1)
        
        # Process based on dimensionality
            # Handle 2D data
        lum_2d = lum[0] if lum.ndim == 3 else lum
        mem_2d = mem[0] if mem.ndim == 3 else mem
        
        print(f"Processing 2D slice of shape {lum_2d.shape}")
        
            # Try 2D GSG
        labels = run_gss_2d(lum_2d, mem_2d)
        er_mask = (labels > 0).astype(np.float32)
        print(f"2D MSP found {er_mask.sum()} ER pixels in {len(np.unique(labels))-1} components")

        # Reshape back to 3D
        er_mask = er_mask[np.newaxis, :, :]  # Add Z dimension back

        
        # Post-processing
        er_mask = binary_fill_holes(er_mask)
        labels = binary_closing(labels > 0, structure=np.ones((4,4))) 
        
        # if is_2d:
        #     # For 2D, process each slice
        #     er_mask[0] = remove_small_objects(er_mask[0].astype(bool), min_size=min_size)
        # else:
        #     # For 3D, process as volume
        #     er_mask = remove_small_objects(er_mask.astype(bool), min_size=min_size)
        
        print(f"After post-processing: {er_mask.sum()} ER {'pixels' if is_2d else 'voxels'}")
        
        out[b, 0] = torch.from_numpy(er_mask.astype(np.float32))
    
    # Remove Z dimension if input was 2D
    if squeeze_output:
        out = out[:, :, 0]
    
    return out

if __name__ == "__main__":
    from cellmap_segmentation_challenge import process
    process(__file__, overwrite=True)