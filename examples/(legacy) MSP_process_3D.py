from cellmap_segmentation_challenge.utils import load_safe_config
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from multi_separator import GreedySeparatorShrinking, GreedySeparatorGrowing3D, GreedySeparatorGrowing2D
import time
import matplotlib.pyplot as plt

# Load training config
config_path = __file__.replace("MSP_process", "train")
config = load_safe_config(config_path)


# Bring the required configurations into the global namespace
batch_size = getattr(config, "batch_size", 8)
input_array_info = getattr(
    config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
)
target_array_info = getattr(config, "target_array_info", input_array_info)

# lumen channel first, membrane channel second
label_groups = {
    "endo": ["endo_lum", "endo_mem"],
}

classes = ["endo"]

# Tunable parameters
SIGMA = 0.4
LUM_SCALE = 1.0  # Attraction force for lumen (object)
MEM_SCALE = 1.0  
BG_SCALE  = 1.0 
MIN_SIZE  = 100        

def run_gss_2d(lum, mem):
    H, W = lum.shape
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    v_cost = (LUM_SCALE * lum_s) + (MEM_SCALE * mem_s) - (BG_SCALE * (1 - lum_s))
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

    shape = np.array([H, W], dtype=np.uint64)
    connectivity = np.array([0, 1, 1, 0], dtype=np.int32)

    gss = GreedySeparatorShrinking()
    gss.setup_grid(shape, connectivity, v_cost, i_cost)
    gss.run()

    labels = np.asarray(gss.vertex_labels(), np.int32).reshape((H, W), order='F')
    labels[mem > 0.3] = 1

    return labels

def run_gss_2d_exp(lum, mem):
    H, W = lum.shape
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    sum_prob = np.maximum(lum_s, mem_s) 
    g = 1 - sum_prob
    eps = 1e-8
    v_cost = np.log((1 - g + eps) / (g + eps))
    BIAS = 0.0
    v_cost += BIAS
    v_cost = v_cost.ravel(order='F').astype(np.float64)

    affinity_R = (lum_s[:, 1:] + lum_s[:, :-1]) / 2
    # membrane_R = np.maximum(mem_s[:, 1:], mem_s[:, :-1])
    membrane_R = np.maximum(mem_s[:, 1:], mem_s[:, :-1])
    eR = affinity_R - MEM_SCALE * membrane_R
    eR_full = np.pad(eR, ((0, 0), (0, 1)), constant_values=0.)

    affinity_D = (lum_s[1:, :] + lum_s[:-1, :]) / 2
    # membrane_D = np.maximum(mem_s[1:, :], mem_s[:-1, :])
    membrane_D = np.maximum(mem_s[1:, :], mem_s[:-1, :])
    eD = affinity_D - MEM_SCALE * membrane_D
    eD_full = np.pad(eD, ((0, 1), (0, 0)), constant_values=0.)
    i_cost = np.concatenate(
        [eR_full.ravel(order='F'),
         eD_full.ravel(order='F')]
    ).astype(np.float64)

    shape = np.array([H, W], dtype=np.uint64)
    connectivity = np.array([0, 1, 1, 0], dtype=np.int32)

    gss = GreedySeparatorShrinking()
    gss.setup_grid(shape, connectivity, v_cost, i_cost)
    gss.run()

    labels = np.asarray(gss.vertex_labels(), np.int32).reshape((H, W), order='F')

    return labels

def run_gsg_2d(lum: np.ndarray, mem: np.ndarray) -> np.ndarray:
    H, W = lum.shape
    # Smooth inputs
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    v_cost = (LUM_SCALE * lum_s) + (MEM_SCALE * mem_s) - (BG_SCALE * (1 - lum_s))

    affinity_R = (lum_s[:, 1:] + lum_s[:, :-1]) / 2
    membrane_R = np.maximum(mem_s[:, 1:], mem_s[:, :-1])
    eR = affinity_R - MEM_SCALE * membrane_R
    eR_full = np.pad(eR, ((0, 0), (0, 1)), constant_values=0.)

    affinity_D = (lum_s[1:, :] + lum_s[:-1, :]) / 2
    membrane_D = np.maximum(mem_s[1:, :], mem_s[:-1, :])
    eD = affinity_D - MEM_SCALE * membrane_D
    eD_full = np.pad(eD, ((0, 1), (0, 0)), constant_values=0.)

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

def run_gsg_2d_exp(lum: np.ndarray, mem: np.ndarray) -> np.ndarray:
    AFFINITY_SCALE = 2.5
    CUT_SCALE = 0.8

    H, W = lum.shape
    # Smooth inputs
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    sum_prob = np.maximum(lum_s, mem_s) 
    g = 1 - sum_prob
    eps = 1e-8
    v_cost = np.log((1 - g + eps) / (g + eps))
    # v_cost = sum_prob - g
    BIAS = 0.0
    v_cost += BIAS

    connectivity = [[0, 1], [1, 0], [0, 3], [3, 0], [0, 5], [5, 0], [3, 3], [5, 5]]
    edge_costs = []

    for offset in connectivity:
        dy, dx = offset
        
        if dy == 0 and dx != 0:  # Horizontal offset
            if dx > 0 and sum_prob.shape[1] > dx:  # Right direction
                affinity_R = (sum_prob[:, dx:] + sum_prob[:, :-dx]) / 2
                membrane = np.maximum(mem_s[:, dx:], mem_s[:, :-dx])
                edge_cost = affinity_R * AFFINITY_SCALE - CUT_SCALE * membrane
                edge_cost_full = np.pad(edge_cost, ((0, 0), (0, dx)), constant_values=0.)
            elif dx < 0 and sum_prob.shape[1] > abs(dx):  # Left direction
                dx_abs = abs(dx)
                affinity_R = (sum_prob[:, :-dx_abs] + sum_prob[:, dx_abs:]) / 2
                membrane = np.maximum(mem_s[:, :-dx_abs], mem_s[:, dx_abs:])
                edge_cost = affinity_R * AFFINITY_SCALE - CUT_SCALE * membrane
                edge_cost_full = np.pad(edge_cost, ((0, 0), (dx_abs, 0)), constant_values=0.)
            else:
                edge_cost_full = np.zeros_like(sum_prob)
                
        elif dy != 0 and dx == 0:  # Vertical offset
            if dy > 0 and sum_prob.shape[0] > dy:  # Down direction
                affinity_D = (sum_prob[dy:, :] + sum_prob[:-dy, :]) / 2
                membrane = np.maximum(mem_s[dy:, :], mem_s[:-dy, :])
                edge_cost = affinity_D * AFFINITY_SCALE - CUT_SCALE * membrane
                edge_cost_full = np.pad(edge_cost, ((0, dy), (0, 0)), constant_values=0.)
            elif dy < 0 and sum_prob.shape[0] > abs(dy):  # Up direction
                dy_abs = abs(dy)
                affinity_D = (sum_prob[:-dy_abs, :] + sum_prob[dy_abs:, :]) / 2
                membrane = np.maximum(mem_s[:-dy_abs, :], mem_s[dy_abs:, :])
                edge_cost = affinity_D * AFFINITY_SCALE - CUT_SCALE * membrane
                edge_cost_full = np.pad(edge_cost, ((dy_abs, 0), (0, 0)), constant_values=0.)
            else:
                edge_cost_full = np.zeros_like(sum_prob)
                
        elif dy != 0 and dx != 0:  # Diagonal offset
            if (dy > 0 and dx > 0 and sum_prob.shape[0] > dy and sum_prob.shape[1] > dx):
                affinity = (sum_prob[dy:, dx:] + sum_prob[:-dy, :-dx]) / 2
                membrane = np.maximum(mem_s[dy:, dx:], mem_s[:-dy, :-dx])
                edge_cost = affinity * AFFINITY_SCALE - CUT_SCALE * membrane
                edge_cost_full = np.pad(edge_cost, ((0, dy), (0, dx)), constant_values=0.)
            else:
                edge_cost_full = np.zeros_like(sum_prob)
        
        edge_costs.append(edge_cost_full)

    # Stack all costs
    costs = np.stack([v_cost] + edge_costs, axis=0).ravel(order='F').astype(np.float64)

    shape = np.array([H, W], dtype=np.uint64)
    

    gsg = GreedySeparatorGrowing2D(shape, connectivity, costs)
    gsg.run()

    labels = np.asarray(gsg.vertex_labels(), np.int32).reshape((H, W), order='F')

    return labels

def run_gss_3d(lum, mem):
    D, H, W = lum.shape

    connectivity = np.array([
    [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]
], dtype=np.int32)
    
    lum_s = gaussian_filter(lum, sigma=SIGMA)
    mem_s = gaussian_filter(mem, sigma=SIGMA)

    sum_prob = np.maximum(lum_s, mem_s) 
    g = 1 - sum_prob
    eps = 1e-8
    v_cost = np.log((1 - g + eps) / (g + eps))
    # v_cost = sum_prob - g
    BIAS = 0.0
    v_cost += BIAS

    # v_cost = (LUM_SCALE * lum_s) + (MEM_SCALE * mem_s) - (BG_SCALE * (1 - lum_s))
    v_cost = v_cost.ravel(order='F').astype(np.float64)
    
    interaction_cost_list = []
    for offset in connectivity:
        # Define slices for the source (s1) and destination (s2) voxels
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        for i in range(3): # For Z, Y, X axes
            if offset[i] > 0:
                s1[i] = slice(None, -offset[i])
                s2[i] = slice(offset[i], None)
            elif offset[i] < 0:
                s1[i] = slice(-offset[i], None)
                s2[i] = slice(None, offset[i])

        # Calculate costs on the overlapping region
        affinity = (lum_s[tuple(s1)] + lum_s[tuple(s2)]) / 2
        membrane = np.maximum(mem_s[tuple(s1)], mem_s[tuple(s2)])
        edge_cost = affinity - MEM_SCALE * membrane

        # Pad the result to full size, aligning with the source voxel
        pad_width = []
        for i in range(3):
            if offset[i] > 0:
                pad_width.append((0, offset[i]))
            else:
                pad_width.append((abs(offset[i]), 0))

        edge_cost_full = np.pad(edge_cost, pad_width, 'constant', constant_values=0)
        interaction_cost_list.append(edge_cost_full.ravel(order='F'))

    i_cost_flat = np.concatenate(interaction_cost_list).astype(np.float64)

    # 3. Setup and Run GSS
    shape = np.array([D, H, W], dtype=np.uint64)
    flat_connectivity = connectivity.flatten()

    gss = GreedySeparatorShrinking()
    gss.setup_grid(shape, flat_connectivity, v_cost, i_cost_flat)
    gss.run()

    labels = np.asarray(gss.vertex_labels(), np.int32).reshape((D, H, W), order='F')

    return labels


def process_func(batch: torch.Tensor) -> torch.Tensor:
    x = batch["input"]
    
    device = x.device
    print(f"Input tensor shape (from MSP process 3D): {x.shape}")
    
    # Handle different input tensor shapes
    if x.ndim == 4:
        # Shape: (B, C, H, W) - 2D case
        B, C, H, W = x.shape
        Z = 1
        x = x.unsqueeze(2)  # Add Z dimension -> (B, C, 1, H, W)
    elif x.ndim == 5:
        # Shape: (B, C, Z, H, W) - 3D case  
        B, C, Z, H, W = x.shape
    else:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")
    
    print(f"After reshaping: {x.shape}")
    print(f"Expected channels: 2 (lumen, membrane), got: {C}")
    
    if C != 2:
        raise ValueError(f"Expected 2 channels (lumen, membrane), got {C}")
    
    out = torch.zeros((B, 1, Z, H, W), dtype=torch.float32, device=device)
    
    for b in range(B):
        lum_t = x[b, 0].cpu()           
        mem_t = x[b, 1].cpu()
        print(f"\nBatch {b}: Lum range [{lum_t.min():.2f}, {lum_t.max():.2f}], "
                    f"Mem range [{mem_t.min():.2f}, {mem_t.max():.2f}]")
        
        CLIP_RANGE = 20.0
        lum_t_clipped = torch.clamp(lum_t, -CLIP_RANGE, CLIP_RANGE)
        mem_t_clipped = torch.clamp(mem_t, -CLIP_RANGE, CLIP_RANGE)

        # Now, use the clipped tensors for the sigmoid calculation
        lum = 1 / (1 + np.exp(-lum_t_clipped.numpy()))
        mem = 1 / (1 + np.exp(-mem_t_clipped.numpy()))

        print(f"After normalized: Lum [{lum.min():.3f}, {lum.max():.3f}], "
                f"Mem [{mem.min():.3f}, {mem.max():.3f}]")
        
        # Ensure in [0,1] range
        lum = np.clip(lum, 0, 1)
        mem = np.clip(mem, 0, 1)
        
        labels = run_gss_3d(lum, mem)
        er_mask = (labels > 0).astype(np.float32)
        er_mask = binary_fill_holes(er_mask)
        # Use a 3D structuring element for closing
        er_mask = binary_closing(er_mask, structure=np.ones((3, 3, 3))) 
        er_mask = remove_small_objects(er_mask.astype(bool), min_size=MIN_SIZE)
        
        print(f"3D MSP found {er_mask.sum()} ER voxels in {len(np.unique(labels))-1} components")

        # Place the processed 3D mask into the output tensor
        out[b, 0] = torch.from_numpy(er_mask.astype(np.float32)).to(device)
    
    return out


if __name__ == "__main__":
    from cellmap_segmentation_challenge import process
    process(__file__, overwrite=True)