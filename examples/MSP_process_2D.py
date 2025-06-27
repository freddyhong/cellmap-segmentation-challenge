from cellmap_segmentation_challenge.utils import load_safe_config
import torch
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_closing,
    binary_erosion,
)
from multi_separator import GreedySeparatorShrinking

# ——— Configuration loading ———
config_path       = "train_2D_V2.py"
config            = load_safe_config(config_path)
batch_size        = getattr(config, "batch_size", 8)
input_array_info  = getattr(
    config,
    "input_array_info",
    {"shape": (1, 128, 128), "scale": (8, 8, 8)},
)
target_array_info = getattr(config, "target_array_info", input_array_info)
classes           = config.classes
C_in, Hc, Wc      = input_array_info["shape"]

def run_gss(lumen: np.ndarray, membrane: np.ndarray = None) -> np.ndarray:
    h, w = lumen.shape
    # 1) Smooth exactly as legacy
    lumen_s = gaussian_filter(lumen, sigma=0.5)
    mem_s   = gaussian_filter(membrane, sigma=0.5) if membrane is not None else None

    # 2) Vertex costs (h*w entries)
    vertex_costs = []
    for i in range(h):
        for j in range(w):
            lv = lumen_s[i, j]
            mv = mem_s[i, j] if mem_s is not None else 0.0
            vertex_costs.append(lv * 2.0 - mv * 3.0 - 0.5)
    assert len(vertex_costs) == h * w

    # 3) Interaction costs (2 passes, 2*h*w entries)
    interaction_costs = []
    for di, dj in [(0, 1), (1, 0)]:
        for i in range(h):
            for j in range(w):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    l1, l2 = lumen_s[i, j], lumen_s[ni, nj]
                    similarity = 1.0 - abs(l1 - l2)
                    strength   = min(l1, l2)
                    penalty    = (max(mem_s[i, j], mem_s[ni, nj]) * 2.0 
                                  if mem_s is not None else 0.0)
                    cost = similarity * strength * 1.5 - penalty
                    interaction_costs.append(cost if abs(cost) > 0.01 else -0.1)
                else:
                    interaction_costs.append(-1.0)
    assert len(interaction_costs) == 2 * h * w

    # 4) Run GSS with the same connectivity as legacy
    gss = GreedySeparatorShrinking()
    # connectivity = [neg_x, pos_x, neg_y, pos_y]
    gss.setup_grid([h, w], [0, 1, 1, 0], vertex_costs, interaction_costs)
    gss.run()

    return np.array(gss.vertex_labels(), dtype=np.int32).reshape(h, w)

def process_func(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C_in, Hc, Wc) raw logits
    returns: (B, 1, Hc, Wc) integer‐labeled masks
    """
    device = x.device
    B, C, H, W = x.shape
    assert C == C_in, f"Expected {C_in} channels, got {C}"

    probs = torch.sigmoid(x).cpu().numpy()  # (B, C_in, Hc, Wc)
    out   = np.zeros((B, 1, H, W), dtype=np.float32)

    for b in range(B):
        lumen_map = probs[b, 0]
        mem_map   = probs[b, 1] if C_in >= 2 else None

        # 1) Legacy‐cost GSS
        labels = run_gss(lumen_map, mem_map)

        # 2) Morphological refinement & false‐positive pruning
        final = np.zeros_like(labels, dtype=np.int32)
        nid   = 1
        for L in np.unique(labels):
            if L == 0:
                continue
            mask = (labels == L)

            # a) close small gaps
            mask = binary_closing(mask, structure=np.ones((3, 3)))
            # b) fill holes
            mask = binary_fill_holes(mask)

            # c) drop tiny components
            if mask.sum() < 20:
                continue

            # d) prune by membrane support at the boundary
            if mem_map is not None:
                boundary = mask ^ binary_erosion(mask)
                if mem_map[boundary].mean() < 0.2:
                    continue

            final[mask] = nid
            nid += 1

        out[b, 0] = final

    return torch.from_numpy(out).to(device)

if __name__ == "__main__":
    from cellmap_segmentation_challenge import process
    print(
        f"Running legacy‐cost GSS + morphological refinements\n"
        f" classes={classes}, input shape={input_array_info['shape']}"
    )
    process(__file__, overwrite=True)