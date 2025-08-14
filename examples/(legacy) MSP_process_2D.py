from cellmap_segmentation_challenge.utils import load_safe_config
import torch

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing, binary_erosion
from multi_separator import GreedySeparatorShrinking
import upath as UPath
from cellmap_data import CellMapImage

# Load training config
config_path = "train_2D_Progressive.py"
config = load_safe_config(config_path)

# Required by process.py
batch_size = getattr(config, "batch_size", 8)
# input_array_info = getattr(config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)})
# target_array_info = getattr(config, "target_array_info", input_array_info)

input_array_info = {"shape": (2, 98, 98), "scale": (8, 8, 8)}
target_array_info = input_array_info

label_groups = {
    "mito": ["mito_lum", "mito_mem"],
}

classes = ["mito"]


from scipy.ndimage import gaussian_filter

# ─── hyper-parameters you asked for ──────────────────────────────
LUM_SCALE_FIXED = 2.0      # LOW lumen weight  (was 2.0)
MEM_SCALE_FIXED = 3.0      # HIGH membrane penalty (was 1.5)
BIAS_FIXED      = 0.5     # keep a moderate bias
# ────────────────────────────────────────────────────────────────

def run_gss(lumen: np.ndarray,
            membrane: np.ndarray | None,
            lum_scale: float = LUM_SCALE_FIXED,
            mem_scale: float = MEM_SCALE_FIXED,
            bias: float      = BIAS_FIXED) -> np.ndarray:
    """
    Lumen / membrane logits  →  MSP segmentation (H, W) int32.
    The three weights can be tuned from the caller.
    """
    h, w = lumen.shape

    # --- smooth exactly like the legacy code --------------------------------
    lum = gaussian_filter(lumen,    sigma=0.5)
    mem = gaussian_filter(membrane, sigma=0.5) if membrane is not None else None

    # --- vertex costs -------------------------------------------------------
    vertex_costs = [
        lum_scale * lum[i, j] - mem_scale * (mem[i, j] if mem is not None else 0.0) - bias
        for i in range(h) for j in range(w)
    ]

    # --- interaction (+x, +y) costs ----------------------------------------
    interaction_costs = []
    for di, dj in [(0, 1), (1, 0)]:                       # +x, +y
        for i in range(h):
            for j in range(w):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    l1, l2 = lum[i, j], lum[ni, nj]
                    similarity = 1.0 - abs(l1 - l2)
                    strength   = min(l1, l2)
                    penalty    = 2.0 * max(mem[i, j], mem[ni, nj]) if mem is not None else 0.0
                    cost = similarity * strength * 1.5 - penalty
                    interaction_costs.append(cost if abs(cost) > .01 else -0.1)
                else:
                    interaction_costs.append(-1.0)

    # --- run the Greedy Separator Shrinking cut ----------------------------
    gss = GreedySeparatorShrinking()
    gss.setup_grid([h, w], [0, 1, 1, 0], vertex_costs, interaction_costs)
    gss.run()
    return np.asarray(gss.vertex_labels(), np.int32).reshape(h, w)

def process_func(x: torch.Tensor) -> torch.Tensor:
    dev, drop_z = x.device, False
    if x.ndim == 4:
        x, drop_z = x.unsqueeze(2), True

    B, C, Z, H, W = x.shape
    probs = torch.sigmoid(x).cpu().numpy()
    out   = np.zeros((B, 1, Z, H, W), np.float32)

    for b in range(B):
        for z in range(Z):
            lum = probs[b, 0, z]
            mem = probs[b, 1, z]

            # *** call run_gss ONCE with the fixed hyper-params ***
            labels = run_gss(lum, mem)          # ← uses the constants above

            # ---- simple post-processing (unchanged) -------------
            final, nid = np.zeros_like(labels), 1
            for lab in np.unique(labels):
                if lab == 0:                     # background
                    continue
                mask = binary_closing(labels == lab, np.ones((3, 3)))
                mask = binary_fill_holes(mask)
                if mask.sum() < 20:
                    continue
                boundary = mask ^ binary_erosion(mask)
                if mem[boundary].mean() < 0.20:
                    continue
                final[mask] = nid; nid += 1

            out[b, 0, z] = final

    if drop_z:
        out = out[:, :, 0]
    return torch.from_numpy(out).to(dev)


if __name__ == "__main__":
    from cellmap_segmentation_challenge import process
    process(__file__, overwrite=True)