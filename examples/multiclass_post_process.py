from cellmap_segmentation_challenge.utils import load_safe_config
import torch
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label as find_instances
import numpy as np

config_path = __file__.replace("multiclass_post_process", "train_3D")
config = load_safe_config(config_path)

batch_size = getattr(config, "batch_size", 8)
input_array_info = getattr(
    config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
)
target_array_info = getattr(config, "target_array_info", input_array_info)
classes = config.classes


# classes = ["mito", "mito_lum", "mito_mem", "cell", "ecs", "ld", "ld_mem", "ld_lum", "lyso", "lyso_lum", "lyso_mem", 
#            "mt", "mt_in", "mt_out", "np", "np_in", "np_out", "nuc", "nucpl", "perox", "perox_lum", "perox_mem", 
#            "ves", "ves_mem", "ves_lum", "vim", "endo", "endo_lum", "endo_mem"]   # for predictions

# classes = ["mito", "endo", "lyso", "cell", "perox", "ves"]
classes = ["lyso"]

def process_func(x):
    if isinstance(x, dict):
        tensor_input = list(x.values())[0]
    else:
        tensor_input = x
    print(tensor_input.shape)
    x_np = tensor_input.cpu().numpy()
    x_np = gaussian_filter(x_np, sigma=0.6)

    binary_mask = x_np > 0.4
    binary_mask = binary_fill_holes(binary_mask)
    instance_labels = find_instances(binary_mask)
    instance_labels = remove_small_objects(instance_labels, min_size=50)
    return torch.tensor(instance_labels.astype(np.float32)).to(tensor_input.device) 

if __name__ == "__main__":
    from cellmap_segmentation_challenge import process

    process(__file__, overwrite=True)
