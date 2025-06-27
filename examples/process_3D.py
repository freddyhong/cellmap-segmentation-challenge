from cellmap_segmentation_challenge.utils import load_safe_config
import torch
from scipy.ndimage import gaussian_filter
import numpy as np

# Load the configuration file
config_path = __file__.replace("process", "train")
config = load_safe_config(config_path)


# Bring the required configurations into the global namespace
batch_size = getattr(config, "batch_size", 8)
input_array_info = getattr(
    config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
)
target_array_info = getattr(config, "target_array_info", input_array_info)
classes = config.classes


# Define the process function, which takes a numpy array as input and returns a numpy array as output
def process_func(x):
    x_np = torch.sigmoid(x).cpu().numpy()
    x_np = gaussian_filter(x_np, sigma=1)
    return torch.tensor((x_np > 0.7).astype(np.float32)).to(x.device)


if __name__ == "__main__":
    from cellmap_segmentation_challenge import process

    # Call the process function with the configuration file
    process(__file__, overwrite=True)
