from cellmap_segmentation_challenge.utils import load_safe_config
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np
import multi_separator
from multi_separator import GreedySeparatorShrinking 
# Load the configuration file
# config_path = __file__.replace("process", "train")
config_path = "train_2D_V2.py" 
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
    return torch.tensor((x_np > 0.65).astype(np.float32)).to(x.device)
def process_func(x):
    device = x.device
    x_sigmoid = torch.sigmoid(x)
    
    # Use a fixed, global threshold to ensure consistency across patches
    # This is key to avoiding visible boundaries
    global_threshold = 0.53
    
    # Simple, consistent processing
    binary = (x_sigmoid > global_threshold).float()
    
    # Apply minimal morphological operations in PyTorch to maintain consistency
    # Using PyTorch operations ensures exact same results regardless of patch
    kernel_size = 3
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
    
    # Ensure proper shape for convolution
    if len(binary.shape) == 3:
        binary = binary.unsqueeze(1)
    
    # Apply consistent smoothing
    binary = F.conv2d(binary, kernel, padding=1)
    binary = (binary > 0.48).float()
    
    return binary


if __name__ == "__main__":
    from cellmap_segmentation_challenge import process

    # Call the process function with the configuration file
    process(__file__, overwrite=True)
