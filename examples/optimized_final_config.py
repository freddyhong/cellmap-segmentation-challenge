# %% FINAL OPTIMIZED CONFIGURATION
from upath import UPath
import torch
from cellmap_segmentation_challenge.models import ResNet

# Settings optimized for your system
learning_rate = 0.003
batch_size = 1

input_array_info = {
    "shape": (1, 48, 48),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 48, 48),
    "scale": (8, 8, 8),
}

epochs = 30
iterations_per_epoch = 15
random_seed = 42
device = "cuda"  # Try GPU with optimized settings
classes = ["endo"]

model_name = "final_optimized"
model_to_load = "final_optimized"
model = ResNet(ndims=2, output_nc=len(classes), ngf=32, n_blocks=6, n_downsampling=2)
load_model = "latest"

logs_save_path = UPath("final_logs/{model_name}").path
model_save_path = UPath("final_models/{model_name}_{epoch}.pth").path
datasplit_path = "endo_datasplit_new.csv"

# Optimized spatial transforms for your system
spatial_transforms = {'mirror': {'axes': {'x': 0.2, 'y': 0.2}}, 'transpose': {'axes': ['x', 'y']}}

validation_time_limit = None
validation_batch_limit = None
gradient_accumulation_steps = 1
weighted_sampler = False
max_grad_norm = 1.0
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
force_all_classes = False

if __name__ == "__main__":
    import os
    os.makedirs("final_logs", exist_ok=True)
    os.makedirs("final_models", exist_ok=True)
    
    from cellmap_segmentation_challenge import train
    train(__file__)
