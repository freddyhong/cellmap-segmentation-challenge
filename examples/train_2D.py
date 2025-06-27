# jrc-cos7-1b
# hela2 - crop 1,3,4,6

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# %% Imports
from upath import UPath
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellmap_data.transforms.augment import (
    Normalize, NaNtoNum, GaussianNoise, GaussianBlur,
    RandomGamma, RandomContrast
)
import torchvision.transforms.v2 as T

from cellmap_segmentation_challenge.utils.loss_fn import AsymmetricMultiClassUnifiedFocalLoss, AsymmetricUnifiedFocalLoss, SymmetricUnifiedFocalLoss, DicePlusLoss
from cellmap_segmentation_challenge.models import ResNet, UNet_2D


# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 4  # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 200  # number of epochs to train the model for
iterations_per_epoch = 500  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = [
    "cell", "endo", "ld", "lyso", "mito", "mt", "np", "nuc", "perox", "ves", "vim",
    "pm", "ecs", "cyto",            # cell separator
    "endo_mem", "endo_lum",         # endo separator
    "ld_mem", "ld_lum",             # ld separator
    "lyso_mem", "lyso_lum",         # lyso separator
    "mito_mem", "mito_lum",         # mito separator
    "np_in", "np_out",              # np separator
    "ne_mem", "ne_lum",             # nuc separator
    "perox_mem", "perox_lum",       # perox separator
    "ves_mem", "ves_lum"            # ves separator
]


# Defining model (comment out all that are not used)
# 2D UNets
model_name = "dice_plus_ultimate"  # name of the model to use
model_to_load = "dice_plus_ultimate"  # name of the pre-trained model to load
model = UNet_2D(1, len(classes))

# # 2D ResNet [uncomment to use]
# model_name = "2d_resnet"  # name of the model to use
# model_to_load = "2d_resnet"  # name of the pre-trained model to load
# model = ResNet(ndims=2, output_nc=len(classes))

load_model = "latest"  # load the "latest" model or the "best" validation model

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/dice_plus_ultimate/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "new_datasplit_ultimate.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
}

train_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    RandomGamma(gamma_range=(0.8, 1.2)),
    RandomContrast(contrast_range=(0.75, 1.25)),
    GaussianNoise(mean=0.0, std=0.05),
    GaussianBlur(kernel_size=3, sigma=1.0, dim=2, channels=1),
])

val_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
])


gradient_accumulation_steps = 8
weighted_sampler = True          
max_grad_norm = 0.8            
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
weight_loss = True 

validation_time_limit = None  # time limit in seconds for the validation step
validation_batch_limit = None  # Skip validation

# use_s3 = True # Use S3 for data loading

criterion = DicePlusLoss
criterion_kwargs = {
    "gamma": 2.0,        # Controls the penalty for false positives/negatives
    "epsilon": 1e-7,     # Numerical stability
    "reduction": "none"  # Keep as "none" for CellMap compatibility
}

# criterion = AsymmetricMultiClassUnifiedFocalLoss
# criterion_kwargs = {
#     "weight":0.5, # balance between Tversky and CE
#     "delta":0.85, # class weighting (foreground emphasis)
#     "gamma":1,
#     "reduction": "none"
# }

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-3,
    eps=1e-8,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.StepLR
scheduler_kwargs = {
    "step_size": 75,        # Reduce every 50 epochs
    "gamma": 0.8           # Reduce by factor of 0.6
}


checkpoint_frequency = 10
memory_management = True
clear_cache_frequency = 10

import gc
torch.cuda.empty_cache()
gc.collect()

# # LOAD CHECKPOINT IMMEDIATELY (not in main block)
# print("🔄 Attempting to load checkpoint...")
# checkpoint_path = "checkpoints/unified_focal (lyso)/unified_focal_100.pth"
# try:
#     print(f"📁 Loading from: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, weights_only=True)
    
#     # Load only compatible layers
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in checkpoint.items() 
#                       if k in model_dict and model_dict[k].shape == v.shape}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
    
#     print(f"✅ Successfully loaded {len(pretrained_dict)} layers from checkpoint")
#     print(f"❌ Skipped {len(checkpoint) - len(pretrained_dict)} incompatible layers")
#     print("🎯 Transfer learning setup complete!")
    
# except Exception as e:
#     print(f"❌ Failed to load checkpoint: {e}")
#     print("⚠️  Starting from scratch...")



if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)