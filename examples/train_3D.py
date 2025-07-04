import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from upath import UPath
import torch
import torch.nn as nn

from cellmap_data.transforms.augment import (
    Normalize, NaNtoNum, GaussianNoise, RandomGamma, RandomContrast
)
import torchvision.transforms.v2 as T

from cellmap_segmentation_challenge.utils.loss import CellMapLossWrapper
from cellmap_segmentation_challenge.models import UNet_3D, ResNet

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 1 # batch size for the dataloader
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 200  # number of epochs to train the model for
iterations_per_epoch = 500  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = [
    "cell", "endo", "ld", "lyso", "mito", "mt", "np", "nuc", "perox", "ves", "vim",
    "pm", "ecs", "cyto", "endo_mem", "endo_lum", "ld_mem", "ld_lum", "lyso_mem", 
    "lyso_lum", "mito_mem", "mito_lum", "np_in", "np_out", "ne_mem", "ne_lum",
    "perox_mem", "perox_lum", "ves_mem", "ves_lum"
]
# classes = get_tested_classes()  # list of classes to segment

# Defining model (comment out all that are not used)
# 3D UNet
model_name = "3D_ultimate_focal"  # name of the model to use
model_to_load = "3D_ultimate_focal"  # name of the pre-trained model to load
# model = UNet_3D(1, len(classes))

# 3D ResNet
# model_name = "3d_resnet"  # name of the model to use
# model_to_load = "3d_resnet"  # name of the pre-trained model to load
model = ResNet(ndims=3, output_nc=len(classes))

# # 3D ViT VNet
# model_name = "3d_vnet"  # name of the model to use
# model_to_load = "3d_vnet"  # name of the pre-trained model to load
# model = ViTVNet(len(classes), img_size=input_array_info["shape"])

load_model = "latest"  # load the latest model or the best validation model

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/3D_ultimate_focal/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "datasplit_ultimate2.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y", "z"]},
}
train_raw_value_transforms = T.Compose([
    Normalize(),                                              # z-score
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    T.RandomChoice([
        RandomGamma(gamma_range=(0.8, 1.2)),          # γ ≈ brightness
        RandomContrast(contrast_range=(0.8, 1.2)),    # ±20 % contrast
    ], p=[0.5, 0.5]), 
    T.RandomApply([GaussianNoise(mean=0.0, std=0.01)], p=0.5),              # subtle noise
])

val_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
])

# class_weights = torch.tensor([
#     # Major structures (common)
#     1.0,   # cell
#     2.0,   # endo (less common)
#     3.0,   # ld (rare)
#     2.5,   # lyso
#     1.5,   # mito (common)
#     2.0,   # mt
#     3.5,   # np (very rare)
#     1.0,   # nuc (common)
#     3.0,   # perox (rare)
#     3.0,   # ves
#     3.5,   # vim
#     1.2,   # pm (common)
#     1.0,   # ecs (common)
#     1.0,   # cyto (common)
#     # Membrane/lumen pairs (usually less common)
#     2.0,   # endo_mem
#     2.0,   # endo_lum
#     3.0,   # ld_mem
#     3.0,   # ld_lum
#     2.5,   # lyso_mem
#     2.5,   # lyso_lum
#     1.5,   # mito_mem
#     1.5,   # mito_lum
#     3.5,   # np_in
#     3.5,   # np_out
#     1.5,   # ne_mem
#     1.5,   # ne_lum
#     3.0,   # perox_mem
#     3.0,   # perox_lum
#     3.0,   # ves_mem
#     3.0,   # ves_lum
# ], device=device)

weight_loss = True  # Don't auto-calculate if using manual weights


# ========== IMPROVED LOSS FUNCTION ==========
# Import your custom loss functions
from cellmap_segmentation_challenge.utils.loss_fn import UnifiedFocalLoss

criterion = UnifiedFocalLoss
criterion_kwargs = {
    "lambda_weight": 0.5,    # Balance between Tversky and CE
    "delta": 0.7,           # Higher delta for better recall on small structures
    "gamma": 2.0,           # Standard focal parameter
    "epsilon": 1e-7,
    "reduction": "none",    # Required by CellMapLossWrapper
    # "pos_weight": None      # Will be set if weight_loss=True
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=3e-4,      # Good for preventing overfitting
    betas=(0.9, 0.999),     # Slightly higher beta2 for stability
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {
    "T_max": epochs,    # 200 epochs full cycle
    "eta_min": 1e-7,    # final LR floor
}
# scheduler = torch.optim.lr_scheduler.StepLR  # Framework default
# scheduler_kwargs = {
#     "step_size": 50,  # Reduce LR every 100 epochs
#     "gamma": 0.6      # Halve the learning rate
# }

# ========== VALIDATION CONFIGURATION ==========
validation_time_limit = None  # 5 minutes max - prevent hanging
validation_batch_limit = None 
filter_by_scale = True  # Framework recommendation

use_amp = True  
gradient_accumulation_steps = 8  # No accumulation
max_grad_norm = 1.0  # No gradient clipping initially
weighted_sampler = False  # Let framework handle sampling
weight_loss = False  # Use weighted loss


# Memory management
torch.backends.cudnn.benchmark = True

# ========== MONITORING ==========
checkpoint_frequency = 25  # Save every 25 epochs
early_stopping_patience = 50  # Be patient with complex multi-class problem

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)