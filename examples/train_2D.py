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
from cellmap_segmentation_challenge.models import UNet_2D, ResNet, UNet2DPlusPlus

# %% Set hyperparameters and other configurations
learning_rate = 0.0001 # learning rate for the optimizer
batch_size = 1 # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 150  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = ["ecs", "pm", "mito_mem", "mito_lum", "mito_ribo", "golgi_mem", "golgi_lum", "ves_mem", "ves_lum", 
           "endo_mem", "endo_lum", "lyso_mem", "lyso_lum", "ld_mem", "ld_lum", "er_mem", "er_lum", "eres_mem", 
           "eres_lum", "ne_mem", "ne_lum", "np_out", "np_in", "hchrom", "echrom", "nucpl", "mt_out", "cyto", 
           "mt_in", "nuc", "golgi", "ves", "endo", "lyso", "ld", "eres", "perox_mem", "perox_lum", "perox", 
           "mito", "er", "ne", "np", "chrom", "mt", "cell", "er_mem_all", "ne_mem_all", "vim"] # For training

# classes = ["mito", "mito_lum", "mito_mem", "cell", "ecs", "ld", "ld_mem", "ld_lum", "lyso", "lyso_lum", "lyso_mem", 
#            "mt", "mt_in", "mt_out", "np", "np_in", "np_out", "nuc", "nucpl", "perox", "perox_lum", "perox_mem", 
#            "ves", "ves_mem", "ves_lum", "vim"]   # for predictions

# classes = get_tested_classes()  # list of classes to segment

# Defining model (comment out all that are not used)
# 3D UNet
model_name = "2D_ultimate_unet_pp_2"  # name of the model to use
model_to_load = "2D_ultimate_unet_pp_2"  # name of the pre-trained model to load
# model = UNet_3D(1, len(classes))

# UNet++
model = UNet2DPlusPlus(1, len(classes))


# 3D ResNet
# model_name = "3d_resnet"  # name of the model to use
# model_to_load = "3d_resnet"  # name of the pre-trained model to load
# model = ResNet(ndims=3, output_nc=len(classes))

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
    "checkpoints/2D_ultimate_unet_pp_2/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path

datasplit_path = "datasplit_ultimate_res.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
}
train_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    T.RandomChoice([
        # Make the range slightly wider
        RandomGamma(gamma_range=(0.4, 1.6)),       
        RandomContrast(contrast_range=(0.4, 1.6)), 
    ], p=[0.5, 0.5]),
    # Apply noise more aggressively
    T.RandomApply([GaussianNoise(mean=0.0, std=0.05)], p=0.9),
])

val_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
])


from cellmap_segmentation_challenge.utils.loss_fn import FocalDicePlusLoss

criterion = FocalDicePlusLoss
criterion_kwargs = {
    "focal_weight": 0.5,
    "dice_weight": 0.5,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "dice_gamma": 2.0,
    "epsilon": 1e-7,
    "reduction": "none",
    "pos_weight": None,
    "label_smoothing": 0.1 # Slightly more smoothing can help generalization
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,      
    weight_decay=1e-5,    
    betas=(0.9, 0.999),    
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {
    "T_max": epochs,
    "eta_min": 1e-6,
}


validation_time_limit = None  # 5 minutes max - prevent hanging
validation_batch_limit = None 
filter_by_scale = True  # Framework recommendation

use_amp = True  
gradient_accumulation_steps = 1  # No accumulation
max_grad_norm = 1.0  # No gradient clipping initially
weighted_sampler = True  # Let framework handle sampling
weight_loss = False 


# Memory management
torch.backends.cudnn.benchmark = True

# ========== MONITORING ==========
checkpoint_frequency = 25  # Save every 25 epochs
early_stopping_patience = 50  # Be patient with complex multi-class problem

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)