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

from cellmap_segmentation_challenge.utils.loss_fn import AsymmetricUnifiedFocalLoss, SymmetricUnifiedFocalLoss
from cellmap_segmentation_challenge.models import ResNet, UNet_3D, ViTVNet
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Set hyperparameters and other configurations
learning_rate = 0.0005  # learning rate for the optimizer
batch_size = 2  # batch size for the dataloader
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 100  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = ["mito", "mito_lum", "mito_mem"]  # classes to segment
# classes = get_tested_classes()  # list of classes to segment

# Defining model (comment out all that are not used)
# 3D UNet
model_name = "3d_unified_focal"  # name of the model to use
model_to_load = "3d_unified_focal"  # name of the pre-trained model to load
model = UNet_3D(1, len(classes))

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
    "checkpoints/3D_unified_focal/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "new_datasplit_mito.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y", "z"]},
}

filter_by_scale = True  # filter the data by scale


gradient_accumulation_steps = 4 
weighted_sampler = True          
max_grad_norm = 1.0             
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
weight_loss = False 


validation_time_limit = None  # time limit in seconds for the validation step
validation_batch_limit = None  # Skip validation

# use_s3 = True # Use S3 for data loading

criterion = AsymmetricUnifiedFocalLoss
criterion_kwargs = {
    "weight":0.6,     # balance between Tversky and CE
    "delta":0.6,      # class weighting (foreground emphasis)
    "gamma":0.5,     
    "reduction": "none"
}


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.StepLR
scheduler_kwargs = {
    "step_size": 20,
    "gamma": 0.1,
}
checkpoint_frequency = 10
memory_management = True
clear_cache_frequency = 50 

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    train(__file__)
