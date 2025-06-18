import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# %% Imports
from upath import UPath
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellmap_segmentation_challenge.models import ResNet, UNet_2D

class JaccardLoss(nn.Module):
    def __init__(self, smooth = 1, reduction='none'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        spatial_dims = tuple(range(2, pred.ndim))
        
        intersection = (pred * target).sum(dim=spatial_dims, keepdim=True)  
        pred_sum = pred.sum(dim=spatial_dims, keepdim=True)  
        target_sum = target.sum(dim=spatial_dims, keepdim=True) 
        
        jaccard_coeff = (intersection + self.smooth) / (pred_sum + target_sum - intersection + self.smooth) 
        
        jaccard_loss = (1 - jaccard_coeff) * self.smooth  
        jaccard_loss = jaccard_loss.expand_as(target)
        
        return jaccard_loss


# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizerlearning_rate = 0.0003    # Moderate start
batch_size = 16  # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the targets
epochs = 300  # number of epochs to train the model for
iterations_per_epoch = 500 # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = ["endo", "cell", "lyso", "mito", "nuc"]

# Defining model (comment out all that are not used)
# 2D UNets
model_name = "Jaccard_unet_New"  # name of the model to use
model_to_load = "Jaccard_unet_New"  # name of the pre-trained model to load
model = UNet_2D(1, len(classes))

# 2D ResNet [uncomment to use]
# model_name = "2d_resnet"  # name of the model to use
# model_to_load = "2d_resnet"  # name of the pre-trained model to load
# # model = ResNet(ndims=2, output_nc=len(classes))
# model = ResNet(
#     ndims=2, 
#     output_nc=len(classes),
#     ngf=32,          
#     n_blocks=6,      
#     n_downsampling=2,  
#     use_dropout=False  
# )

load_model = "latest"  # load the "latest" model or the "best" validation model
# load_model = "checkpoints/2d_unet_38.pth"

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    #"rotate": {"axes": {"x": [-20, 20], "y": [-20, 20]}},
}

validation_time_limit = None
validation_batch_limit = None 
# use_s3 = True # Use S3 for data loading


gradient_accumulation_steps = 2  
weighted_sampler = True          
max_grad_norm = 1.0             
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
force_all_classes = True
weight_loss = False

criterion = JaccardLoss
criterion_kwargs = {"smooth": 1.0, "reduction": "none"}

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=learning_rate,
#     weight_decay=1e-4,  # Regularization for long training
#     eps=1e-8
# )
checkpoint_frequency = 10

memory_management = True
clear_cache_frequency = 50 


if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)