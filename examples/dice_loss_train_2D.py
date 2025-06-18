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

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1, squared=False, reduction='none', class_weights=None, pos_weight=None):
#         super().__init__()
#         self.smooth = smooth
#         self.squared = squared
#         self.reduction = reduction
#         self.pos_weight = pos_weight
#         self.class_weights = class_weights

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
        
#         batch_size, num_classes = pred.shape[:2]
#         spatial_dims = tuple(range(2, pred.ndim))
        
#         # Calculate Dice per class per sample
#         intersection = (pred * target).sum(dim=spatial_dims)  # [B, C]
#         pred_sum = pred.sum(dim=spatial_dims)                 # [B, C]
#         target_sum = target.sum(dim=spatial_dims)             # [B, C]
        
#         if self.squared:
#             pred_sum = (pred * pred).sum(dim=spatial_dims)
#             target_sum = (target * target).sum(dim=spatial_dims)
        
#         # Calculate Dice coefficient per sample per class
#         dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
#         dice_loss_per_class = 1 - dice_coeff  # [B, C]
        
#         # Apply class weights
#         if self.class_weights is not None:
#             if isinstance(self.class_weights, (list, tuple)):
#                 weights = torch.tensor(self.class_weights, device=pred.device, dtype=pred.dtype)
#             else:
#                 weights = self.class_weights
            
#             # Reshape for broadcasting [1, C]
#             weights = weights.view(1, -1)
#             dice_loss_per_class = dice_loss_per_class * weights

#         elif self.pos_weight is not None:
#             if self.pos_weight.dim() == 1:
#                 weight = self.pos_weight.unsqueeze(0)
#             else:
#                 weight = self.pos_weight.view(1, -1)
#             dice_loss_per_class = dice_loss_per_class * weight
        
#         spatial_shape = target.shape[2:]  

#         dice_loss_spatial = dice_loss_per_class.view(batch_size, num_classes, *([1] * len(spatial_shape)))
#         dice_loss_spatial = dice_loss_spatial.expand(-1, -1, *spatial_shape)
        
#         return dice_loss_spatial
class DicePlusLoss(nn.Module):
    """
    Dice++ Loss
    """
    def __init__(self, gamma=2.0, epsilon=1e-7, reduction='none', pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Clip predictions and targets to avoid numerical instability (like K.clip in paper)
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        # Get dimensions
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))  # (2, 3) for 2D, (2, 3, 4) for 3D
        
        # Calculate Dice++ components exactly as in paper
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)  # False positives with gamma
        
        # Dice++ coefficient (as in paper)
        dice_score = (2 * tp + self.epsilon) / (2 * tp + fn + fp + self.epsilon)
        dice_loss = 1 - dice_score  # Shape: [B, C]
        
        # Expand back to spatial dimensions for CellMapLossWrapper
        spatial_shape = target.shape[2:]
        dice_loss_spatial = dice_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        dice_loss_spatial = dice_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return dice_loss_spatial.mean()
        elif self.reduction == 'sum':
            return dice_loss_spatial.sum()
        else:
            return dice_loss_spatial

# %% Set hyperparameters and other configurations
learning_rate = 0.0001 
batch_size = 16  # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the targets
epochs = 400  # number of epochs to train the model for
iterations_per_epoch = 1000 # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = ["cell", "endo", "ld", "mito", "mt", "ves", "lyso"]

# Defining model (comment out all that are not used)
# 2D UNets
model_name = "dice_loss_V2"  # name of the model to use
model_to_load = "dice_loss_V2"  # name of the pre-trained model to load
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

# load_model = "latest"  # load the "latest" model or the "best" validation model
load_model = "dice_loss_V2_114.pth"

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/DiceLoss/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "datasplit.csv" 

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.3, "y": 0.3}},
    "transpose": {"axes": ["x", "y"]},
    # "rotate": {"axes": {"x": [-20, 20], "y": [-20, 20]}},
}

validation_time_limit = None
validation_batch_limit = None 
use_s3 = True # Use S3 for data loading

gradient_accumulation_steps = 2  
weighted_sampler = True          
max_grad_norm = 1.0             
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
force_all_classes = True
weight_loss = False 
random_validation = True
validation_prob = 0.15

criterion = DicePlusLoss
criterion_kwargs = {
    "gamma": 2.0,        
    "epsilon": 1e-7,     
    "reduction": "none"
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,  # Regularization for long training
    eps=1e-8
)

checkpoint_frequency = 10
memory_management = True
clear_cache_frequency = 50 

scheduler = torch.optim.lr_scheduler.StepLR
scheduler_kwargs = {
    "step_size": 25,    
    "gamma": 0.5,      
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)