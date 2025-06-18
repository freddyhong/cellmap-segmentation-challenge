import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


# %% Imports
from upath import UPath
import torch
import torch.nn as nn
import torch.nn.functional as F
from cellmap_segmentation_challenge.models import UNet_2D
import numpy as np
from matplotlib.colors import Normalize

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='none', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
                shape = [1, len(self.alpha)] + [1] * (len(inputs.shape) - 2)
                alpha_t = alpha_t.view(*shape)
                focal_weight = focal_weight * (alpha_t * targets + (1 - alpha_t) * (1 - targets))
            else:
                focal_weight = focal_weight * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, squared=False, reduction='none', pos_weight=None):
        super().__init__()
        self.smooth = smooth
        self.squared = squared
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))
        
        # Flatten spatial dimensions for calculation
        pred_flat = pred.view(batch_size, num_classes, -1)
        target_flat = target.view(batch_size, num_classes, -1)
        
        # Calculate intersection and sums
        intersection = (pred_flat * target_flat).sum(dim=2)
        
        if self.squared:
            pred_sum = (pred_flat * pred_flat).sum(dim=2)
            target_sum = (target_flat * target_flat).sum(dim=2)
        else:
            pred_sum = pred_flat.sum(dim=2)
            target_sum = target_flat.sum(dim=2)
        
        # Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1 - dice_coeff  # Shape: [B, C]
        
        # Expand back to spatial dimensions
        spatial_shape = target.shape[2:]
        dice_loss_spatial = dice_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        dice_loss_spatial = dice_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return dice_loss_spatial.mean()
        elif self.reduction == 'sum':
            return dice_loss_spatial.sum()
        else:
            return dice_loss_spatial

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
        
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        # Get dimensions
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim)) 
        
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)  # False positives with gamma
        
        dice_score = (2 * tp + self.epsilon) / (2 * tp + fn + fp + self.epsilon)
        dice_loss = 1 - dice_score  # Shape: [B, C]
        
        spatial_shape = target.shape[2:]
        dice_loss_spatial = dice_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        dice_loss_spatial = dice_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return dice_loss_spatial.mean()
        elif self.reduction == 'sum':
            return dice_loss_spatial.sum()
        else:
            return dice_loss_spatial

class CombinedLoss(nn.Module):
    def __init__(self, losses, reduction='none', pos_weight=None):
        super().__init__()
        self.losses = nn.ModuleList([loss[0] for loss in losses])
        self.weights = [loss[1] for loss in losses]
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss
        return total_loss

# %% Set hyperparameters
learning_rate = 0.0001 
batch_size = 16
input_array_info = {"shape": (1, 128, 128), "scale": (8, 8, 8)}
target_array_info = {"shape": (1, 128, 128), "scale": (8, 8, 8)}
epochs = 250
iterations_per_epoch = 500
random_seed = 42

device = "cuda"
classes = ["cell", "endo", "ld", "mito", "mt", "ves", "lyso"]

# Model configuration
model_name = "combined_loss"
model_to_load = "combined_loss"
model = UNet_2D(1, len(classes))

load_model = "latest"

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/CombinedLoss/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Spatial transformations - reduced to avoid too much augmentation
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
}

# Training settings
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

validation_time_limit = None
validation_batch_limit = None 
use_s3 = True


def create_focal_dice_loss(**kwargs):
    # Remove 'reduction' from kwargs to avoid duplicate argument error
    kwargs_focal = kwargs.copy()
    kwargs_focal.pop('reduction', None)
    
    # Alpha values: higher for rare classes (mito, lyso), lower for common (cell)
    focal = FocalLoss(
        # alpha=[
        #     0.08,  # cell
        #     0.12,  # endo 
        #     0.18,  # ld 
        #     0.18,  # mito
        #     0.15,  # mt
        #     0.14,  # ves 
        #     0.15,  # lyso
        # ],
        alpha=0.5,
        gamma=2.0,
        reduction='none',  # Always 'none' for CellMapLossWrapper
        **kwargs_focal
    )
    
    kwargs_dice = kwargs.copy()
    kwargs_dice.pop('reduction', None)
    
    # dice = DiceLoss(
    #     smooth=1.0, 
    #     squared=True,  
    #     reduction='none', 
    #     **kwargs_dice
    # )
    
    dice = DicePlusLoss(
        gamma=2.0,
        epsilon=1e-7,
        reduction='none',
        **kwargs_dice
    )

    kwargs_combined = kwargs.copy()
    kwargs_combined.pop('reduction', None)
    
    return CombinedLoss(
        [(focal, 0.5), (dice, 0.5)], 
        reduction='none',
        **kwargs_combined
    )

criterion = create_focal_dice_loss
criterion_kwargs = {}

# Optimizer with gradient clipping
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
    eps=1e-8
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
# scheduler_kwargs = {
#     "T_0": 40,      # Restart every 40 epochs
#     "T_mult": 2,    # Double the period after each restart
#     "eta_min": 1e-6 # Minimum learning rate
# }


scheduler = torch.optim.lr_scheduler.StepLR
scheduler_kwargs = {
    "step_size": 25,    
    "gamma": 0.1,      
}


if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)