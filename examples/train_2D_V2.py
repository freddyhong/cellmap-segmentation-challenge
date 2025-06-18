# This is an example of a training configuration file that trains a 2D U-Net model to predict nuclei and endoplasmic reticulum in the CellMap Segmentation Challenge dataset.

# The configuration file defines the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file as an argument to start the training process. The `train` function reads the configuration file, sets up the data loaders, model, optimizer, loss function, and other components, and trains the model for the specified number of epochs.

# The configuration file includes the following components:
# 1. Hyperparameters: learning rate, batch size, input and target array information, epochs, iterations per epoch, random seed, and initial number of features for the model.
# 2. Model: 2D U-Net model with two classes (nuclei and endoplasmic reticulum). (You can also use a 2D ResNet model by uncommenting the relevant lines.)
# 3. Paths: paths for saving logs, model checkpoints, and data split file.
# 4. Spatial transformations: spatial transformations to apply to the training data.

# This configuration file can be used to run training via two different commands:
# 1. `python train_2D.py`: Run the training script directly.
# 2. `csc train train_2D.py`: Run the training script using the `csc train` command-line interface.

# Training progress can be monitored using TensorBoard by running `tensorboard --logdir tensorboard` in the terminal.

# Once the model is trained, you can use the `predict` function to make predictions on new data using the trained model. See the `predict_2D.py` example for more details.

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

class CustomWeightedLoss(nn.Module):
    def __init__(self, class_weights=None, pos_weight=None, reduction='none'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=pred.device, dtype=pred.dtype)

            weights = weights.view(1, -1, 1, 1)

            weight_mask = torch.where(target > 0.5, weights, 1.0)
            loss = loss * weight_mask
        
        return loss
    

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 16  # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 400  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"
classes = ["endo", "cell", "lyso", "mito", "nuc"]

# Defining model (comment out all that are not used)
# 2D UNets
model_name = "2d_unet_V2"  # name of the model to use
model_to_load = "2d_unet_V2"  # name of the pre-trained model to load
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
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    # "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}


gradient_accumulation_steps = 2  
weighted_sampler = True          
max_grad_norm = 1.0             
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
force_all_classes = True
weight_loss = False

custom_pos_weight = torch.tensor([0.8, 0.5, 1.0, 0.7, 0.6], dtype=torch.float32)

criterion = CustomWeightedLoss
criterion_kwargs = {
    "class_weights": [0.8, 0.5, 1.0, 0.7, 0.6],  # endo, cell, lyso, mito, nuc
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
    "step_size": 50,    
    "gamma": 0.7,      
}


validation_time_limit = None  # time limit in seconds for the validation step
validation_batch_limit = None  # Skip validation

use_s3 = True # Use S3 for data loading

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)