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
from cellmap_segmentation_challenge.models import UNet_3D, ResNet, UNet3DPlusPlus

# %% Set hyperparameters and other configurations
learning_rate = 0.0001 # was 0.0001
batch_size = 2 # batch size for the dataloader
input_array_info = {
    "shape": (96, 96, 96),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (96, 96, 96),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 150  # number of epochs to train the model for
iterations_per_epoch = 500  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

device = "cuda"

# classes = ["actin","bm","cent","cent_dapp","chlor_lum","chlor_mem","chlor_sg","cyto","echrom","ecs","endo_lum","endo_mem",
#            "er_lum","er_mem","eres_lum","eres_mem","glyco","golgi_lum","golgi_mem","hchrom","isg_ins","isg_lum","isg_mem",
#            "ld_lum","ld_mem","lyso_lum","lyso_mem","mito_lum","mito_mem","mito_ribo","mt_in","mt_out","ne_lum","ne_mem",
#            "nechrom","nhchrom","np_in","np_out","nucleo","nucpl","pd","perox_lum","perox_mem","pm","rbc","ribo","tbar",
#            "vac_lum","vac_mem","ves_lum","ves_mem","vim","yolk_lum","yolk_mem"] # for multi-class

classes = ["mito", "mito_lum", "mito_mem", "cell", "ecs", "ld", "ld_mem", "ld_lum", "lyso", "lyso_lum", "lyso_mem", 
           "mt", "mt_in", "mt_out", "np", "np_in", "np_out", "nuc", "nucpl", "perox", "perox_lum", "perox_mem", 
           "ves", "ves_mem", "ves_lum", "vim", "endo", "endo_lum", "endo_mem"]   # for predictions

# Defining model (comment out all that are not used)
# 3D UNet
model_name = "3D_resnet_focal_tversky_no_logcosh"  # name of the model to use
model_to_load = "3D_resnet_focal_tversky_no_logcosh"  # name of the pre-trained model to load
# model = UNet_3D(1, len(classes))

# model = UNet3DPlusPlus(1, len(classes))

model = ResNet(
    ndims=3, 
    output_nc=len(classes), 
    use_dropout = True, 
    )

# model = ViTVNet(len(classes), img_size=input_array_info["shape"])

load_model = "latest"  # load the latest model or the best validation model

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/3D_resnet_focal_tversky_no_logcosh/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path


datasplit_path = "datasplit_small_mid_res2.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.2}},
    "transpose": {"axes": ["x", "y", "z"]},
    # "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
}
train_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    T.RandomChoice([
        RandomGamma(gamma_range=(0.8, 1.2)),      # Tighter range
        RandomContrast(contrast_range=(0.8, 1.2)),   # Tighter range
    ], p=[0.5, 0.5]),
    T.RandomApply([GaussianNoise(mean=0.0, std=0.01)], p=0.5),
])

# train_raw_value_transforms = T.Compose([
#     Normalize(),
#     NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
#     # You can slightly widen the range of your existing transforms
#     T.RandomChoice([
#         RandomGamma(gamma_range=(0.7, 1.3)),
#         RandomContrast(contrast_range=(0.7, 1.3)),
#     ]),
#     T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
#     T.RandomApply([GaussianNoise(mean=0.0, std=0.05)], p=0.5),
#     T.RandomErasing(p=0.15, scale=(0.02, 0.04), ratio=(0.5, 2.0), value=0),
# ])

val_raw_value_transforms = T.Compose([
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
])


from cellmap_segmentation_challenge.utils.loss_fn import FocalDicePlusLoss, DiceBCELoss, DicePlusLoss, FocalLoss, MultiClassComboLoss, MultiClassFocalDicePlusLoss, FocalTverskyLoss, AsymmetricUnifiedFocalLoss

# criterion = MultiClassComboLoss
# criterion_kwargs = {
#     "alpha": 0.4,
#     "reduction": "mean",
# }

# alpha_weights = torch.ones(len(classes))

# criterion = MultiClassFocalDicePlusLoss
# criterion_kwargs = {
#     "focal_weight" : 0.55,
#     "dice_weight" : 0.45,
#     "focal_alpha" : alpha_weights,
#     "focal_gamma": 2.0,
#     "dice_gamma": 2.0,
#     "logcosh": True,
#     "reduction": "none",
# }


# criterion = FocalLoss
# criterion_kwargs = {
#     "alpha": 0.5,
#     "gamma": 2.0,
#     "reduction": "none",
#     "pos_weight": None,
#     "label_smoothing": 0.1,
# }

# criterion = DiceBCELoss
# criterion_kwargs = {
#     "bce_weight": 0.4,
#     "dice_weight": 0.6,
#     "reduction": "none",
#     "pos_weight": None, 
# }

# criterion = FocalDicePlusLoss
# criterion_kwargs = {
#     "focal_weight": 0.55,
#     "dice_weight": 0.45,
#     "focal_alpha": 0.25,
#     "focal_gamma": 2.0,
#     "dice_gamma": 2.0,
#     "epsilon": 1e-7,
#     "reduction": "none",
#     "pos_weight": None,
#     "logcosh": True,
#     "Euc_distance": True,
#     "label_smoothing": 0.1
# }

criterion = FocalTverskyLoss
criterion_kwargs = {
    "focal_weight": 0.55,
    "tversky_weight": 0.45,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "tversky_alpha": 0.45,
    "tversky_beta": 0.55,
    "tversky_gamma": 2.0,
    "epsilon": 1e-7,
    "reduction": "none",
    "pos_weight": None,
    "logcosh": False,
    "tverskyplus": False,
    "label_smoothing": 0.1
}

# criterion = AsymmetricUnifiedFocalLoss
# criterion_kwargs = {
#     "weight": 0.5,
#     "delta": 0.6,
#     "gamma": 0.6,
#     "reduction": None
# }


# criterion = DicePlusLoss
# criterion_kwargs = {
#     "gamma" : 2.0,
#     "epsilon" : 1e-7,
#     "reduction": "none",
#     "pos_weight": "none",
# }

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,      
    weight_decay=1e-4,    
    betas=(0.9, 0.999),    
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {
    "T_max": epochs,
    "eta_min": 1e-7,
}


validation_time_limit = None  # 5 minutes max - prevent hanging
validation_batch_limit = None 
filter_by_scale = False 

use_amp = True  
gradient_accumulation_steps = 8  # No accumulation
max_grad_norm = 1.0  # No gradient clipping initially
weighted_sampler = True  # Let framework handle sampling
weight_loss = False 


# Memory management
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)