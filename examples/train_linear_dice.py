import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# %% Imports
from upath import UPath
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage import filters, feature, morphology
import warnings
warnings.filterwarnings('ignore')

class DicePlusLoss(nn.Module):
    """Dice++ Loss"""
    def __init__(self, gamma=2.0, epsilon=1e-7, reduction='none', pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))
        
        tp = (target * pred).sum(dim=spatial_dims)
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)
        
        dice_score = (2 * tp + self.epsilon) / (2 * tp + fn + fp + self.epsilon)
        dice_loss = 1 - dice_score
        
        spatial_shape = target.shape[2:]
        dice_loss_spatial = dice_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        dice_loss_spatial = dice_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return dice_loss_spatial.mean()
        elif self.reduction == 'sum':
            return dice_loss_spatial.sum()
        else:
            return dice_loss_spatial


class EnhancedLogisticRegression(nn.Module):
    """Enhanced logistic regression with better features"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.n_features = 10  # INCREASED - more features for better performance
        
        # Enhanced feature set
        self.feature_names = [
            'raw_intensity', 'gauss_0.5', 'gauss_1.0', 'gauss_2.0',
            'sobel_edges', 'gradient_mag', 'laplacian', 'hessian_det',
            'local_std', 'intensity_percentile'
        ]
        
        # Normalization parameters
        self.register_buffer('feat_mean', torch.zeros(self.n_features))
        self.register_buffer('feat_std', torch.ones(self.n_features))
        self.normalize = False
        
        # IMPROVEMENT 1: Add dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)
        
        # IMPROVEMENT 2: Use deeper classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(self.n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def extract_features(self, x):
        """Extract enhanced feature set"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        batch_size, _, height, width = x.shape
        device = x.device
        
        # Move to CPU for feature extraction
        x_cpu = x.cpu().numpy()
        
        # Initialize feature tensor
        features = np.zeros((batch_size, self.n_features, height, width))
        
        for b in range(batch_size):
            img = x_cpu[b, 0]
            
            # Basic features
            features[b, 0] = img  # Raw intensity
            features[b, 1] = filters.gaussian(img, sigma=0.5)
            features[b, 2] = filters.gaussian(img, sigma=1.0)
            features[b, 3] = filters.gaussian(img, sigma=2.0)  # Larger scale
            
            # Edge features
            features[b, 4] = filters.sobel(img)
            
            # Gradient magnitude
            grad_x = ndimage.sobel(img, axis=1)
            grad_y = ndimage.sobel(img, axis=0)
            features[b, 5] = np.sqrt(grad_x**2 + grad_y**2)
            
            # IMPROVEMENT 3: Add more sophisticated features
            # Laplacian (blob detection)
            features[b, 6] = ndimage.laplace(img)
            
            # Hessian determinant (ridge/valley detection)
            try:
                hxx = ndimage.sobel(grad_x, axis=1)
                hyy = ndimage.sobel(grad_y, axis=0)
                hxy = ndimage.sobel(grad_x, axis=0)
                features[b, 7] = hxx * hyy - hxy * hxy
            except:
                features[b, 7] = np.zeros_like(img)
            
            # Local standard deviation (texture)
            try:
                features[b, 8] = ndimage.generic_filter(img, np.std, size=5)
            except:
                features[b, 8] = np.zeros_like(img)
            
            # Local intensity percentile
            try:
                features[b, 9] = ndimage.percentile_filter(img, 75, size=5)
            except:
                features[b, 9] = img.copy()
        
        # Convert back to tensor
        features = torch.from_numpy(features).float().to(device)
        
        # Handle NaN/inf values
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize if enabled
        if self.normalize:
            features = (features - self.feat_mean.view(1, -1, 1, 1)) / (self.feat_std.view(1, -1, 1, 1) + 1e-8)
        
        return features
    
    def set_normalization_params(self, mean, std):
        """Set normalization parameters"""
        self.feat_mean = mean
        self.feat_std = std
        self.normalize = True
    
    def forward(self, x):
        """Forward pass"""
        # Handle dictionary input from CellMap
        if isinstance(x, dict):
            for key in ['raw', 'input', 'data']:
                if key in x:
                    x = x[key]
                    break
            else:
                x = list(x.values())[0]
        
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor input, got {type(x)}")
        
        # Extract features
        features = self.extract_features(x)
        
        # Apply dropout during training
        if self.training:
            features = self.dropout(features)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits


# %% ENHANCED Configuration
learning_rate = 0.005  # IMPROVEMENT 4: Lower learning rate for stability
batch_size = 24        # IMPROVEMENT 5: Slight reduction for memory efficiency
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 128, 128), 
    "scale": (8, 8, 8),
}

# IMPROVED TRAINING SCHEDULE
epochs = 35                    # Slightly more epochs for deeper model
iterations_per_epoch = 400     # Balanced for good coverage

random_seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["mito", "mito_lum", "mito_mem", "mito_ribo"]

# Model
model_name = "enhanced_logistic_dice"
model_to_load = "enhanced_logistic_dice"
model = EnhancedLogisticRegression(len(classes))
load_model = "latest"

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/EnhancedLogistic/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit_mito.csv"

# IMPROVEMENT 6: Enhanced data augmentation
spatial_transforms = {
    "mirror": {"axes": {"x": 0.4, "y": 0.4}},     # Increased probability
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-15, 15], "y": [-15, 15]}},  # Add rotation
    "elastic": {"control_point_spacing": [32, 32], "jitter_sigma": [1, 1]},  # Add elastic deformation
}

validation_time_limit = None
validation_batch_limit = None
use_s3 = True
gradient_accumulation_steps = 2   # IMPROVEMENT 7: Use gradient accumulation
weighted_sampler = True
max_grad_norm = 0.5              # IMPROVEMENT 8: Tighter gradient clipping
torch.backends.cudnn.benchmark = True
filter_by_scale = True
use_mutual_exclusion = False
force_all_classes = True         # IMPROVEMENT 9: Force all classes
weight_loss = True
random_validation = True
validation_prob = 0.2            # IMPROVEMENT 10: More validation

# Loss function
criterion = DicePlusLoss
criterion_kwargs = {
    "gamma": 1.5,                # IMPROVEMENT 11: Less aggressive gamma
    "epsilon": 1e-7,     
    "reduction": "none"
}

# IMPROVEMENT 12: Better optimizer with scheduling
optimizer = torch.optim.AdamW(    # AdamW instead of Adam
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,           # Increased regularization
    eps=1e-8,
    betas=(0.9, 0.999)
)

# IMPROVEMENT 13: Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {
    "T_max": epochs,
    "eta_min": learning_rate * 0.01  # Minimum learning rate
}

# Checkpoint and monitoring
checkpoint_frequency = 5
early_stopping_patience = 10     # IMPROVEMENT 14: Add early stopping concept
save_best_model = True

# Enhanced normalization
compute_normalization_stats = True
normalization_samples = 150      # More samples for better statistics

print("🚀 Enhanced Logistic Regression Configuration Loaded!")
print(f"📊 Features: {model.n_features}")
print(f"🎯 Classes: {len(classes)}")
print(f"⚡ Total training steps: {epochs * iterations_per_epoch}")
print(f"📈 Learning rate schedule: {learning_rate} → {learning_rate * 0.01}")

if __name__ == "__main__":
    # Compute normalization statistics
    if compute_normalization_stats:
        from cellmap_segmentation_challenge.utils import get_dataloader
        from tqdm import tqdm
        
        print("🔢 Computing enhanced normalization statistics...")
        
        # Get dataloader
        train_loader, _ = get_dataloader(
            datasplit_path=datasplit_path,
            classes=classes,
            batch_size=batch_size,
            input_array_info=input_array_info,
            target_array_info=target_array_info,
            spatial_transforms=None,
            iterations_per_epoch=normalization_samples,
            device=device,
            weighted_sampler=False,
        )
        
        # Move model to device
        model = model.to(device)
        
        # Collect feature statistics
        all_features = []
        train_loader.refresh()
        
        print("📊 Extracting features from training data...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader.loader, total=normalization_samples)):
                if i >= normalization_samples:
                    break
                
                try:
                    # Find input data
                    input_data = None
                    if isinstance(batch, dict):
                        for key in ['raw', 'input', 'data']:
                            if key in batch:
                                input_data = batch[key]
                                break
                        if input_data is None:
                            for k, v in batch.items():
                                if isinstance(v, torch.Tensor):
                                    input_data = v
                                    break
                    else:
                        input_data = batch
                    
                    if input_data is None:
                        continue
                    
                    # Extract features
                    features = model.extract_features(input_data)
                    
                    # Collect statistics
                    feat_flat = features.view(features.shape[0], features.shape[1], -1)
                    feat_mean_batch = feat_flat.mean(dim=[0, 2])
                    feat_std_batch = feat_flat.std(dim=[0, 2])
                    
                    all_features.append({
                        'mean': feat_mean_batch,
                        'std': feat_std_batch,
                        'n_pixels': feat_flat.shape[0] * feat_flat.shape[2]
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing batch {i}: {e}")
                    continue
        
        if all_features:
            total_pixels = sum(f['n_pixels'] for f in all_features)
            feat_mean = torch.zeros(model.n_features).to(device)
            feat_std = torch.zeros(model.n_features).to(device)
            
            for f in all_features:
                weight = f['n_pixels'] / total_pixels
                feat_mean += f['mean'] * weight
                feat_std += f['std'] * weight
            
            model.set_normalization_params(feat_mean, feat_std)
            
            print(f"\n📈 Enhanced feature statistics (computed on {total_pixels:,} pixels):")
            for i, name in enumerate(model.feature_names):
                print(f"  {name}: mean={feat_mean[i]:.4f}, std={feat_std[i]:.4f}")
        else:
            print("⚠️ Warning: Could not compute normalization statistics")
        
        # Clean up
        del train_loader
        torch.cuda.empty_cache()
    
    from cellmap_segmentation_challenge import train
    train(__file__)