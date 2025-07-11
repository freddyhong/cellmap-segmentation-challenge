import torch
import torch.nn as nn
import torch.nn.functional as F

class DicePlusLoss(nn.Module):
    """
    Dice++ Loss
    https://github.com/mlyg/DicePlusPlus
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
        
        # Clip predictions and targets to avoid numerical instability
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        # Get dimensions
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))  # (2, 3) for 2D, (2, 3, 4) for 3D
        
        # Calculate Dice++ components exactly as in paper
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)  # False positives with gamma
        
        # Dice++ coefficient
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
        
class FocalLoss(nn.Module):
    """
    Focal Loss for independent binary classification per channel.
    Optimized for CellMap's multi-class segmentation challenge.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'none',
        pos_weight: torch.Tensor = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W, D] - Raw logits
            target: [B, C, H, W, D] - Binary targets
        """
        # Ensure target is same dtype as pred
        if target.dtype != pred.dtype:
            target = target.to(pred.dtype)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Calculate BCE loss with pos_weight for each channel independently
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(pred.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pos_weight, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )
    
        pred_prob = torch.sigmoid(pred)
        p_t = torch.where(target >= 0.5, pred_prob, 1 - pred_prob)
        
        if self.alpha is not None:
            alpha_t = torch.where(target >= 0.5, self.alpha, 1 - self.alpha)
        else:
            alpha_t = 1.0
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class FocalDicePlusLoss(nn.Module):
    """
    Combined Focal + Dice++ Loss
    Similar to unified focal loss (focal + CE loss)
    https://github.com/mlyg/unified-focal-loss
    """
    def __init__(
        self,
        focal_weight: float = 0.6,
        dice_weight: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_gamma: float = 2.0,
        epsilon: float = 1e-7,
        reduction: str = 'none',
        pos_weight: torch.Tensor = None,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            focal_weight: Weight for focal loss component (0.0 to 1.0)
            dice_weight: Weight for dice++ loss component (0.0 to 1.0)
            focal_alpha: Alpha parameter for focal loss (class balancing)
            focal_gamma: Gamma parameter for focal loss (focusing)
            dice_gamma: Gamma parameter for dice++ loss (FN/FP weighting)
            epsilon: Small constant for numerical stability
            reduction: 'none', 'mean', or 'sum'
            pos_weight: Per-class positive weights for focal loss
            label_smoothing: Label smoothing factor (0.0 to 0.2)
        """
        super().__init__()
        
        # Validate weights
        if abs(focal_weight + dice_weight - 1.0) > 1e-6:
            print(f"Warning: focal_weight ({focal_weight}) + dice_weight ({dice_weight}) != 1.0")
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        # Initialize component losses
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction=reduction,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing
        )
        
        self.dice_loss = DicePlusLoss(
            gamma=dice_gamma,
            epsilon=epsilon,
            reduction=reduction,
            pos_weight=pos_weight
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate individual loss components
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Combine with weights
        combined_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return combined_loss
    
    def get_component_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        combined = self.focal_weight * focal + self.dice_weight * dice
        
        return {
            'focal': focal.mean().item() if hasattr(focal, 'mean') else focal,
            'dice': dice.mean().item() if hasattr(dice, 'mean') else dice,
            'combined': combined.mean().item() if hasattr(combined, 'mean') else combined
        }