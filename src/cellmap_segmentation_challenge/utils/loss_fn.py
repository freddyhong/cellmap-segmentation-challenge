import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, reduction='none', pos_weight=None):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        # BCEWithLogitsLoss is numerically stable
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight, reduction='none'
        )

        # Sigmoid for Dice calculation
        pred_sigmoid = torch.sigmoid(pred)
        
        # Standard Dice calculation
        spatial_dims = tuple(range(2, pred.ndim))
        intersection = (pred_sigmoid * target).sum(dim=spatial_dims)
        union = pred_sigmoid.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
        dice_score = (2. * intersection + 1e-7) / (union + 1e-7)
        dice_loss = 1 - dice_score

        # Expand Dice loss to match BCE loss shape
        batch_size, num_classes = pred.shape[:2]
        spatial_shape = target.shape[2:]
        dice_loss = dice_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        dice_loss = dice_loss.expand(-1, -1, *spatial_shape)

        # Combine the two losses
        combined_loss = self.bce_weight * bce + self.dice_weight * dice_loss

        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss

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
        
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        # Get dimensions
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))  # (2, 3) for 2D, (2, 3, 4) for 3D
        
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)  # False positives with gamma
        
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
        

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-7, reduction='none', pos_weight=None, logcosh=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.logcosh = logcosh
    
    def forward(self, pred, target):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        # Get dimensions
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))  # (2, 3) for 2D, (2, 3, 4) for 3D
        
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred))).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred)).sum(dim=spatial_dims)  # False positives with gamma
        
        tversky_score = (tp + self.epsilon) / (tp + self.alpha * fn + self.beta * fp + self.epsilon)
        if self.logcosh:
            error = 1 - tversky_score
            tversky_loss = torch.log(torch.cosh(error))  
        else:
            tversky_loss = 1 - tversky_score  
        
        spatial_shape = target.shape[2:]
        tversky_loss_spatial = tversky_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        tversky_loss_spatial = tversky_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return tversky_loss_spatial.mean()
        elif self.reduction == 'sum':
            return tversky_loss_spatial.sum()
        else:
            return tversky_loss_spatial


class TverskyPlusLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-7, reduction='none', pos_weight=None, logcosh=False, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.logcosh = logcosh
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        target = torch.clamp(target, self.epsilon, 1 - self.epsilon)
        
        # Get dimensions
        batch_size, num_classes = pred.shape[:2]
        spatial_dims = tuple(range(2, pred.ndim))  # (2, 3) for 2D, (2, 3, 4) for 3D
        
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)  # False positives with gamma
        
        tversky_score = (2 * tp + self.epsilon) / (2 * tp + self.alpha * fn + self.beta * fp + self.epsilon)
        if self.logcosh:
            error = 1 - tversky_score
            tversky_loss = torch.log(torch.cosh(error))  
        else:
            tversky_loss = 1 - tversky_score  
        
        spatial_shape = target.shape[2:]
        tversky_loss_spatial = tversky_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        tversky_loss_spatial = tversky_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return tversky_loss_spatial.mean()
        elif self.reduction == 'sum':
            return tversky_loss_spatial.sum()
        else:
            return tversky_loss_spatial

class LogCoshDicePlusLoss(nn.Module):
    """
    Dice++ Losss
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
        
        tp = (target * pred).sum(dim=spatial_dims)  # True positives
        fn = ((target * (1 - pred)).pow(self.gamma)).sum(dim=spatial_dims)  # False negatives with gamma
        fp = (((1 - target) * pred).pow(self.gamma)).sum(dim=spatial_dims)  # False positives with gamma
        
        dice_score = (2 * tp + self.epsilon) / (2 * tp + fn + fp + self.epsilon)
        error = 1 - dice_score
        dice_loss = torch.log(torch.cosh(error))  # Shape: [B, C]
        
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
        if target.dtype != pred.dtype:
            target = target.to(pred.dtype)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
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
        logcosh: bool = False,
        Euc_distance: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        # Validate weights
        if abs(focal_weight + dice_weight - 1.0) > 1e-6:
            print(f"Warning: focal_weight ({focal_weight}) + dice_weight ({dice_weight}) != 1.0")
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.Euc_distance = Euc_distance
        
        # Initialize component losses
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction=reduction,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing
        )
        if logcosh:
            self.dice_loss = LogCoshDicePlusLoss(
                gamma=dice_gamma,
                epsilon=epsilon,
                reduction=reduction,
                pos_weight=pos_weight
            )
        else:
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
        if self.Euc_distance:
            combined_loss = torch.sqrt((self.focal_weight * focal)**2 + (self.dice_weight * dice)**2)
        else:
            combined_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return combined_loss
    

class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.6,
        tversky_weight: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        tversky_gamma: float = 2.0,
        epsilon: float = 1e-7,
        reduction: str = 'none',
        pos_weight: torch.Tensor = None,
        logcosh: bool = False,
        tverskyplus: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        # Validate weights
        if abs(focal_weight + tversky_weight - 1.0) > 1e-6:
            print(f"Warning: focal_weight ({focal_weight}) + tversky_weight ({tversky_weight}) != 1.0")
        
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.reduction = reduction
        self.tverskyplus = tverskyplus
        
        # Initialize component losses
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction=reduction,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing
        )
        if self.tverskyplus:
            self.tversky_loss = TverskyPlusLoss(
                alpha = tversky_alpha,
                beta=tversky_beta,
                epsilon=epsilon,
                reduction=reduction,
                logcosh=logcosh,
                pos_weight=pos_weight,
                gamma=tversky_gamma
            )
        else:
            self.tversky_loss = TverskyLoss(
                alpha = tversky_alpha,
                beta=tversky_beta,
                epsilon=epsilon,
                reduction=reduction,
                logcosh=logcosh,
                pos_weight=pos_weight
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate individual loss components
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        combined_loss = self.focal_weight * focal + self.tversky_weight * tversky
        
        return combined_loss

class AsymmetricUnifiedFocalLoss(nn.Module):
    """
    Implementation of the Asymmetric Unified Focal Loss for multi-label semantic segmentation.
    This loss combines a modified Focal Loss and a modified Focal Tversky Loss to
    handle class imbalance effectively.

    Reference: https://doi.org/10.1016/j.compmedimag.2021.102026
    """
    def __init__(self, weight: float = 0.5, delta: float = 0.6, gamma: float = 0.5, reduction = None):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred_logits)
        
        spatial_dims = tuple(range(2, pred_logits.ndim))
        
        tp = (pred_prob * target).sum(dim=spatial_dims)
        fp = (pred_prob * (1 - target)).sum(dim=spatial_dims)
        fn = ((1 - pred_prob) * target).sum(dim=spatial_dims)
        
        tversky_index = (tp + 1e-7) / (tp + self.delta * fp + (1 - self.delta) * fn + 1e-7)
        tversky_index = torch.clamp(tversky_index, 0, 1)

        focal_tversky_loss = torch.pow(1 - tversky_index, 1 - self.gamma)

        bce_loss = F.binary_cross_entropy_with_logits(torch.clamp(pred_logits, -30, 30), target, reduction='none')

        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t)**self.gamma
        
        alpha_t = torch.where(target == 1, 1.0, focal_term)
        
        delta_t = torch.where(target == 1, self.delta, 1 - self.delta)
        
        asymmetric_focal_loss = delta_t * alpha_t * bce_loss
        
        asymmetric_focal_loss = asymmetric_focal_loss.mean()
        
        final_loss = self.weight * asymmetric_focal_loss + (1 - self.weight) * focal_tversky_loss.mean()
        final_loss = torch.nan_to_num(final_loss, nan=0.0, posinf=1e4, neginf=0.0)
        return final_loss


class MultiClassComboLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, ce_weight: torch.Tensor = None, **kwargs):
        super(MultiClassComboLoss, self).__init__()
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == pred_logits.ndim:
            target_indices = torch.argmax(target, dim=1)
        else:
            target_indices = target
        
        target_indices = target_indices.long()
        
        ce = self.cross_entropy_loss(pred_logits, target_indices)
        ce = ce.unsqueeze(1)

        pred_softmax = F.softmax(pred_logits, dim=1)
        target_one_hot = F.one_hot(target_indices, num_classes=pred_logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        spatial_dims = tuple(range(2, pred_logits.ndim))
        intersection = (pred_softmax * target_one_hot).sum(dim=spatial_dims)
        union = pred_softmax.sum(dim=spatial_dims) + target_one_hot.sum(dim=spatial_dims)
        
        dice_score = (2. * intersection + 1e-7) / (union + 1e-7)
        dice_loss = 1 - dice_score

        batch_size, num_classes = pred_logits.shape[:2]
        spatial_shape = target_indices.shape[1:]
        dice_loss = dice_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        dice_loss = dice_loss.expand(-1, -1, *spatial_shape)
        
        combo_loss = (self.alpha * ce) + ((1 - self.alpha) * dice_loss)

        return combo_loss
    
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        log_softmax_pred = F.log_softmax(pred_logits, dim=1)

        log_pt = log_softmax_pred.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = - (1 - pt) ** self.gamma * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(pred_logits.device)
            alpha_t = alpha.gather(0, target_indices.view(-1)).view(*target_indices.shape)
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss

class MultiClassDicePlusLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, epsilon: float = 1e-7, logcosh: bool = False):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.logcosh = logcosh

    def forward(self, pred_logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        pred_softmax = F.softmax(pred_logits, dim=1)
        
        num_classes = pred_logits.shape[1]
        target_one_hot = F.one_hot(target_indices, num_classes=num_classes)
        
        # Permute one-hot to [B, C, H, W, D]
        if pred_logits.ndim == 5: # 3D
            target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        elif pred_logits.ndim == 4: # 2D
             target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        pred_softmax = torch.clamp(pred_softmax, self.epsilon, 1 - self.epsilon)
        
        spatial_dims = tuple(range(2, pred_logits.ndim))
        tp = (target_one_hot * pred_softmax).sum(dim=spatial_dims)
        fn = ((target_one_hot * (1 - pred_softmax)).pow(self.gamma)).sum(dim=spatial_dims)
        fp = (((1 - target_one_hot) * pred_softmax).pow(self.gamma)).sum(dim=spatial_dims)
        
        if self.logcosh:
            dice_score = (2 * tp + self.epsilon) / (2 * tp + fn + fp + self.epsilon)
            error = 1 - dice_score
            dice_loss_log = torch.log(torch.cosh(error))
            dice_loss = dice_loss_log.mean(dim=1)
        else:
            dice_score = (2 * tp + self.epsilon) / (2 * tp + fn + fp + self.epsilon)
            dice_loss = 1 - dice_score.mean(dim=1)

        return dice_loss.mean()

class MultiClassFocalDicePlusLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_alpha: torch.Tensor = None,
        focal_gamma: float = 2.0,
        dice_gamma: float = 2.0,
        logcosh: bool = False,
        reduction = "none"
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.logcosh = logcosh
        self.focal_loss = MultiClassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = MultiClassDicePlusLoss(gamma=dice_gamma, logcosh=logcosh)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == pred_logits.ndim:
            target_indices = torch.argmax(target, dim=1).long()
        else:
            target_indices = target.long()
        focal = self.focal_loss(pred_logits, target_indices)
        dice = self.dice_loss(pred_logits, target_indices)

        combined_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return combined_loss