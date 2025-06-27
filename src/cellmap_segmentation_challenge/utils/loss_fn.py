import torch
import torch.nn as nn
import torch.nn.functional as F
#-------------- Dice Plus loss function --------------#

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

#-------------- Helper function to identify axis for 2D/3D tensors --------------#

def identify_axis(shape):
    if len(shape) == 5: return [2,3,4]
    elif len(shape) == 4: return [2,3]
    else: raise ValueError('Shape of tensor is neither 2D nor 3D.')

#-------------- Custom Focal Loss Functions --------------#

# Define all custom loss classes here
class SymmetricFocalLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super().__init__()
        self.delta, self.gamma, self.epsilon = delta, gamma, epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        ce = -y_true * torch.log(y_pred)
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * ce[:,0] * (1 - self.delta)
        fore_ce = torch.pow(1 - y_pred[:,1], self.gamma) * ce[:,1] * self.delta
        return torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=-1), dim=-1))

class AsymmetricFocalLoss(SymmetricFocalLoss):
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        ce = -y_true * torch.log(y_pred)
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * ce[:,0] * (1 - self.delta)
        fore_ce = ce[:,1] * self.delta
        return torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=-1), dim=-1))

class SymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super().__init__()
        self.delta, self.gamma, self.epsilon = delta, gamma, epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice = (tp + self.epsilon) / (tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)
        back = (1-dice[:,0]) * torch.pow(1-dice[:,0], -self.gamma)
        fore = (1-dice[:,1]) * torch.pow(1-dice[:,1], -self.gamma)
        return torch.mean(torch.stack([back, fore], dim=-1))

class AsymmetricFocalTverskyLoss(SymmetricFocalTverskyLoss):
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice = (tp + self.epsilon) / (tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)
        back = 1 - dice[:,0]
        fore = (1-dice[:,1]) * torch.pow(1-dice[:,1], -self.gamma)
        return torch.mean(torch.stack([back, fore], dim=-1))

class SymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, reduction='none'):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        ftl = SymmetricFocalTverskyLoss(self.delta, self.gamma)(y_pred, y_true)
        fl = SymmetricFocalLoss(self.delta, self.gamma)(y_pred, y_true)
        return self.weight * ftl + (1 - self.weight) * fl

class AsymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2, reduction='none'):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        ftl = AsymmetricFocalTverskyLoss(self.delta, self.gamma)(y_pred, y_true)
        fl = AsymmetricFocalLoss(self.delta, self.gamma)(y_pred, y_true)
        return self.weight * ftl + (1 - self.weight) * fl
    
import torch
import torch.nn as nn
import torch.nn.functional as F

#-------------- Helper function to identify axis for 2D/3D tensors --------------#
def identify_axis(shape):
    if len(shape) == 5: return [2,3,4]
    elif len(shape) == 4: return [2,3]
    else: raise ValueError('Shape of tensor is neither 2D nor 3D.')

class MultiClassUnifiedFocalLoss(nn.Module):
    """
    Multi-class version of Unified Focal Loss - MINIMAL FIXES ONLY
    Preserving exact logic from binary version
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, reduction='none'):
        super().__init__()
        self.weight = weight      # λ: balance between focal and tversky components
        self.delta = delta        # δ: control output imbalance (foreground emphasis)
        self.gamma = gamma        # γ: focal parameter
        self.reduction = reduction
        self.epsilon = 1e-7
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, C, H, W] - predicted logits
            y_true: [B, C, H, W] - ground truth (one-hot or soft labels)
        """
        # Convert logits to probabilities
        y_pred = torch.softmax(y_pred, dim=1)
        # FIX 1: Clamp to prevent log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        
        # Ensure we have proper dimensions
        batch_size, num_classes = y_pred.shape[:2]
        spatial_dims = tuple(range(2, y_pred.ndim))
        
        # Modified Focal Loss Component
        focal_loss = self._modified_focal_loss(y_pred, y_true, spatial_dims)
        
        # Modified Focal Tversky Loss Component  
        tversky_loss = self._modified_focal_tversky_loss(y_pred, y_true, spatial_dims)
        
        # Combine losses
        unified_loss = self.weight * tversky_loss + (1 - self.weight) * focal_loss
        
        # Expand back to spatial dimensions for CellMapLossWrapper compatibility
        spatial_shape = y_true.shape[2:]
        unified_loss_spatial = unified_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
        unified_loss_spatial = unified_loss_spatial.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return unified_loss_spatial.mean()
        elif self.reduction == 'sum':
            return unified_loss_spatial.sum()
        else:
            return unified_loss_spatial
    
    def _modified_focal_loss(self, y_pred, y_true, spatial_dims):
        """Modified Focal Loss - keeping exact binary logic"""
        # Cross entropy loss per class
        ce_loss = -y_true * torch.log(y_pred)
        
        # FIX 2: Remove keepdim to match dimensions properly
        pt = (y_pred * y_true).sum(dim=spatial_dims)  # Shape: [B, C]
        
        # Apply focal modulation - EXACT LOGIC FROM BINARY VERSION
        focal_weight = self.delta * torch.pow(1 - pt, 1 - self.gamma)
        
        # FIX 3: Expand focal_weight to match ce_loss dimensions before multiplication
        for _ in spatial_dims:
            focal_weight = focal_weight.unsqueeze(-1)
        
        focal_loss = focal_weight * ce_loss
        focal_loss = focal_loss.sum(dim=spatial_dims)  # Sum over spatial dimensions
        
        return focal_loss
    
    def _modified_focal_tversky_loss(self, y_pred, y_true, spatial_dims):
        """Modified Focal Tversky Loss - keeping exact binary logic"""
        # Calculate Tversky Index components - EXACT LOGIC FROM BINARY VERSION
        tp = (y_true * y_pred).sum(dim=spatial_dims)
        fn = (y_true * (1 - y_pred)).sum(dim=spatial_dims)  
        fp = ((1 - y_true) * y_pred).sum(dim=spatial_dims)
        
        # Modified Tversky Index (mTI) - EXACT FORMULA FROM BINARY VERSION
        mti = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        
        # Apply focal modulation - EXACT LOGIC FROM BINARY VERSION
        tversky_loss = torch.pow(1 - mti, self.gamma)
        
        return tversky_loss


class AsymmetricMultiClassUnifiedFocalLoss(MultiClassUnifiedFocalLoss):
    """
    Asymmetric version - MINIMAL FIXES ONLY
    Preserving exact logic from binary version
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, reduction='none', pos_weight=None, **kwargs):
        super().__init__(weight, delta, gamma, reduction)
        self.pos_weight = pos_weight
    
    def forward(self, y_pred, y_true):
        """Override to match exact binary version logic"""
        # Convert to probabilities and clamp
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        
        # Get axis for operations
        axis = identify_axis(y_true.size())
        
        # EXACT LOGIC FROM BINARY AsymmetricUnifiedFocalLoss
        ftl = self._asymmetric_focal_tversky(y_pred, y_true, axis)
        fl = self._asymmetric_focal(y_pred, y_true, axis)
        
        # Combine losses - EXACT FORMULA
        unified_loss = self.weight * ftl + (1 - self.weight) * fl
        
        # FIX 4: Expand to spatial dimensions for CellMapLossWrapper
        batch_size, num_classes = y_pred.shape[:2]
        spatial_shape = y_true.shape[2:]
        
        # The binary version returns a mean, so we need to expand properly
        if len(unified_loss.shape) == 0:  # scalar
            unified_loss = unified_loss.expand(batch_size, num_classes, *spatial_shape)
        else:
            # Reshape to match expected output
            unified_loss = unified_loss.view(batch_size, -1)
            if unified_loss.shape[1] == 1:  # binary case
                unified_loss = unified_loss.expand(batch_size, num_classes)
            unified_loss = unified_loss.view(batch_size, num_classes, *([1] * len(spatial_shape)))
            unified_loss = unified_loss.expand(-1, -1, *spatial_shape)
        
        if self.reduction == 'mean':
            return unified_loss.mean()
        elif self.reduction == 'sum':
            return unified_loss.sum()
        else:
            return unified_loss
    
    def _asymmetric_focal_tversky(self, y_pred, y_true, axis):
        """EXACT LOGIC from AsymmetricFocalTverskyLoss"""
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice = (tp + self.epsilon) / (tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)
        
        # FIX 5: Handle multi-class indexing
        if len(dice.shape) == 2 and dice.shape[1] > 1:  # Multi-class
            back = 1 - dice[:,0]
            fore = (1-dice[:,1]) * torch.pow(1-dice[:,1], -self.gamma)
            # For multi-class, average over all foreground classes
            if dice.shape[1] > 2:
                fore_all = []
                for i in range(1, dice.shape[1]):
                    fore_all.append((1-dice[:,i]) * torch.pow(1-dice[:,i], -self.gamma))
                fore = torch.stack(fore_all, dim=-1).mean(dim=-1)
            return torch.mean(torch.stack([back, fore], dim=-1))
        else:
            # Binary case logic
            back = 1 - dice[:,0] if len(dice.shape) > 1 else 1 - dice
            fore = (1-dice[:,1]) * torch.pow(1-dice[:,1], -self.gamma) if len(dice.shape) > 1 else (1-dice) * torch.pow(1-dice, -self.gamma)
            return torch.mean(torch.stack([back, fore], dim=-1))
    
    def _asymmetric_focal(self, y_pred, y_true, axis):
        """EXACT LOGIC from AsymmetricFocalLoss"""
        ce = -y_true * torch.log(y_pred)
        
        # FIX 6: Handle multi-class indexing
        if len(ce.shape) > 2 and ce.shape[1] > 1:  # Multi-class
            back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * ce[:,0] * (1 - self.delta)
            fore_ce = ce[:,1] * self.delta
            # For multi-class, average over all foreground classes
            if ce.shape[1] > 2:
                fore_all = []
                for i in range(1, ce.shape[1]):
                    fore_all.append(ce[:,i] * self.delta)
                fore_ce = torch.stack(fore_all, dim=-1).mean(dim=-1)
            # Sum over spatial dimensions if they exist
            if len(axis) > 0:
                back_ce = back_ce.sum(axis=[a-1 for a in axis if a > 1])  # Adjust axis after class dim
                fore_ce = fore_ce.sum(axis=[a-1 for a in axis if a > 1])
            return torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=-1), dim=-1))
        else:
            # Binary case logic
            back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * ce[:,0] * (1 - self.delta)
            fore_ce = ce[:,1] * self.delta
            return torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=-1), dim=-1))