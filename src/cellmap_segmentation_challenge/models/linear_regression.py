import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class LinearRegression(nn.Module):
    def __init__(self, num_classes: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.feature_channels = 6
        self.feature_names = ['Original', 'Mean_Filter', 'Std_Filter', 'Laplacian', 'Gradient_Magnitude', 'Gaussian']
        
        self.linear = nn.Conv2d(
            in_channels=self.feature_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True
        )
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        self.register_buffer('laplacian_kernel', torch.tensor([
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        gaussian_kernel = self._create_gaussian_kernel(kernel_size)
        self.register_buffer('gaussian_kernel', gaussian_kernel)
        
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        self.register_buffer('mean_kernel', mean_kernel)
        
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.feature_transform = nn.Conv2d(6, 12, kernel_size=1)
        self.classifier = nn.Conv2d(12, num_classes, kernel_size=1)
    
    def _calculate_std_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate standard deviation filter using unfold/fold operations."""
        unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        window_mean = unfolded.mean(dim=1, keepdim=True)
        window_var = ((unfolded - window_mean) ** 2).mean(dim=1, keepdim=True)
        window_std = torch.sqrt(window_var + 1e-8)
        output = F.fold(window_std, output_size=(x.shape[2], x.shape[3]), kernel_size=1)
        return output
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float = 1.0) -> torch.Tensor:
        """Create a Gaussian kernel for smoothing."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g_x = coords.view(-1, 1).repeat(1, kernel_size)
        g_y = coords.view(1, -1).repeat(kernel_size, 1)
        gaussian = torch.exp(-(g_x**2 + g_y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        return gaussian.view(1, 1, kernel_size, kernel_size)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hand-crafted features from input image."""
        batch_size, channels, height, width = x.shape
        features = torch.zeros(batch_size, self.feature_channels, height, width, device=x.device)
        
        if channels == 1:
            input_channel = x.squeeze(1)  
        else:
            input_channel = x[:, 0]  
        
        # Original image
        features[:, 0] = input_channel
        
        # For convolutions, we need to add channel dimension back
        x_single = input_channel.unsqueeze(1)  
        
        # Mean filter
        mean_filtered = F.conv2d(x_single, self.mean_kernel, padding=self.padding)
        features[:, 1] = mean_filtered.squeeze(1)  
        
        # Standard deviation filter
        std_filtered = self._calculate_std_filter(x_single)
        features[:, 2] = std_filtered.squeeze(1)  
        
        # Laplacian (edge detection)
        laplacian_filtered = F.conv2d(x_single, self.laplacian_kernel, padding=1)
        features[:, 3] = laplacian_filtered.squeeze(1) 
        
        # Gradient magnitude (Sobel)
        grad_x = F.conv2d(x_single, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_single, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        features[:, 4] = gradient_magnitude.squeeze(1) 
        
        # Gaussian smoothing
        gaussian_filtered = F.conv2d(x_single, self.gaussian_kernel, padding=self.padding)
        features[:, 5] = gaussian_filtered.squeeze(1)  
        
        return features
    
    def get_learned_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the learned weights and biases as numpy arrays."""
        weights = self.linear.weight.data.cpu().numpy()
        biases = self.linear.bias.data.cpu().numpy()
        return weights, biases
    
    def forward(self, x):
        """Forward pass through the model."""
        features = self.extract_features(x)
        return self.linear(features)