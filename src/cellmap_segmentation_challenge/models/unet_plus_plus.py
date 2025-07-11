""" 
3D UNet++ Implementation
2D Algorithm from: https://github.com/MrGiovanni/UNetPlusPlus
Implemented as 3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm3d(mid_channels),
            nn.InstanceNorm3d(mid_channels), # Changed to InstanceNorm since batch size is 1
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.InstanceNorm3d(out_channels), # Changed to InstanceNorm since batch size is 1
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class NestedUpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.reduce_channels = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
            total_in_channels = (in_channels // 2) + skip_channels
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce_channels = None
            # Total channels after concatenation: transposed + all skip connections  
            total_in_channels = (in_channels // 2) + skip_channels
            
        self.conv = DoubleConv3D(total_in_channels, out_channels)

    def forward(self, x_up, skip_connections):
        x_up = self.up(x_up)
        if self.reduce_channels is not None:
            x_up = self.reduce_channels(x_up)

        if skip_connections:
            target = skip_connections[0]
            x_up = self._match_size(x_up, target)
        
        if skip_connections:
            x = torch.cat([x_up] + skip_connections, dim=1)
        else:
            x = x_up
            
        return self.conv(x)
    
    def _match_size(self, x_up, target):
        diffD = target.size()[2] - x_up.size()[2]  # depth
        diffH = target.size()[3] - x_up.size()[3]  # height  
        diffW = target.size()[4] - x_up.size()[4]  # width
        
        x_up = F.pad(x_up, [
            diffW // 2, diffW - diffW // 2,
            diffH // 2, diffH - diffH // 2, 
            diffD // 2, diffD - diffD // 2
        ])
        return x_up


class UNet3DPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, trilinear=True):
        super(UNet3DPlusPlus, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.pool = nn.MaxPool3d(2)
        self.conv0_0 = DoubleConv3D(n_channels, 32)      # X^0,0
        self.conv1_0 = DoubleConv3D(32, 64)              # X^1,0  
        self.conv2_0 = DoubleConv3D(64, 128)             # X^2,0
        self.conv3_0 = DoubleConv3D(128, 256)            # X^3,0
        self.conv4_0 = DoubleConv3D(256, 512)            # X^4,0 (bottleneck)
        
        # Level 1 nested blocks
        self.conv0_1 = NestedUpBlock3D(64, 32, 32, trilinear)        # X^0,1
        self.conv1_1 = NestedUpBlock3D(128, 64, 64, trilinear)       # X^1,1
        self.conv2_1 = NestedUpBlock3D(256, 128, 128, trilinear)     # X^2,1
        self.conv3_1 = NestedUpBlock3D(512, 256, 256, trilinear)     # X^3,1
        
        # Level 2 nested blocks  
        self.conv0_2 = NestedUpBlock3D(64, 64, 32, trilinear)        # X^0,2
        self.conv1_2 = NestedUpBlock3D(128, 128, 64, trilinear)      # X^1,2
        self.conv2_2 = NestedUpBlock3D(256, 256, 128, trilinear)     # X^2,2
        
        # Level 3 nested blocks
        self.conv0_3 = NestedUpBlock3D(64, 96, 32, trilinear)        # X^0,3
        self.conv1_3 = NestedUpBlock3D(128, 192, 64, trilinear)      # X^1,3
        
        # Level 4 nested block (final)
        self.conv0_4 = NestedUpBlock3D(64, 128, 32, trilinear)       # X^0,4
        
        # Output layers for deep supervision
        if self.deep_supervision:
            self.out1 = nn.Conv3d(32, n_classes, kernel_size=1)  # From X^0,1
            self.out2 = nn.Conv3d(32, n_classes, kernel_size=1)  # From X^0,2  
            self.out3 = nn.Conv3d(32, n_classes, kernel_size=1)  # From X^0,3
            self.out4 = nn.Conv3d(32, n_classes, kernel_size=1)  # From X^0,4
        else:
            self.final_conv = nn.Conv3d(32, n_classes, kernel_size=1)  # Only final output

    def forward(self, x):
        x0_0 = self.conv0_0(x)           
        x1_0 = self.conv1_0(self.pool(x0_0))  
        x2_0 = self.conv2_0(self.pool(x1_0)) 
        x3_0 = self.conv3_0(self.pool(x2_0))  
        x4_0 = self.conv4_0(self.pool(x3_0))  
        
        # Column 1
        x3_1 = self.conv3_1(x4_0, [x3_0])                    
        x2_1 = self.conv2_1(x3_1, [x2_0])                    
        x1_1 = self.conv1_1(x2_1, [x1_0])                  
        x0_1 = self.conv0_1(x1_1, [x0_0])                  
        
        # Column 2
        x2_2 = self.conv2_2(x3_1, [x2_0, x2_1])             
        x1_2 = self.conv1_2(x2_2, [x1_0, x1_1])            
        x0_2 = self.conv0_2(x1_2, [x0_0, x0_1])            
        
        # Column 3
        x1_3 = self.conv1_3(x2_2, [x1_0, x1_1, x1_2])        
        x0_3 = self.conv0_3(x1_3, [x0_0, x0_1, x0_2])       
        
        # Column 4
        x0_4 = self.conv0_4(x1_3, [x0_0, x0_1, x0_2, x0_3])  
        
        # Output handling
        if self.deep_supervision:    
            output1 = self.out1(x0_1)
            output2 = self.out2(x0_2) 
            output3 = self.out3(x0_3)
            output4 = self.out4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final_conv(x0_4)

