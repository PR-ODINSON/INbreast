# Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images
# Original implementation from: https://github.com/ozan-oktay/Attention-Gated-Networks
# Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridAttentionBlock2D(nn.Module):
    """
    2D Grid Attention Block as implemented in Oktay et al. (Attention-Gated-Networks).
    
    W-path: Linear transformation of attended features (1x1 conv + BatchNorm)
    Theta-path: Projection of skip feature map x (downsampled)
    Phi-path: Projection of gating signal g from coarser decoder level
    Psi-path: Attention coefficient generation (1x1 conv -> Sigmoid) followed by upsampling
    """
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int = None, sub_sample_factor: tuple = (2, 2)):
        super().__init__()
        
        # Default intermediate channel reduction
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        self.sub_sample_factor = sub_sample_factor

        # Theta path: feature transformation (stride subsampling)
        self.theta = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=sub_sample_factor,
            stride=sub_sample_factor,
            padding=0,
            bias=False
        )

        # Phi path: gating signal transformation
        self.phi = nn.Conv2d(
            in_channels=gating_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # Psi path: attention coefficient projection
        self.psi = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # W path: linear transformation after attention weighting
        self.W = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(in_channels)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, g: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: skip-connection tensor from encoder, shape (B, in_channels, H, W)
            g: gating signal tensor from coarser decoder stage, shape (B, gating_channels, H_g, W_g)
            return_attention: if True, returns (output, attention_map)
        Returns:
            attended_x: attention-weighted skip features, shape (B, in_channels, H, W)
            (optional) alpha: attention coefficient map, shape (B, 1, H, W)
        """
        input_size = x.shape[2:]  # (H, W)

        # 1. Project skip features x through theta
        theta_x = self.theta(x)
        theta_x_size = theta_x.shape[2:]

        # 2. Project gating signal g through phi and interpolate to theta_x size
        phi_g = self.phi(g)
        phi_g = F.interpolate(phi_g, size=theta_x_size, mode='bilinear', align_corners=True)

        # 3. Additive attention + non-linear activation (ReLU)
        f = F.relu(theta_x + phi_g, inplace=True)

        # 4. Generate attention coefficient map
        psi_f = self.psi(f)
        alpha = torch.sigmoid(psi_f)

        # 5. Upsample attention map to original skip feature resolution (H, W)
        alpha = F.interpolate(alpha, size=input_size, mode='bilinear', align_corners=True)

        # 6. Apply attention weighting and linear W transform
        y = alpha * x
        output = self.W(y)

        if return_attention:
            return output, alpha
        return output

