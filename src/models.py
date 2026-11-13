import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure third_party attention gate can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from third_party.attention_gate import GridAttentionBlock2D
except ImportError:
    from third_party.Attention_Gated_Networks.models.layers.grid_attention_layer import GridAttentionBlock2D


def verify_all_trainable(model: nn.Module, model_name: str = "Model") -> None:
    """
    Hard Rule 1 Enforcement:
    Every parameter in every model (encoder, decoder, attention gates, head)
    must have requires_grad=True and be in the optimizer.
    Fails if total_params != trainable_params.
    """
    # Ensure all parameters have requires_grad explicitly set to True
    for name, param in model.named_parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[{model_name}] Parameter Verification: Total = {total_params:,} | Trainable = {trainable_params:,} (100% Unfrozen)")
    assert total_params == trainable_params, (
        f"Hard Rule 1 Violation! In {model_name}: total params ({total_params}) != "
        f"trainable params ({trainable_params}). Frozen weights detected!"
    )
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' in {model_name} has requires_grad=False!"


class DecoderBlock(nn.Module):
    """
    Standard double-convolution block with Upsampling for U-Net decoders.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # Bilinear upsampling to match skip resolution
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net with ResNet34 ImageNet-pretrained encoder and Oktay et al. Grid Attention Gates.
    The attention gates filter features from the encoder skip connections using the gating signal
    from the coarser decoder stage before concatenation.
    """
    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1):
        super().__init__()
        import segmentation_models_pytorch as smp

        # 1. ImageNet-pretrained encoder from SMP
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )

        # ResNet34 channels: [3, 64, 64, 128, 256, 512]
        encoder_channels = self.encoder.out_channels
        c0, c1, c2, c3, c4, c5 = encoder_channels

        # 2. Vendored Oktay et al. Grid Attention Blocks on skip connections
        # Skip 4: x4 (256 ch) gated by bottleneck x5 (512 ch)
        self.gate4 = GridAttentionBlock2D(in_channels=c4, gating_channels=c5, inter_channels=c4 // 2)
        self.dec4 = DecoderBlock(in_channels=c5, skip_channels=c4, out_channels=256)

        # Skip 3: x3 (128 ch) gated by dec4 output (256 ch)
        self.gate3 = GridAttentionBlock2D(in_channels=c3, gating_channels=256, inter_channels=c3 // 2)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=c3, out_channels=128)

        # Skip 2: x2 (64 ch) gated by dec3 output (128 ch)
        self.gate2 = GridAttentionBlock2D(in_channels=c2, gating_channels=128, inter_channels=c2 // 2)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=c2, out_channels=64)

        # Skip 1: x1 (64 ch) gated by dec2 output (64 ch)
        self.gate1 = GridAttentionBlock2D(in_channels=c1, gating_channels=64, inter_channels=c1 // 2)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=c1, out_channels=32)

        # Final upsampling to original input resolution (512x512)
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Encoder feature hierarchy
        features = self.encoder(x)
        # x0: (B, 3, 512, 512)
        # x1: (B, 64, 256, 256)
        # x2: (B, 64, 128, 128)
        # x3: (B, 128, 64, 64)
        # x4: (B, 256, 32, 32)
        # x5: (B, 512, 16, 16)
        x0, x1, x2, x3, x4, x5 = features

        # Stage 4: Gate skip4 with bottleneck x5
        if return_attention:
            gated_x4, a4 = self.gate4(x4, x5, return_attention=True)
        else:
            gated_x4 = self.gate4(x4, x5)
        d4 = self.dec4(x5, gated_x4)

        # Stage 3: Gate skip3 with d4
        if return_attention:
            gated_x3, a3 = self.gate3(x3, d4, return_attention=True)
        else:
            gated_x3 = self.gate3(x3, d4)
        d3 = self.dec3(d4, gated_x3)

        # Stage 2: Gate skip2 with d3
        if return_attention:
            gated_x2, a2 = self.gate2(x2, d3, return_attention=True)
        else:
            gated_x2 = self.gate2(x2, d3)
        d2 = self.dec2(d3, gated_x2)

        # Stage 1: Gate skip1 with d2
        if return_attention:
            gated_x1, a1 = self.gate1(x1, d2, return_attention=True)
        else:
            gated_x1 = self.gate1(x1, d2)
        d1 = self.dec1(d2, gated_x1)

        # Stage 0: Final upsample to original resolution
        d0 = self.dec0(d1)
        logits = self.segmentation_head(d0)

        if return_attention:
            return logits, {"gate1": a1, "gate2": a2, "gate3": a3, "gate4": a4}
        return logits


def build_model(model_name: str, backbone: str = "resnet34", encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1) -> nn.Module:
    """
    Factory function to build and verify the three specified segmentation models:
    - unet: smp.Unet(encoder_name=backbone, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
    - deeplabv3: smp.DeepLabV3(encoder_name=backbone, encoder_weights=encoder_weights, classes=classes)
    - attunet: AttentionUNet with ResNet34 encoder and Oktay et al. Grid Attention Gates
    """
    import segmentation_models_pytorch as smp

    name = model_name.lower().strip()
    if name == "unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif name in ["deeplabv3", "deeplab"]:
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif name in ["attunet", "attention_unet", "attentionunet"]:
        model = AttentionUNet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from 'unet', 'attunet', 'deeplabv3'.")

    # Hard Rule 1 Enforcement: verify 100% of parameters are unfrozen and trainable
    verify_all_trainable(model, model_name=f"{name.upper()}_{backbone}")
    return model


def get_optimizer_param_groups(model: nn.Module, encoder_lr: float = 1e-4, decoder_lr: float = 1e-3, weight_decay: float = 1e-4):
    """
    Splits model parameters into differential learning rate groups:
    - Pretrained encoder: lower learning rate (e.g. 1e-4)
    - Decoder, attention gates, and head: higher learning rate (e.g. 1e-3)
    Both groups have requires_grad=True and are passed to AdamW.
    """
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            raise RuntimeError(f"Hard Rule 1 violation: parameter '{name}' is not trainable!")
        
        # Categorize into encoder vs rest (decoder, gates, head)
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay, "name": "encoder"},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder_head"}
    ]
    return param_groups

