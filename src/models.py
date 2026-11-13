import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import the vendored Oktay attention gate
# ---------------------------------------------------------------------------
def _import_grid_attention():
    gate_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..",
            "third_party", "Attention-Gated-Networks",
            "models", "layers", "grid_attention_layer.py",
        )
    )
    spec = importlib.util.spec_from_file_location("grid_attention_layer", gate_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GridAttentionBlock2D

GridAttentionBlock2D = _import_grid_attention()


def verify_all_trainable(model: nn.Module, model_name: str = "Model") -> None:
    """Every parameter, including BatchNorm affine weights, must be trainable."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
    for name, param in model.named_parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] parameters: total={total_params:,} trainable={trainable_params:,}")
    if total_params != trainable_params:
        raise AssertionError(
            f"{model_name}: frozen parameter detected ({trainable_params} trainable of {total_params})."
        )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            raise AssertionError(f"{model_name}: '{name}' has requires_grad=False")


def assert_imagenet_conv1(model: nn.Module) -> None:
    """Encoder stem must match torchvision ResNet34 IMAGENET1K_V1 conv1."""
    from torchvision.models import ResNet34_Weights, resnet34

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "conv1"):
        raise AssertionError("Model has no encoder.conv1; cannot verify ImageNet initialization.")
    reference = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    loaded = model.encoder.conv1.weight.detach().float().cpu()
    official = reference.conv1.weight.detach().float().cpu()
    if tuple(loaded.shape) != tuple(official.shape):
        raise AssertionError(f"conv1 shape {tuple(loaded.shape)} != ImageNet {tuple(official.shape)}")
    max_abs = (loaded - official).abs().max().item()
    if max_abs > 1e-5:
        raise AssertionError(f"ImageNet conv1 weights were not loaded (max abs diff {max_abs:.3e}).")
    print(f"[ImageNet] encoder.conv1 matches torchvision resnet34 IMAGENET1K_V1 (max abs diff {max_abs:.3e}).")


def assert_optimizer_covers_all(model: nn.Module, param_groups: list) -> None:
    """Each parameter is in exactly one optimizer group."""
    model_ids = [id(p) for p in model.parameters()]
    if len(model_ids) != len(set(model_ids)):
        raise AssertionError("Model listed the same parameter object more than once.")
    grouped = []
    for group in param_groups:
        for param in group["params"]:
            grouped.append(id(param))
    if len(grouped) != len(set(grouped)):
        raise AssertionError("A parameter is in more than one optimizer group.")
    if set(grouped) != set(model_ids):
        missing = len(set(model_ids) - set(grouped))
        extra = len(set(grouped) - set(model_ids))
        raise AssertionError(f"Optimizer coverage mismatch (missing={missing}, extra={extra}).")


def count_parameter_groups(model: nn.Module) -> dict:
    encoder = 0
    decoder = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if name.startswith("encoder.") or name == "encoder":
            encoder += n
        else:
            decoder += n
    total = encoder + decoder
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "encoder": encoder,
        "decoder_head": decoder,
    }


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def segmentation_head(in_channels: int, classes: int) -> nn.Module:
    """Same module smp.Unet uses: 3x3 conv, padding 1, no activation (logits)."""
    try:
        from segmentation_models_pytorch.base import SegmentationHead
    except ImportError:
        from segmentation_models_pytorch.base.heads import SegmentationHead

    return SegmentationHead(
        in_channels=in_channels,
        out_channels=classes,
        kernel_size=3,
        activation=None,
        upsampling=1,
    )


class AttentionUNet(nn.Module):
    """ResNet34 U-Net whose skip tensors are filtered by Oktay grid attention gates.

    Gated skips are encoder stages x1 (stem), x2, x3, and x4. The input image is
    not a gated skip. The bottleneck x5 is only a gating signal.
    """

    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        _c0, c1, c2, c3, c4, c5 = self.encoder.out_channels

        self.gate4 = GridAttentionBlock2D(in_channels=c4, gating_channels=c5, inter_channels=c4 // 2)
        self.dec4 = DecoderBlock(in_channels=c5, skip_channels=c4, out_channels=256)
        self.gate3 = GridAttentionBlock2D(in_channels=c3, gating_channels=256, inter_channels=c3 // 2)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=c3, out_channels=128)
        self.gate2 = GridAttentionBlock2D(in_channels=c2, gating_channels=128, inter_channels=c2 // 2)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=c2, out_channels=64)
        self.gate1 = GridAttentionBlock2D(in_channels=c1, gating_channels=64, inter_channels=max(c1 // 2, 1))
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=c1, out_channels=32)
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # smp.Unet decoder ends at 16 channels and uses a 3x3 segmentation head.
        self.segmentation_head = segmentation_head(16, classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        _x0, x1, x2, x3, x4, x5 = self.encoder(x)

        def _gate(gate, skip, signal):
            if return_attention:
                return gate(skip, signal, return_attention=True)
            return gate(skip, signal), None

        gated_x4, a4 = _gate(self.gate4, x4, x5)
        d4 = self.dec4(x5, gated_x4)
        gated_x3, a3 = _gate(self.gate3, x3, d4)
        d3 = self.dec3(d4, gated_x3)
        gated_x2, a2 = _gate(self.gate2, x2, d3)
        d2 = self.dec2(d3, gated_x2)
        gated_x1, a1 = _gate(self.gate1, x1, d2)
        d1 = self.dec1(d2, gated_x1)
        logits = self.segmentation_head(self.dec0(d1))
        if return_attention:
            return logits, {"gate1": a1, "gate2": a2, "gate3": a3, "gate4": a4}
        return logits


def build_model(model_name: str, backbone: str = "resnet34", encoder_weights: str = "imagenet", in_channels: int = 3, classes: int = 1) -> nn.Module:
    import segmentation_models_pytorch as smp

    name = model_name.lower().strip()
    if name == "unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif name in ("deeplabv3", "deeplab"):
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif name in ("attunet", "attention_unet", "attentionunet"):
        model = AttentionUNet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose unet, attunet, or deeplabv3.")

    verify_all_trainable(model, model_name=f"{name}_{backbone}")
    if encoder_weights:
        assert_imagenet_conv1(model)
    return model


def get_optimizer_param_groups(model: nn.Module, encoder_lr: float = 1e-4, decoder_lr: float = 1e-3, weight_decay: float = 1e-4):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            raise RuntimeError(f"Parameter '{name}' is not trainable.")
        if name.startswith("encoder.") or name == "encoder":
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    param_groups = [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay, "name": "encoder"},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder_head"},
    ]
    assert_optimizer_covers_all(model, param_groups)
    return param_groups


def export_parameter_counts(path: str = "results/params.csv", encoder_weights: str = "imagenet") -> None:
    """Instantiate each model and write real parameter counts."""
    import pandas as pd

    rows = []
    for model_name in ("unet", "attunet", "deeplabv3"):
        model = build_model(model_name, backbone="resnet34", encoder_weights=encoder_weights, in_channels=3, classes=1)
        counts = count_parameter_groups(model)
        print(
            f"{model_name}: total={counts['total']:,} trainable={counts['trainable']:,} "
            f"encoder={counts['encoder']:,} decoder_head={counts['decoder_head']:,}"
        )
        rows.append({"model": model_name, **counts})
        del model
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write results/params.csv from instantiated models")
    parser.add_argument("--write-params", action="store_true")
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument("--out", default="results/params.csv")
    args = parser.parse_args()
    export_parameter_counts(path=args.out, encoder_weights=None if args.weights in ("", "none") else args.weights)
