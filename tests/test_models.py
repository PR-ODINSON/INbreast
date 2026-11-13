import os
import sys
try:
    import pytest
except ImportError:
    class DummyPytest:
        class mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(fn):
                    return fn
                return decorator
    pytest = DummyPytest()
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_model, get_optimizer_param_groups
from src.losses import DiceBCELoss


@pytest.mark.parametrize("model_name", ["unet", "attunet", "deeplabv3"])
def test_hard_rule_1_all_parameters_trainable(model_name):
    """
    Hard Rule 1 Test:
    Asserts DO NOT freeze any weights. Every parameter in every model
    (encoder, decoder, attention gates, head) must have requires_grad=True
    and be in the optimizer. Fails if total_params != trainable_params.
    """
    # Build model (weights="imagenet" or None for quick offline test)
    try:
        model = build_model(model_name, backbone="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    except Exception:
        # If offline/no internet to download weights in test environment, test with weights=None
        model = build_model(model_name, backbone="resnet34", encoder_weights=None, in_channels=3, classes=1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[{model_name.upper()}] Total: {total_params:,} | Trainable: {trainable_params:,}")
    assert total_params == trainable_params, (
        f"HARD RULE 1 VIOLATION: {model_name} has frozen weights! "
        f"Total={total_params}, Trainable={trainable_params}"
    )

    # Check every individual parameter
    for name, param in model.named_parameters():
        assert param.requires_grad is True, f"Parameter '{name}' has requires_grad=False!"

    # Check optimizer param groups include all parameters
    param_groups = get_optimizer_param_groups(model, encoder_lr=1e-4, decoder_lr=1e-3)
    opt_params_count = sum(sum(p.numel() for p in grp["params"]) for grp in param_groups)
    assert opt_params_count == total_params, (
        f"Optimizer parameter mismatch: {opt_params_count} in optimizer vs {total_params} in model!"
    )


@pytest.mark.parametrize("model_name", ["unet", "attunet", "deeplabv3"])
def test_forward_backward_pass(model_name):
    """
    Verifies output shape (B, 1, H, W) and confirms forward + backward gradient flow.
    """
    try:
        model = build_model(model_name, backbone="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    except Exception:
        model = build_model(model_name, backbone="resnet34", encoder_weights=None, in_channels=3, classes=1)

    model.train()
    B, C, H, W = 2, 3, 256, 256  # 256x256 for fast unit testing
    dummy_input = torch.randn(B, C, H, W)
    dummy_target = torch.randint(0, 2, (B, 1, H, W)).float()

    # Forward pass
    output = model(dummy_input)
    assert output.shape == (B, 1, H, W), f"Expected shape {(B, 1, H, W)}, got {output.shape}"

    # Backward pass
    criterion = DiceBCELoss()
    loss = criterion(output, dummy_target)
    loss.backward()

    # Verify gradients exist and are non-null
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for '{name}' is None after backward pass!"
        assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaN!"

    print(f"\n[PASS] Forward and backward pass verified for {model_name.upper()}.")


def test_attention_unet_attention_maps():
    """
    Verifies that Attention U-Net can extract attention maps from its attention gates.
    """
    try:
        model = build_model("attunet", backbone="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    except Exception:
        model = build_model("attunet", backbone="resnet34", encoder_weights=None, in_channels=3, classes=1)

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits, att_maps = model(dummy_input, return_attention=True)

    assert logits.shape == (1, 1, 256, 256)
    for gname in ["gate1", "gate2", "gate3", "gate4"]:
        assert gname in att_maps
        alpha = att_maps[gname]
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0, f"Attention map {gname} out of bounds [0, 1]!"

    print("\n[PASS] Attention coefficient maps verified for Attention U-Net.")


if __name__ == "__main__":
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("[INFO] segmentation_models_pytorch is not installed in this environment.")
        print("[INFO] Model tests will run on your server after `pip install -r requirements.txt`.")
        sys.exit(0)

    for m in ["unet", "attunet", "deeplabv3"]:
        test_hard_rule_1_all_parameters_trainable(m)
        test_forward_backward_pass(m)
    test_attention_unet_attention_maps()
