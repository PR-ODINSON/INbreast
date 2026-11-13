import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation with logits input.
    """
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        cardinality = probs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss:
    Loss = dice_weight * DiceLoss + bce_weight * BCELoss
    """
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets.float())
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme foreground/background class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss: balances precision and recall via alpha (false positives weight)
    and beta (false negatives weight).
    When alpha = beta = 0.5, equivalent to Dice loss.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        true_pos = (probs * targets).sum()
        false_neg = ((1.0 - probs) * targets).sum()
        false_pos = (probs * (1.0 - targets)).sum()

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return 1.0 - tversky


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function returning the configured loss criterion.
    """
    train_cfg = config.get("training", {})
    loss_type = train_cfg.get("loss_type", "dice_bce").lower()

    if loss_type == "dice_bce":
        dice_w = train_cfg.get("dice_weight", 1.0)
        bce_w = train_cfg.get("bce_weight", 1.0)
        return DiceBCELoss(dice_weight=dice_w, bce_weight=bce_w)
    elif loss_type == "focal":
        alpha = train_cfg.get("focal_alpha", 0.25)
        gamma = train_cfg.get("focal_gamma", 2.0)
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "tversky":
        alpha = train_cfg.get("tversky_alpha", 0.7)
        beta = train_cfg.get("tversky_beta", 0.3)
        return TverskyLoss(alpha=alpha, beta=beta)
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'. Supported: 'dice_bce', 'focal', 'tversky', 'dice', 'bce'.")

