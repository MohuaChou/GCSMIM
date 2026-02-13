import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSegLoss(nn.Module):
    """Dice + Cross Entropy"""
    def __init__(self, num_classes, weights=(0.5, 0.5)):
        super().__init__()
        self.dice = DiceLoss3D(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.weights = weights

    def forward(self, preds, targets):
        # targets: accept [B,1,D,H,W] or [B,D,H,W]
        if targets.ndim == 5 and targets.size(1) == 1:
            targets_ce = targets.squeeze(1)
        else:
            targets_ce = targets

        dice_loss = self.dice(preds, targets)
        ce_loss = self.ce(preds, targets_ce.long())
        return self.weights[0] * dice_loss + self.weights[1] * ce_loss


class DiceLoss3D(nn.Module):
    """3D Dice Loss"""
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        pred: [B,C,D,H,W]
        target: [B,1,D,H,W] or [B,D,H,W]
        """
        pred = torch.softmax(pred, dim=1)

        if target.ndim == 5 and target.size(1) == 1:
            target = target.squeeze(1)  # [B,D,H,W]

        target_onehot = F.one_hot(target.long(), self.num_classes)  # [B,D,H,W,C]
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

        intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
