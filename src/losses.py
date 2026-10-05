import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=5,
        ignore_index=255,
        smooth=1.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = F.softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # Valid pixels
        # ----------------------------------------------------

        valid_mask = (
            targets != self.ignore_index
        )

        safe_targets = targets.clone()

        safe_targets[
            ~valid_mask
        ] = 0

        # ----------------------------------------------------
        # One-hot targets
        # ----------------------------------------------------

        target_one_hot = F.one_hot(
            safe_targets,
            num_classes=self.num_classes
        )

        target_one_hot = target_one_hot.permute(
            0,
            3,
            1,
            2
        ).float()

        # ----------------------------------------------------
        # Apply valid mask
        # ----------------------------------------------------

        valid_mask = (
            valid_mask
            .unsqueeze(1)
            .float()
        )

        probabilities = (
            probabilities *
            valid_mask
        )

        target_one_hot = (
            target_one_hot *
            valid_mask
        )

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            self.num_classes,
            -1
        )

        target_one_hot = target_one_hot.reshape(
            target_one_hot.shape[0],
            self.num_classes,
            -1
        )

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        intersection = (
            probabilities *
            target_one_hot
        ).sum(dim=2)

        denominator = (
            probabilities.sum(dim=2)
            +
            target_one_hot.sum(dim=2)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Ignore classes that do not exist in the target
        # ----------------------------------------------------

        class_present = (
            target_one_hot.sum(dim=2) > 0
        )

        dice = dice[class_present]

        if dice.numel() == 0:
            return logits.sum() * 0.0

        dice_loss = (
            1.0 -
            dice.mean()
        )

        return dice_loss


# ============================================================
# Combined Loss
#
# Weighted Cross Entropy + Dice
#
# This is intentionally kept simple.
# We do NOT add another focal-loss experiment here.
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        num_classes=5,
        ignore_index=255,
        ce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # ----------------------------------------------------
        # Weighted Cross Entropy
        # ----------------------------------------------------

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index
        )

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = self.cross_entropy(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (
            self.ce_weight * ce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss