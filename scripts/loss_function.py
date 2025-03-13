import torch


class WeightedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, fp_weight: float = 10.0, fn_weight: float = 1.0):
        super().__init__()
        self.fp_weight = fp_weight  # Weight for False Positives
        self.fn_weight = fn_weight  # Weight for False Negatives
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # Compute per-sample loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        my_loss = self.bce(logits, targets)  # Compute base BCE loss

        # Compute FP and FN masks
        fp_mask = (targets == 0).float()  # False positives (when target is 0)
        fn_mask = (targets == 1).float()  # False negatives (when target is 1)

        # Apply different weights
        weighted_loss = (self.fp_weight * fp_mask + self.fn_weight * fn_mask) * my_loss

        return weighted_loss.mean()  # Return mean weighted loss
