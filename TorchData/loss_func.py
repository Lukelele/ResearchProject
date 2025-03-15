import torch
import torch.nn.functional as F
from math import exp
from .metric import calculate_ssim_torch

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=(11, 11)):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return 1 - calculate_ssim_torch(img1, img2, self.window_size)
    


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, alpha=1, beta=1):
        """
        Weighted Mean Squared Error Loss that assigns different weights to different target values.
        
        Args:
            signal_weight (float): Weight for positive signals (where target > 0).
                                  1-signal_weight will be applied to zero values.
        """
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, output, target):
        """
        Calculate the weighted MSE loss.
        
        Args:
            output (torch.Tensor): Model predictions
            target (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Weighted MSE loss value
        """

        weights = torch.ones_like(target)
        weights[target > 0] = self.alpha  # Apply signal_weight to positive targets
        weights[target == 0] = self.beta
        
        return torch.mean(weights * (output - target) ** 2)
