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