import torch
import math


def calculate_psnr(img1, img2):
    """
    img1, img2: torch tensors with range [0, 1]
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    psnr = 10 * math.log10(1.0 / mse)
    return psnr