import torch


def add_gaussian_noise(img, sigma=25):
    """
    img: torch tensor, shape [C, H, W], value range [0, 1]
    sigma: noise level in 0-255 scale
    """
    noise = torch.randn_like(img) * (sigma / 255.0)
    noisy_img = img + noise
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
    return noisy_img