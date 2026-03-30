import torch
from torch.utils.data import DataLoader

from datasets.synthetic_dataset import SyntheticDenoiseDataset
from models.simple_cnn import SimpleDenoiser
from utils.metrics import calculate_psnr


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SyntheticDenoiseDataset(
        image_dir="data/Berkeley Segmentation Dataset 500 (BSDS500)/images/val",
        sigma=25,
        patch_size=128
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = SimpleDenoiser().to(device)
    model.load_state_dict(torch.load("simple_denoiser.pth", map_location=device))
    model.eval()

    total_noisy_psnr = 0.0
    total_denoised_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(noisy_imgs)
            outputs = torch.clamp(outputs, 0, 1)

            noisy_img = noisy_imgs.squeeze(0).cpu()
            clean_img = clean_imgs.squeeze(0).cpu()
            denoised_img = outputs.squeeze(0).cpu()

            noisy_psnr = calculate_psnr(noisy_img, clean_img)
            denoised_psnr = calculate_psnr(denoised_img, clean_img)

            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            num_samples += 1

    avg_noisy_psnr = total_noisy_psnr / num_samples
    avg_denoised_psnr = total_denoised_psnr / num_samples

    print(f"Number of validation samples: {num_samples}")
    print(f"Average Noisy PSNR: {avg_noisy_psnr:.2f} dB")
    print(f"Average Denoised PSNR: {avg_denoised_psnr:.2f} dB")


if __name__ == "__main__":
    evaluate()