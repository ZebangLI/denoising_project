import torch
import matplotlib.pyplot as plt
from utils.metrics import calculate_psnr
from datasets.synthetic_dataset import SyntheticDenoiseDataset
from models.simple_cnn import SimpleDenoiser


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SyntheticDenoiseDataset(
        image_dir="data/Berkeley Segmentation Dataset 500 (BSDS500)/images/train",
        sigma=25,
        patch_size=128
    )

    noisy_img, clean_img = dataset[0]

    model = SimpleDenoiser().to(device)
    model.load_state_dict(torch.load("simple_denoiser.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        input_tensor = noisy_img.unsqueeze(0).to(device)   # [1, 3, 128, 128]
        output = model(input_tensor)
        denoised_img = output.squeeze(0).cpu()            # [3, 128, 128]

    noisy_psnr = calculate_psnr(noisy_img, clean_img)
    denoised_psnr = calculate_psnr(torch.clamp(denoised_img, 0, 1), clean_img)

    print(f"Noisy PSNR: {noisy_psnr:.2f} dB")
    print(f"Denoised PSNR: {denoised_psnr:.2f} dB")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(clean_img.permute(1, 2, 0))
    plt.title("Clean")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img.permute(1, 2, 0))
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(torch.clamp(denoised_img, 0, 1).permute(1, 2, 0))
    plt.title("Denoised")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()