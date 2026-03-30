import torch
import matplotlib.pyplot as plt

from datasets.sidd_dataset import SIDDDataset
from models.simple_cnn import SimpleDenoiser
from utils.metrics import calculate_psnr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SIDDDataset(
        root_dir="data/sidd/Data",
        patch_size=128
    )

    noisy_img, clean_img = dataset[0]

    model = SimpleDenoiser().to(device)
    model.load_state_dict(torch.load("simple_denoiser_sidd.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        input_tensor = noisy_img.unsqueeze(0).to(device)
        output = model(input_tensor)
        denoised_img = torch.clamp(output.squeeze(0).cpu(), 0, 1)

    noisy_psnr = calculate_psnr(noisy_img, clean_img)
    denoised_psnr = calculate_psnr(denoised_img, clean_img)

    print(f"Noisy PSNR: {noisy_psnr:.2f} dB")
    print(f"Denoised PSNR: {denoised_psnr:.2f} dB")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(clean_img.permute(1, 2, 0))
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img.permute(1, 2, 0))
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_img.permute(1, 2, 0))
    plt.title("Denoised")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()