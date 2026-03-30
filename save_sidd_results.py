import os
import torch
import matplotlib.pyplot as plt

from datasets.sidd_dataset import SIDDDataset
from models.simple_cnn import SimpleDenoiser


def save_sidd_results(num_examples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = "results/sidd_examples"
    os.makedirs(save_dir, exist_ok=True)

    dataset = SIDDDataset(
        root_dir="data/sidd/Data",
        patch_size=128
    )

    model = SimpleDenoiser().to(device)
    model.load_state_dict(torch.load("simple_denoiser_sidd.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        for i in range(num_examples):
            noisy_img, clean_img = dataset[i]

            input_tensor = noisy_img.unsqueeze(0).to(device)
            output = model(input_tensor)
            denoised_img = torch.clamp(output.squeeze(0).cpu(), 0, 1)

            plt.figure(figsize=(12, 4))

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

            save_path = os.path.join(save_dir, f"sidd_result_{i+1}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    save_sidd_results(num_examples=10)