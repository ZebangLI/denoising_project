import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.synthetic_dataset import SyntheticDenoiseDataset
from models.simple_cnn import SimpleDenoiser


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SyntheticDenoiseDataset(
        image_dir="data/Berkeley Segmentation Dataset 500 (BSDS500)/images/train",
        sigma=25,
        patch_size=128
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = SimpleDenoiser().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "simple_denoiser.pth")
    print("Training finished. Model saved as simple_denoiser.pth")


if __name__ == "__main__":
    train()