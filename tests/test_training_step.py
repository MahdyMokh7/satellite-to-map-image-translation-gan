import torch
import os
from torch import nn, optim
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from models.generator import GeneratorUNet
from utils.dataloader import get_dataloader, transform

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small batch on purpose
    data_dir = os.path.join(root_dir, 'data', 'processed')
    loader = get_dataloader(
        data_dir=data_dir,
        split='val',
        batch_size=2,
        shuffle=False,
        transform=transform
    )

    sat_images, gt_maps = next(iter(loader))
    sat_images = sat_images.to(device)
    gt_maps = gt_maps.to(device)

    generator = GeneratorUNet().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.L1Loss()

    generator.train()

    fake_maps = generator(sat_images)
    loss = criterion(fake_maps, gt_maps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "❌ Loss is NaN"
    print("✅ Training step sanity check passed")
    print(f"Loss value: {loss.item():.4f}")

if __name__ == "__main__":
    main()
