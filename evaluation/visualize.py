import torch
import os
import sys

# --- Project setup ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import yaml
import matplotlib.pyplot as plt
from models.generator import GeneratorUNet
from utils.dataloader import get_dataloader, transform

# --- Load config ---
config_file_path = os.path.join(root_dir, 'configs', 'config.yaml')
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

training_config = config["training"]
dataset_config = config["dataset"]
generator_config = config["generator"]
logging_config = config["outputs"]



# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = GeneratorUNet(
    in_channels=generator_config['in_channels'],
    out_channels=generator_config['out_channels'],
    base_filters=generator_config['base_filters']
).to(device)

checkpoint_path = os.path.join(root_dir, 'results', 'checkpoints', 'cgan', 'generator_best.pth')

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

data_dir = os.path.join(root_dir, *dataset_config['paths']["data_dir"].split('/'))

# --- DataLoader ---
test_loader = get_dataloader(
    data_dir=data_dir,
    split='test',
    batch_size=1,
    shuffle=False,
    transform=transform
)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

# --- Visualization ---
with torch.no_grad():
    for i, (satellite, real_map) in enumerate(test_loader):
        satellite = satellite.to(device)
        real_map = real_map.to(device)

        fake_map = model(satellite)

        # Convert tensors to numpy (C,H,W → H,W,C)
        satellite_np = denormalize(satellite.squeeze(0)).cpu().permute(1, 2, 0).numpy()
        fake_map_np = denormalize(fake_map.squeeze(0)).cpu().permute(1, 2, 0).numpy()
        real_map_np = denormalize(real_map.squeeze(0)).cpu().permute(1, 2, 0).numpy()


        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(satellite_np)
        ax[0].set_title("Satellite")
        ax[0].axis("off")

        ax[1].imshow(fake_map_np)
        ax[1].set_title("Generated Map")
        ax[1].axis("off")

        ax[2].imshow(real_map_np)
        ax[2].set_title("Ground Truth")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

        if i >= 4:  # visualize first 5 samples
            break
