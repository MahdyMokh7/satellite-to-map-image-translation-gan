import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml
from models.generator import GeneratorUNet
from utils.dataloader import SatelliteMapDataset  # <- Dataloader import
from utils.helpers import calculate_psnr, calculate_ssim  # <- Helpers import
from utils.logger import setup_logger, save_checkpoint, save_samples  # <- Logger import

# Load configuration file
config_file_path = os.path.join(root_dir, 'configs', 'config.yaml')
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

training_config = config["training"]
dataset_config = config["dataset"]
generator_config = config["generator"]
loss_config = config["loss"]
logging_config = config["outputs"]

batch_size = training_config["batch_size"]
epochs = training_config["epochs"]
lr = training_config["optimizer"]["lr"]
beta1 = training_config["optimizer"]["beta1"]
device = torch.device(config["project"]["device"] if torch.cuda.is_available() else "cpu")
log_interval = training_config["log_interval"]
save_interval = training_config["save_interval"]

logger = setup_logger()  # <- Logger function from logger.py

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = os.path.join(root_dir, 'data', 'processed')
train_loader = DataLoader(SatelliteMapDataset(data_dir, split='train', transform=transform), 
                          batch_size=batch_size, shuffle=True, num_workers=4)  # <- Dataloader class from dataloader.py
val_loader = DataLoader(SatelliteMapDataset(data_dir, split='val', transform=transform), 
                        batch_size=batch_size, shuffle=False, num_workers=4)  # <- Dataloader class from dataloader.py

generator = GeneratorUNet(
    in_channels=generator_config["in_channels"], 
    out_channels=generator_config["out_channels"],
    base_filters=generator_config["base_filters"],
    num_downs=generator_config["num_downs"]
).to(device)

criterion = nn.L1Loss()

optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

scaler = GradScaler()

# Training Loop
def train(epoch):
    generator.train()
    running_loss = 0.0
    for i, (satellite, map) in enumerate(train_loader):
        satellite = satellite.to(device)
        map = map.to(device)

        optimizer.zero_grad()

        with autocast():
            generated_map = generator(satellite)
            loss = criterion(generated_map, map)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if i % log_interval == 0:
            logger.info(f"Train Epoch: {epoch} [{i * len(satellite)}/{len(train_loader.dataset)}] Loss: {loss.item()}")  # <- Logger function from logger.py

    avg_loss = running_loss / len(train_loader)
    logger.info(f"Train Epoch: {epoch} Average Loss: {avg_loss}")  # <- Logger function from logger.py
    return avg_loss

# Validation Loop
def validate(epoch):
    generator.eval()
    val_loss = 0.0
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for satellite, map in val_loader:
            satellite = satellite.to(device)
            map = map.to(device)

            generated_map = generator(satellite)

            loss = criterion(generated_map, map)
            val_loss += loss.item()

            psnr = calculate_psnr(generated_map, map)  # <- Helper function from helpers.py
            ssim = calculate_ssim(generated_map, map)  # <- Helper function from helpers.py
            psnr_values.append(psnr)
            ssim_values.append(ssim)

    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)

    logger.info(f"Validation Epoch: {epoch} Loss: {avg_val_loss} PSNR: {avg_psnr} SSIM: {avg_ssim}")  # <- Logger function from logger.py
    return avg_val_loss, avg_psnr, avg_ssim

def main():
    fixed_noise = torch.randn(8, 3, 64, 64, device=device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        val_loss, psnr, ssim = validate(epoch)

        if epoch % save_interval == 0:
            save_checkpoint(generator, optimizer, epoch, train_loss, checkpoint_dir=os.path.join(logging_config["checkpoints_dir"], 'baseline'), filename=f'chekpoint_baseline_{epoch // save_interval}.pth')  # <- Helper function from helpers.py

        if epoch % save_interval == 0:
            save_samples(generator, epoch, fixed_noise, sample_dir=os.path.join(logging_config["samples_dir"], 'baseline'), num_samples=2)  # <- Helper function from helpers.py

if __name__ == "__main__":
    main()
