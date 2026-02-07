import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.amp import GradScaler, autocast
import yaml
from models.generator import GeneratorUNet
from utils.helpers import calculate_psnr, calculate_ssim
from utils.logger import setup_logger, save_checkpoint, save_samples
from utils.dataloader import get_dataloader, transform
from tqdm import tqdm


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
lr = training_config["optimizer"]["lr_G"]
beta1 = training_config["optimizer"]["beta1"]
device = torch.device(config["project"]["device"] if torch.cuda.is_available() else "cpu")
log_interval = training_config["log_interval"]
save_interval = training_config["save_interval"]
num_workers = training_config["num_workers"]

device_str = config["project"]["device"] if torch.cuda.is_available() else "cpu"

logger = setup_logger(log_dir='./results/logs/baseline')

data_dir = os.path.join(root_dir, 'data', 'processed')
train_loader = get_dataloader(
    data_dir=data_dir,
    split='train',
    batch_size=batch_size,
    shuffle=True,
    transform=transform
)
val_loader = get_dataloader(
    data_dir=data_dir,
    split='val',
    batch_size=batch_size,
    shuffle=False,
    transform=transform
)

generator = GeneratorUNet(
    in_channels=generator_config["in_channels"], 
    out_channels=generator_config["out_channels"],
    base_filters=generator_config["base_filters"],
    num_downs=generator_config["num_downs"]
).to(device)

criterion = nn.L1Loss()

optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# used for mix precision training
# scaler = GradScaler(device_str)

def train(epoch):
    generator.train()
    running_loss = 0.0
    for i, (satellite, map) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):

        satellite = satellite.to(device).float()
        map = map.to(device).float()

        optimizer.zero_grad()

        # with autocast(device_str):
        generated_map = generator(satellite)
        loss = criterion(generated_map, map)

        # These for Grad scaler (FP16 and FP32 mixed precision training)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # if i % log_interval == 0:
        #     logger.info(f"Train Epoch: {epoch} [{i * len(satellite)}/{len(train_loader.dataset)}] Loss: {loss.item()}")

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def validate(epoch):
    generator.eval()
    val_loss = 0.0
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for satellite, map in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):

            satellite = satellite.to(device).float()
            map = map.to(device).float()

            generated_map = generator(satellite)

            loss = criterion(generated_map, map)
            val_loss += loss.item()

            psnr = calculate_psnr(generated_map, map)
            ssim = calculate_ssim(generated_map, map)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)

    return avg_val_loss, avg_psnr, avg_ssim

def main():
    fixed_noise = torch.randn(8, 3, 64, 64, device=device)

    for epoch in range(1, epochs + 1):

        logger.info("=" * 50)
        logger.info(f"Epoch {epoch}/{epochs}")

        train_loss = train(epoch)
        val_loss, psnr, ssim = validate(epoch)

        logger.info(
        f"Epoch {epoch} Summary | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val PSNR: {psnr:.2f} | "
        f"Val SSIM: {ssim:.4f}"
        )
        logger.info("=" * 50)   

        if epoch % save_interval == 0:
            save_checkpoint(generator, optimizer, epoch, train_loss, checkpoint_dir=os.path.join(".","results", "checkpoints","baseline"), filename=f'chekpoint_baseline_{epoch}.pth')

        if epoch % save_interval == 0:
            save_samples(generator, epoch, fixed_noise, sample_dir=os.path.join(".","results", "samples", 'baseline'), num_samples=2)

if __name__ == "__main__":
    main()
