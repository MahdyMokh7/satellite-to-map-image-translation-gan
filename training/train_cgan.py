import os
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

# --- Project setup ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from models.generator import GeneratorUNet
from models.discriminator import DiscriminatorPatchGAN
from utils.dataloader import get_dataloader, transform
from utils.helpers import calculate_psnr, calculate_ssim
from utils.logger import setup_logger, save_checkpoint, save_samples

# --- Load config ---
config_file_path = os.path.join(root_dir, 'configs', 'config.yaml')
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

training_config = config["training"]
dataset_config = config["dataset"]
generator_config = config["generator"]
discriminator_config = config["discriminator"]
logging_config = config["outputs"]
loss_config = config["loss"]

batch_size = training_config["batch_size"]
epochs = training_config["epochs"]
lr = training_config["optimizer"]["lr"]
beta1 = training_config["optimizer"]["beta1"]
num_workers = training_config["num_workers"]
log_interval = training_config["log_interval"]
save_interval = training_config["save_interval"]

lambda_L1 = loss_config["l1"]["weight"]
adv_weight = loss_config["adversarial"]["weight"]

device = torch.device(config["project"]["device"] if torch.cuda.is_available() else "cpu")
logger = setup_logger(log_dir=logging_config["logging_dir"])

# --- Dataloaders ---
data_dir = os.path.join(root_dir, 'data', 'processed')
train_loader = get_dataloader(data_dir, split='train', batch_size=batch_size, shuffle=True, transform=transform)
val_loader = get_dataloader(data_dir, split='val', batch_size=batch_size, shuffle=False, transform=transform)

# --- Models ---
generator = GeneratorUNet(
    in_channels=generator_config["in_channels"],
    out_channels=generator_config["out_channels"],
    base_filters=generator_config["base_filters"]
).to(device)

discriminator = DiscriminatorPatchGAN(
    in_channels=generator_config["in_channels"],
    out_channels=generator_config["out_channels"],
    base_filters=discriminator_config["base_filters"]
).to(device)

# --- Losses ---
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# --- Optimizers ---
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# --- Learning Rate Schedulers (Reduce LR on plateau) ---
scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=5, verbose=True)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', factor=0.5, patience=5, verbose=True)


# # --- Mixed precision ---
# scaler_G = GradScaler()
# scaler_D = GradScaler()

# --- Training helpers ---
def train_one_epoch(epoch):
    generator.train()
    discriminator.train()
    running_G_loss = 0.0
    running_D_loss = 0.0

    for i, (satellite, real_map) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        satellite = satellite.to(device).float()
        real_map = real_map.to(device).float()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        fake_map = generator(satellite)
        pred_fake = discriminator(satellite, fake_map)

        # Adversarial ground truths (match shape dynamically)
        valid = torch.ones_like(pred_fake, device=device)

        loss_G_adv = adversarial_loss(pred_fake, valid) * adv_weight
        loss_G_l1 = l1_loss(fake_map, real_map) * lambda_L1
        loss_G = loss_G_adv + loss_G_l1
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        optimizer_G.step()

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        pred_real = discriminator(satellite, real_map)
        pred_fake_detach = discriminator(satellite, fake_map.detach())

        # Adversarial ground truths for discriminator
        valid = torch.ones_like(pred_real, device=device)
        fake = torch.zeros_like(pred_real, device=device)

        loss_D_real = adversarial_loss(pred_real, valid) * adv_weight
        loss_D_fake = adversarial_loss(pred_fake_detach, fake) * adv_weight
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
        optimizer_D.step()

        running_G_loss += loss_G.item()
        running_D_loss += loss_D.item()

    avg_G_loss = running_G_loss / len(train_loader)
    avg_D_loss = running_D_loss / len(train_loader)
    return avg_G_loss, avg_D_loss


def validate(epoch):
    generator.eval()
    val_loss = 0.0
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for satellite, real_map in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            satellite = satellite.to(device).float()
            real_map = real_map.to(device).float()

            fake_map = generator(satellite)
            loss = l1_loss(fake_map, real_map)
            val_loss += loss.item()

            psnr_values.append(calculate_psnr(fake_map, real_map))
            ssim_values.append(calculate_ssim(fake_map, real_map))

    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    return avg_val_loss, avg_psnr, avg_ssim

def save_history_and_exit(history):
    """Helper function to save history and exit cleanly."""
    history_path = os.path.join(logging_config["logging_dir"], "history.npy")
    np.save(history_path, history)
    logger.info(f"Training history saved at {history_path}")
    sys.exit(0)  # Exit the program cleanly


def main():
    try:
        fixed_noise = torch.randn(8, 3, 64, 64, device=device)  # For sample saving

        history = {
            'train_G_loss': [],
            'train_D_loss': [],
            'val_loss': [],
            'train_PSNR': [],
            'val_PSNR': [],
            'train_SSIM': [],
            'val_SSIM': []
        }

        best_ssim = 0.0
        patience = 10 
        epochs_no_improve = 0

        custom_note = input("Type Developers Custom Note:")
        logger.info(f"\nStarting Training...\nDeveloper Custom Note: {custom_note}\n\n")

        for epoch in range(1, epochs + 1):
            logger.info("=" * 50)
            logger.info(f"Epoch {epoch}/{epochs}")

            # --- Training ---
            G_loss, D_loss = train_one_epoch(epoch)

            # --- Validation ---
            val_loss, val_psnr, val_ssim = validate(epoch)

            # Record metrics
            history['train_G_loss'].append(G_loss)
            history['train_D_loss'].append(D_loss)
            history['val_loss'].append(val_loss)
            history['train_PSNR'].append(-1)  
            history['val_PSNR'].append(val_psnr)
            history['train_SSIM'].append(-1) 
            history['val_SSIM'].append(val_ssim)

            # --- Step the schedulers based on validation SSIM ---
            scheduler_G.step(val_ssim)
            scheduler_D.step(val_ssim)

            # Optional: log current LR
            logger.info(f"Current LR -> G: {optimizer_G.param_groups[0]['lr']:.6f}, D: {optimizer_D.param_groups[0]['lr']:.6f}")

            # --- Logging ---
            logger.info(
                f"Epoch {epoch} Summary | "
                f"G Loss: {G_loss:.4f} | D Loss: {D_loss:.4f} | "
                f"Val L1 Loss: {val_loss:.4f} | "
                f"Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}"
            )
            logger.info("=" * 50)

            # --- Save all checkpoints every save_interval ---
            if epoch % save_interval == 0:
                save_checkpoint(generator, optimizer_G, epoch, G_loss,
                                checkpoint_dir=logging_config["checkpoints_dir"],
                                filename=f'generator_epoch_{epoch}.pth')
                save_checkpoint(discriminator, optimizer_D, epoch, D_loss,
                                checkpoint_dir=logging_config["checkpoints_dir"],
                                filename=f'discriminator_epoch_{epoch}.pth')
                save_samples(generator, epoch, fixed_noise,
                            sample_dir=logging_config["samples_dir"],
                            num_samples=4)

            # --- Best checkpoint & early stopping based on SSIM ---
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                epochs_no_improve = 0
                save_checkpoint(generator, optimizer_G, epoch, G_loss,
                                checkpoint_dir=logging_config["checkpoints_dir"],
                                filename='generator_best.pth')
                save_checkpoint(discriminator, optimizer_D, epoch, D_loss,
                                checkpoint_dir=logging_config["checkpoints_dir"],
                                filename='discriminator_best.pth')
                logger.info(f"Best model updated at epoch {epoch} with SSIM {val_ssim:.4f}")
            else:
                epochs_no_improve += 1


            # Early stopping
            if epochs_no_improve >= patience:
                logger.info(f"No improvement in SSIM for {patience} epochs. Stopping training early.")
                break

    # Saving training history
    except (KeyboardInterrupt, EOFError):
        logger.info("Training interrupted by user (Ctrl + C).")
        save_history_and_exit(history)




if __name__ == "__main__":
    main()
