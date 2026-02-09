import torch
import os
import sys

# --- Project setup ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from utils.helpers import calculate_psnr, calculate_ssim
from models.generator import GeneratorUNet
from utils.dataloader import get_dataloader, transform
import yaml

config_file_path = os.path.join(root_dir, 'configs', 'config.yaml')
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

training_config = config["training"]
dataset_config = config["dataset"]
generator_config = config["generator"]
discriminator_config = config["discriminator"]
logging_config = config["outputs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeneratorUNet(
    in_channels=generator_config['in_channels'], 
    out_channels=generator_config['out_channels'], 
    base_filters=generator_config['base_filters']
).to(device)

checkpoint_path = os.path.join(root_dir, 'results', 'checkpoints', 'cgan', 'generator_best.pth')

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

data_dir = os.path.join(root_dir, *dataset_config['paths']["data_dir"].split('/'))

test_loader = get_dataloader(data_dir=data_dir, split='test', batch_size=1, shuffle=False, transform=transform)

psnr_values, ssim_values = [], []
with torch.no_grad():
    model.eval()
    for satellite, real_map in test_loader:
        satellite = satellite.to(device)
        real_map = real_map.to(device)
        fake_map = model(satellite)
        
        psnr_values.append(calculate_psnr(fake_map, real_map))
        ssim_values.append(calculate_ssim(fake_map, real_map))

avg_psnr = sum(psnr_values) / len(psnr_values)
avg_ssim = sum(ssim_values) / len(ssim_values)

print(f'Average PSNR: {avg_psnr}')
print(f'Average SSIM: {avg_ssim}')
