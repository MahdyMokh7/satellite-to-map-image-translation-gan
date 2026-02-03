import torch
from utils.helpers import calculate_psnr, calculate_ssim
from models.generator import GeneratorUNet
from utils.dataloader import get_dataloader
import os
import yaml

config = yaml.safe_load(open('config.yaml'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeneratorUNet(in_channels=3, out_channels=3, base_filters=32).to(device)
model.load_state_dict(torch.load('path_to_checkpoint'))

test_loader = get_dataloader(config['data_dir'], split='test', batch_size=1, shuffle=False)

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
