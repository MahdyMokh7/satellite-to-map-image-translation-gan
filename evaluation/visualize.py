import torch
import matplotlib.pyplot as plt
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

model.eval()
with torch.no_grad():
    for i, (satellite, real_map) in enumerate(test_loader):
        satellite = satellite.to(device)
        real_map = real_map.to(device)
        fake_map = model(satellite)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(satellite.cpu().numpy().transpose(1, 2, 0))
        ax[0].set_title('Satellite')
        ax[1].imshow(fake_map.cpu().numpy().transpose(1, 2, 0))
        ax[1].set_title('Generated Map')
        ax[2].imshow(real_map.cpu().numpy().transpose(1, 2, 0))
        ax[2].set_title('Ground Truth')
        plt.show()

        if i >= 4:
            break
