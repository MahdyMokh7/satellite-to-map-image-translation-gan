import torch
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

# PSNR Calculation (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_val / torch.sqrt(mse))


# SSIM Calculation (Structural Similarity Index)
def calculate_ssim(img1, img2):
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    ssim_value = ssim(img1, img2, multichannel=True)
    return ssim_value


def visualize_image(image_tensor, num_images=5, normalize=True):
    image_tensor = image_tensor.cpu().detach()
    grid = torchvision.utils.make_grid(image_tensor, nrow=num_images, normalize=normalize)
    return grid
