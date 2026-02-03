import torch
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

# PSNR Calculation (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2, max_val=1.0):
    """
    img1, img2: (B, C, H, W) or (C, H, W)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr.mean().item()


# SSIM Calculation (Structural Similarity Index)
def calculate_ssim(img1, img2):
    """
    img1, img2: (B, C, H, W) or (C, H, W)
    Assumes images are in [-1, 1]
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1 = img1.detach().cpu()
    img2 = img2.detach().cpu()

    ssim_values = []

    for i in range(img1.size(0)):
        x = img1[i].permute(1, 2, 0).numpy()
        y = img2[i].permute(1, 2, 0).numpy()

        # Convert from [-1, 1] → [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        ssim_val = ssim(
            x,
            y,
            channel_axis=-1,
            data_range=1.0
        )
        ssim_values.append(ssim_val)

    return float(np.mean(ssim_values))



def visualize_image(image_tensor, num_images=5, normalize=True):
    image_tensor = image_tensor.cpu().detach()
    grid = torchvision.utils.make_grid(image_tensor, nrow=num_images, normalize=normalize)
    return grid
