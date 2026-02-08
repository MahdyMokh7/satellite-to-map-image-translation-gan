import torch
import torch.nn as nn

class DiscriminatorPatchGAN16(nn.Module):
    """
    PatchGAN Discriminator for 64x64 image translation.
    Input: Concatenation of Satellite + Map image -> 6 channels
    Output: Patch-wise real/fake predictions
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=32):
        super(DiscriminatorPatchGAN16, self).__init__( )

        # Input channels = concatenated input (satellite + map)
        self.model = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(in_channels + out_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final conv: 4x4 -> 4x4 (single channel)
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)
            # Note: No Sigmoid here, use BCEWithLogitsLoss
        )

    def forward(self, input_image, target_image):
        # Concatenate input and target along channel dimension
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)


class DiscriminatorPatchGAN32(nn.Module):
    """
    PatchGAN-32 Discriminator for 64x64 image translation.
    Focuses on STRUCTURE instead of texture.
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=32):
        super().__init__()

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels + out_channels, base_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Keep spatial resolution
            nn.Conv2d(base_filters * 4, base_filters * 8, 4, 1, 1),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(base_filters * 8, 1, 4, 1, 1)
        )

    def forward(self, input_image, target_image):
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)


import torch
import torch.nn as nn

class DiscriminatorPatchGAN64(nn.Module):
    """
    PatchGAN-64 Discriminator for 128x128 image translation.
    Focuses on STRUCTURE instead of texture.
    Output patches will be 64x64.
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=32):
        super().__init__()

        self.model = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(in_channels + out_channels, base_filters, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1), 
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters * 4, base_filters * 8, 4, 1, 1),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 16x16 -> 1 (real/fake)
            nn.Conv2d(base_filters * 8, 1, 4, 1, 1) 
        )

    def forward(self, input_image, target_image):
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)
