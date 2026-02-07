import torch
import torch.nn as nn
import torch.nn.functional as F

"""
2D convolutional layer (down sample layer)

Args:
    in_channels (int): Number of input channels (3 for RGB)
    base_filters (int): Number of output channels - or number of FILTERS we now apply
    kernel_size (int): 4×4 convolutional filter
    stride (int): 2 (downsamples spatial dimensions by half)
    padding (int): 1 (preserves alignment with even kernel size)
    
Shape:
    Input: (batch, in_channels, H, W)
    Output: (batch, base_filters, H//2, W//2) ~ (batch, base_filters, H/2, W/2)
    
Example: 64×64 RGB → 32×32 with 64 feature maps
"""

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=32, num_downs=4):
        super(GeneratorUNet, self).__init__()

        # Encoder (Downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 4, 2, 1),  # Output: (batch_size, 32, 32, 32)
            nn.LeakyReLU(0.2, True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1),  # Output: (batch_size, 64, 16, 16)
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1),  # Output: (batch_size, 128, 8, 8)
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, True)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 4, 2, 1),  # Output: (batch_size, 256, 4, 4)
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, 3, 1, 1),  # Output: (batch_size, 256, 4, 4)
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(True),
            nn.Conv2d(base_filters * 8, base_filters * 8, 3, 1, 1),  # Output: (batch_size, 256, 4, 4)
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(True)
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 4, 2, 1),  # Output: (batch_size, 128, 8, 8)
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4 + base_filters * 4, base_filters * 2, 4, 2, 1),  # Output: (batch_size, 64, 16, 16)
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2 + base_filters * 2, base_filters, 4, 2, 1),  # Output: (batch_size, 32, 32, 32)
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters + base_filters, base_filters, 4, 2, 1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, True)
        )
        
        self.up4_1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.final = nn.Sequential(
            nn.Conv2d(base_filters, out_channels, 3, 1, 1),  # Output: (batch_size, 3, 64, 64)
            nn.Tanh() 
        )
    
    def forward(self, x):
        # Encoder path with downsampling
        x1 = self.down1(x)  # Output: (batch_size, 32, 32, 32)
        x2 = self.down2(x1)  # Output: (batch_size, 64, 16, 16)
        x3 = self.down3(x2)  # Output: (batch_size, 128, 8, 8)
        x4 = self.down4(x3)  # Output: (batch_size, 256, 4, 4)

        # Bottleneck
        x5 = self.bottleneck(x4)  # Output: (batch_size, 256, 4, 4)

        # Decoder path with upsampling and skip connections
        x6 = self.up1(x5)  # Output: (batch_size, 128, 8, 8)
        x6 = torch.cat([x6, x3], dim=1)  # Skip connection: (batch_size, 256, 8, 8)

        x7 = self.up2(x6)  # Output: (batch_size, 64, 16, 16)
        x7 = torch.cat([x7, x2], dim=1)  # Skip connection: (batch_size, 128, 16, 16)

        x8 = self.up3(x7)  # Output: (batch_size, 32, 32, 32)
        x8 = torch.cat([x8, x1], dim=1)  # Skip connection: (batch_size, 64, 32, 32)

        # Final upsampling layer (without skip connection)
        x9 = self.up4(x8)  # Output: (batch_size, 32, 64, 64)

        x10 = self.up4_1(x9)  # Extra refinement layer: (batch_size, 32, 128, 128)

        # Final layer to get the output
        out = self.final(x10)  # Output: (batch_size, 3, 64, 64)
        return out
