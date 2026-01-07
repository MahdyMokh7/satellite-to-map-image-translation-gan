import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=32, num_downs=5):
        """
        Generator Network based on U-Net architecture.
        
        Args:
            in_channels (int): The number of input channels (e.g., 3 for RGB images).
            out_channels (int): The number of output channels (e.g., 3 for RGB images).
            base_filters (int): The base number of filters in the first layer.
            num_downs (int): Number of downsampling layers.
        """
        super(GeneratorUNet, self).__init__()

        self.downs = nn.ModuleList()
        for i in range(num_downs):
            in_channels_i = in_channels if i == 0 else base_filters * (2**(i - 1))
            out_channels_i = base_filters * (2**i)
            self.downs.append(self.conv_block(in_channels_i, out_channels_i))

        self.bottleneck = self.conv_block(base_filters * (2**(num_downs - 1)), base_filters * (2**num_downs))

        self.ups = nn.ModuleList()
        for i in range(num_downs):
            in_channels_i = base_filters * (2**(num_downs - i))
            out_channels_i = base_filters * (2**(num_downs - i - 1))
            self.ups.append(self.upconv_block(in_channels_i, out_channels_i))

        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down_features = []
        for down in self.downs:
            x = down(x)
            down_features.append(x)

        x = self.bottleneck(x)

        for i, up in enumerate(self.ups):
            x = up(x)
            x = x + down_features[-(i + 1)]
        
        x = self.final_conv(x)
        return torch.tanh(x)
