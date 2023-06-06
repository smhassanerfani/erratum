import torch
import torch.nn as nn


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(DBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial_layer(x)
        return torch.sigmoid(self.layers(out))
   

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoding=True, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs)
            if encoding
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)


class RBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._block = nn.Sequential(
            GBlock(channels, channels, kernel_size=3, padding=1),
            GBlock(channels, channels, activation=False, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self._block(x)


def main():
    x = torch.randn((2, 3, 256, 256))
    disc = Discriminator()
    print(disc(x).shape)

if __name__ == '__main__':
    main()