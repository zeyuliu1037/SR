import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    def __init__(self, color_channels=1, upscale_factor=2):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(color_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, color_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


if __name__ == "__main__":
    model = ESPCN(upscale_factor=2)
    print(model)