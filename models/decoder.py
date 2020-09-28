import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_shape=[1, 64, 64], n_latents=10):
        super().__init__()

        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=n_latents,
                      out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=1024),
            nn.ReLU())

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2D(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.ConvTranspose2D(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.ConvTranspose2D(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.ConvTranspose2D(
                in_channels=32,
                out_channels=output_shape[0],
                kernel_size=4,
                stride=2,
                padding=1))

        self.apply(init_weight)

    def forward(self, x, sigmoid=True):
        batch_size = x.size(0)
        x = self.fc_decoder(x)
        x = x.view(batch_size, 4, 4, 64)
        x = self.conv_decoder(x)
        if sigmoid:
            return torch.sigmoid(x)
        else:
            return x


def init_weight(m):
    if isinstance(m, nn.ConvTranspose2D) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
