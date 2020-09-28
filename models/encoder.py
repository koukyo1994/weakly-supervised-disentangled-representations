import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_shape=[1, 64, 64], n_latents=10):
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2D(in_channels=input_shape[0],
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ReLU())

        in_channels = int((input_shape[1] / (2 ** 4)) * (input_shape[2] / (2 ** 4)) * 64)

        self.fc_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_channels,
                      out_features=256),
            nn.ReLU())

        self.mu_encoder = nn.Linear(in_features=256, out_features=n_latents)
        self.logvar_encoder = nn.Linear(in_features=256, out_features=n_latents)

        self.apply(init_weight)

    def forward(self, x):
        emb = self.fc_encoder(self.conv_encoder(x))
        mu = self.mu_encoder(emb)
        logvar = self.logvar_encoder(emb)

        return mu, logvar


def init_weight(m):
    if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
