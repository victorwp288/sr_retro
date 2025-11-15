from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, n_feats, res_scale):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual * self.res_scale


def build_tail(scale, n_feats, res_scale):
    if scale == 2:
        return nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
        )
    if scale == 4:
        layers = [
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            ResidualBlock(n_feats, res_scale),
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
        ]
        return nn.Sequential(*layers)
    raise ValueError("scale must be 2 or 4")


class EDSR(nn.Module):
    def __init__(self, scale, n_feats, n_resblocks, res_scale):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)
        body = []
        for _ in range(n_resblocks):
            body.append(ResidualBlock(n_feats, res_scale))
        self.body = nn.Sequential(*body)
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.tail = build_tail(scale, n_feats, res_scale)

    def forward(self, x):
        features = self.head(x)
        residual = self.body(features)
        residual = self.body_conv(residual)
        features = features + residual
        output = self.tail(features)
        return output
