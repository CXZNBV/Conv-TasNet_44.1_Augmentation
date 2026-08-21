# model.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, B, H, P, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(B, H, 1),
            nn.PReLU(),
            nn.GroupNorm(1, H),
            nn.Conv1d(H, H, P, padding=d*(P//2), dilation=d, groups=H),
            nn.PReLU(),
            nn.GroupNorm(1, H),
            nn.Conv1d(H, B, 1),
        )
        self.norm = nn.GroupNorm(1, B)

    def forward(self, x):
        return self.norm(x + self.conv(x))

class ConvTasNet(nn.Module):
    def __init__(self, N=256, L=20, B=128, H=256,
                 P=3, X=8, R=3, C=2, stride=10):
        super().__init__()
        self.C = C
        self.encoder = nn.Conv1d(1, N, L, stride=stride, bias=False)
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=stride, bias=False)
        self.ln = nn.GroupNorm(1, N)
        self.bottleneck = nn.Conv1d(N, B, 1)

        self.tcn = nn.Sequential(*[
            self._make_block(B, H, P, d=2**i)
            for _ in range(R) for i in range(X)
        ])

        self.mask_net = nn.Sequential(
            nn.Conv1d(B, N * C, 1),
            nn.Sigmoid()
        )

    def _make_block(self, B, H, P, d):
        return ResidualBlock(B, H, P, d)

    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)                              # [B, 1, T]
        T_orig = x.shape[-1]

        enc = torch.relu(self.encoder(x))               # [B, N, L']
        enc = self.ln(enc)
        feat = self.bottleneck(enc)                     # [B, B_dim, L']
        feat = self.tcn(feat)                           # [B, B_dim, L']
        masks = self.mask_net(feat)                     # [B, N*C, L']

        N = enc.shape[1]
        masks = masks.view(x.shape[0], self.C, N, -1)   # [B, C, N, L']
        masked = masks * enc.unsqueeze(1)               # [B, C, N, L']

        sources = []
        for i in range(self.C):
            s = self.decoder(masked[:, i])              # [B, 1, T_dec]
            if s.shape[-1] < T_orig:
                padding = T_orig - s.shape[-1]
                s = torch.nn.functional.pad(s, (0, padding))
            else:
                s = s[..., :T_orig]
            sources.append(s.squeeze(1))

        return torch.stack(sources, dim=1)              # [B, C, T]