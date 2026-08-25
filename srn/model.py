import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LUTTransform(nn.Module):
    anchors_a: Tensor
    anchors_b: Tensor
    bit_weights: Tensor
    table_idx: Tensor

    def __init__(self, n_in: int, n_out: int, n_t: int, n_c: int) -> None:
        super().__init__()

        a = torch.randint(0, n_in, (n_t, n_c))
        b = torch.randint(0, n_in, (n_t, n_c))

        # enforce a_r != b_r
        collision = a == b
        while collision.any():
            b[collision] = torch.randint(0, n_in, (int(collision.sum()),))
            collision = a == b
        self.register_buffer("anchors_a", a)
        self.register_buffer("anchors_b", b)

        self.S = nn.Parameter(torch.zeros(n_t, 2**n_c, n_out))

        self.register_buffer("bit_weights", 2 ** torch.arange(n_c - 1, -1, -1))
        self.register_buffer("table_idx", torch.arange(n_t))

    def U(self, u: Tensor) -> Tensor:
        return 0.5 / (1 + u.abs())

    def forward(self, x: Tensor) -> Tensor:
        u = x[..., self.anchors_a] - x[..., self.anchors_b]
        bits = (u > 0).long()
        j = (bits * self.bit_weights).sum(-1)

        S_j = self.S[self.table_idx, j]

        r_star = u.abs().argmin(-1)
        u_star = torch.gather(u, -1, r_star.unsqueeze(-1)).squeeze(-1)

        flip_val = self.bit_weights[r_star]
        j_flip = j ^ flip_val
        S_flip = self.S[self.table_idx, j_flip]

        weight = self.U(u_star).unsqueeze(-1)
        y_i = S_j + (weight - weight.detach()) * (S_flip - S_j).detach()
        return y_i.sum(-2)


class SpikingConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_t: int,
        n_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding

        n_in = in_channels * (kernel_size**2)
        self.lut = LUTTransform(n_in=n_in, n_out=out_channels, n_t=n_t, n_c=n_c)

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape

        patches = F.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        patches = patches.transpose(1, 2)
        y: Tensor = self.lut(patches)

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return y.transpose(1, 2).reshape(B, -1, H_out, W_out)


class ResidualSpikingConv2d(nn.Module):
    conv: SpikingConv2d

    def __init__(self, channels: int, n_t: int, n_c: int, kernel_size: int = 3):
        super().__init__()
        self.conv = SpikingConv2d(channels, channels, n_t, n_c, kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.conv(x)
        return x + y


class SpikingResNet(nn.Module):
    classifier: LUTTransform

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = SpikingConv2d(3, 16, n_t=8, n_c=6, kernel_size=3, stride=1)

        self.stage1 = nn.Sequential(
            *[ResidualSpikingConv2d(16, n_t=16, n_c=8) for _ in range(6)]
        )
        self.transition1 = SpikingConv2d(16, 32, n_t=16, n_c=8, kernel_size=3, stride=2)

        self.stage2 = nn.Sequential(
            *[ResidualSpikingConv2d(32, n_t=16, n_c=8) for _ in range(6)]
        )
        self.transition2 = SpikingConv2d(32, 64, n_t=16, n_c=8, kernel_size=3, stride=2)

        self.stage3 = nn.Sequential(
            *[ResidualSpikingConv2d(64, n_t=16, n_c=8) for _ in range(6)]
        )

        self.classifier = LUTTransform(n_in=64, n_out=num_classes, n_t=32, n_c=6)

    def forward(self, x: Tensor) -> Tensor:
        for fn in [
            self.stem,
            self.stage1,
            self.transition1,
            self.stage2,
            self.transition2,
            self.stage3,
        ]:
            x = fn(x)

        x = x.mean(dim=(-2, -1))
        logits: Tensor = self.classifier(x)
        return logits
