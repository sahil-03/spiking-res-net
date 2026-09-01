"""Models for controlled CIFAR-10 ANN, LUT and LIF comparisons."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PairwiseLUT(nn.Module):
    """Relative-order address LUT with a straight-through local-boundary gradient."""
    def __init__(self, n_in: int, n_out: int, tables: int = 4, bits: int = 4, seed: int = 0):
        super().__init__()
        if bits > 12:
            raise ValueError("bits > 12 is intentionally disallowed: resident tables grow exponentially")
        g = torch.Generator().manual_seed(seed)
        a = torch.randint(n_in, (tables, bits), generator=g)
        b = torch.randint(n_in - 1, (tables, bits), generator=g)
        b += (b >= a).long()
        self.register_buffer("anchors_a", a)
        self.register_buffer("anchors_b", b)
        self.register_buffer("bit_weights", 2 ** torch.arange(bits - 1, -1, -1))
        self.table = nn.Parameter(torch.empty(tables, 2**bits, n_out))
        nn.init.normal_(self.table, std=1 / math.sqrt(tables * n_out))
        self.last_addresses: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        u = x[..., self.anchors_a] - x[..., self.anchors_b]
        address = ((u > 0).long() * self.bit_weights).sum(-1)
        self.last_addresses = address.detach()
        ti = torch.arange(self.table.shape[0], device=x.device)
        selected = self.table[ti, address]
        r = u.abs().argmin(-1)
        flipped = address ^ self.bit_weights[r]
        alternate = self.table[ti, flipped]
        ustar = u.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        surrogate = (0.5 / (1 + ustar.abs())).unsqueeze(-1)
        return (selected + (surrogate - surrogate.detach()) * (alternate - selected).detach()).sum(-2)


class RankLUT(nn.Module):
    """Multi-neuron relative rank code hashed into sparse learned tables."""
    def __init__(self, n_in: int, n_out: int, tables: int = 4, rank_size: int = 4,
                 table_size: int = 256, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        anchors = torch.stack([torch.randperm(n_in, generator=g)[:rank_size] for _ in range(tables)])
        self.register_buffer("anchors", anchors)
        self.register_buffer("powers", torch.tensor([1, 7, 43, 257][:rank_size]))
        self.table_size = table_size
        self.table = nn.Parameter(torch.empty(tables, table_size, n_out))
        nn.init.normal_(self.table, std=1 / math.sqrt(tables * n_out))
        self.last_addresses: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        v = x[..., self.anchors]
        soft_rank = torch.sigmoid((v.unsqueeze(-1) - v.unsqueeze(-2)) * 4).sum(-1)
        hard_rank = v.argsort(-1).argsort(-1).float()
        rank = hard_rank + soft_rank - soft_rank.detach()
        code = (hard_rank.long() * self.powers).sum(-1) % self.table_size
        self.last_addresses = code.detach()
        ti = torch.arange(self.table.shape[0], device=x.device)
        chosen = self.table[ti, code]
        # Straight-through rank-dependent gain supplies useful input gradients.
        gain = 1 + 0.02 * ((rank * self.powers).sum(-1) - (rank * self.powers).sum(-1).detach())
        return (chosen * gain.unsqueeze(-1)).sum(-2)


class LUTConv(nn.Module):
    def __init__(self, cin: int, cout: int, kind: str, tables: int, bits: int, stride: int, seed: int):
        super().__init__()
        self.stride = stride
        if kind == "pairwise":
            self.lut = PairwiseLUT(cin * 9, cout, tables, bits, seed)
        else:
            self.lut = RankLUT(cin * 9, cout, tables, min(bits, 4), 2**bits, seed)

    def forward(self, x: Tensor) -> Tensor:
        p = F.unfold(x, 3, padding=1, stride=self.stride).transpose(1, 2)
        y = self.lut(p)
        h = (x.shape[-2] + 2 - 3) // self.stride + 1
        w = (x.shape[-1] + 2 - 3) // self.stride + 1
        return y.transpose(1, 2).reshape(x.shape[0], -1, h, w)


class LUTBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, kind: str, tables: int, bits: int, seed: int):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(cin), nn.BatchNorm2d(cout)
        self.c1 = LUTConv(cin, cout, kind, tables, bits, stride, seed)
        self.c2 = LUTConv(cout, cout, kind, tables, bits, 1, seed + 1)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(x))
        return self.skip(x) + self.c2(F.relu(self.bn2(self.c1(y))))


class PreActBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(cin), nn.BatchNorm2d(cout)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(x))
        return self.skip(x) + self.c2(F.relu(self.bn2(self.c1(y))))


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + x.abs()).pow(2)


class LIFBlock(PreActBlock):
    def __init__(self, cin: int, cout: int, stride: int, steps: int = 4):
        super().__init__(cin, cout, stride)
        self.steps = steps
        self.last_event_rate = 0.0
    def forward(self, x: Tensor) -> Tensor:
        mem = torch.zeros_like(x)
        spikes = []
        for _ in range(self.steps):
            mem = 0.5 * mem + x
            s = SpikeFn.apply(mem - 1.0)
            mem = mem - s
            spikes.append(s)
        self.last_event_rate = float(torch.stack(spikes).detach().mean())
        return super().forward(torch.stack(spikes).mean(0))


class CIFARNet(nn.Module):
    def __init__(self, model: str = "ann", width: int = 16, blocks: int = 3,
                 tables: int = 4, bits: int = 4, seed: int = 0):
        super().__init__()
        self.model_name = model
        self.stem = nn.Conv2d(3, width, 3, padding=1, bias=False)
        layers = []
        cin = width
        for stage, cout in enumerate((width, 2 * width, 4 * width)):
            for i in range(blocks):
                stride = 2 if stage and i == 0 else 1
                if model == "ann": layer = PreActBlock(cin, cout, stride)
                elif model == "lif": layer = LIFBlock(cin, cout, stride)
                else: layer = LUTBlock(cin, cout, stride, model, tables, bits, seed + len(layers) * 2)
                layers.append(layer); cin = cout
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(self.stem(x))
        return self.fc(F.relu(self.bn(x)).mean((-2, -1)))

    def instrumentation(self) -> dict[str, float]:
        addresses, events = [], []
        for m in self.modules():
            if isinstance(m, (PairwiseLUT, RankLUT)) and m.last_addresses is not None:
                addresses.append(m.last_addresses.flatten())
            if isinstance(m, LIFBlock): events.append(m.last_event_rate)
        if addresses:
            ent, coll = [], []
            for a in addresses:
                _, c = a.unique(return_counts=True); p = c.float() / c.sum()
                ent.append(float(-(p * p.log2()).sum())); coll.append(1 - float(c.numel()) / a.numel())
            return {"address_entropy_bits": sum(ent)/len(ent), "address_collision_rate": sum(coll)/len(coll)}
        return {"event_rate": sum(events)/max(1, len(events))} if events else {}


def build_model(name: str, **kwargs) -> CIFARNet:
    return CIFARNet(name, **kwargs)


def model_stats(model: nn.Module) -> dict[str, int]:
    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    table_bytes = sum(m.table.numel() * m.table.element_size() for m in model.modules() if isinstance(m, (PairwiseLUT, RankLUT)))
    return {"parameters": params, "parameter_bytes": param_bytes, "buffer_bytes": buffer_bytes,
            "resident_model_bytes": param_bytes + buffer_bytes, "table_bytes": table_bytes}
