"""Shared helpers for SD-RSIAT tests (CPU, tiny dims, fixed seed, no timm).

Run any test from the repository root, e.g.:
    python tests/sdrsiat/test_soft_proj.py
"""
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

D, R = 16, 4


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyAdapter(nn.Module):
    """Mirror of network/vision_transformer_adapter.py::Adapter with the repo's
    config (layernorm 'none', scale 0.1, ReLU, functional dropout on h).
    Unlike the repo's 'lora' init, up_proj is random so gradients reach down_proj."""

    def __init__(self, d=D, r=R, dropout=0.0, scale=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d, r)          # weight [r, d]
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(r, d)            # weight [d, r]
        self.dropout = dropout
        self.scale = scale
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.normal_(self.up_proj.weight, std=0.3)
            nn.init.normal_(self.down_proj.bias, std=0.1)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        down = self.non_linear_func(self.down_proj(x))
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale
        return up + residual if add_residual else up


class TinyBlock(nn.Module):
    def __init__(self, d=D, r=R, dropout=0.0):
        super().__init__()
        self.frozen = nn.Linear(d, d)
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.adaptmlp = TinyAdapter(d, r, dropout)

    def forward(self, x):
        adapt_x = self.adaptmlp(x, add_residual=False)      # parallel adapter
        return x + torch.tanh(self.frozen(x)) + adapt_x


class TinyBackbone(nn.Module):
    def __init__(self, d=D, r=R, n_layers=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[TinyBlock(d, r, dropout) for _ in range(n_layers)])

    def forward(self, x):                                   # x [B, N, d]
        return self.blocks(x)[:, 0]                          # CLS token feature [B, d]


class TinyNet(nn.Module):
    def __init__(self, d=D, r=R, n_layers=2, n_cls=3, dropout=0.0):
        super().__init__()
        self.convnet = TinyBackbone(d, r, n_layers, dropout)
        self.fc = nn.Linear(d, n_cls, bias=False)

    def extract_vector(self, x):
        return self.convnet(x)

    def logits(self, feats):
        return F.linear(F.normalize(feats, dim=1), F.normalize(self.fc.weight, dim=1))


class TinyProjector(nn.Module):
    """Residual autoencoder like utils/toolkit.py::AutoencoderSigmoid."""

    def __init__(self, d=D, code=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, code), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(code, d), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x)) + x


def compute_losses(net, old_net, proj, protos, x, y, scale=10.0):
    """(L_cos, L_align, L_orth) with the same structure as RSIAT's incremental losses."""
    feats = net.extract_vector(x)
    logits = net.logits(feats)
    loss_c = F.cross_entropy(scale * logits, y)
    with torch.no_grad():
        f_old = old_net.extract_vector(x)
    fo = proj(f_old)
    loss_align = F.mse_loss(feats, fo)
    po = F.normalize(proj(protos), dim=1)
    so = F.normalize(fo, dim=1)
    loss_orth = (po @ so.t()).sum() / (po.shape[0] * so.shape[0])
    return logits, loss_c, loss_align, loss_orth


def make_old_new(seed=0, n_layers=2, dropout=0.0):
    """Current net = old net + small perturbation of adapters (so L_align has gradient)."""
    import copy
    set_seed(seed)
    net = TinyNet(n_layers=n_layers, dropout=dropout)
    old = copy.deepcopy(net)
    for p in old.parameters():
        p.requires_grad = False
    old.eval()
    with torch.no_grad():
        for blk in net.convnet.blocks:
            blk.adaptmlp.down_proj.weight.add_(0.05 * torch.randn_like(blk.adaptmlp.down_proj.weight))
            blk.adaptmlp.up_proj.weight.add_(0.05 * torch.randn_like(blk.adaptmlp.up_proj.weight))
    proj = TinyProjector()
    protos = torch.randn(3, D)
    x = torch.randn(2, 5, D)                                 # batch=2, tokens=5
    y = torch.tensor([0, 2])
    return net, old, proj, protos, x, y


def run_all(module_globals):
    names = sorted(k for k, v in module_globals.items() if k.startswith("test_") and callable(v))
    for n in names:
        set_seed(0)
        module_globals[n]()
        print("PASS", n)
    print("ALL {} TESTS PASSED in {}".format(len(names), os.path.basename(module_globals.get("__file__", "?"))))
