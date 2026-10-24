"""Hard projection => adapter output on old-task data is invariant, while the
adapter stays plastic elsewhere. Run: python tests/sdrsiat/test_hard_invariance.py"""
from _common import D, R, TinyAdapter, run_all, set_seed
import torch

from models.sdrsiat import SDRSIATController, collect_adapter_statistics


def _setup(seed=0):
    set_seed(seed)
    k = 3
    ad = TinyAdapter(D, R, dropout=0.0)
    with torch.no_grad():
        # hidden units 2,3 are dead on ALL inputs before the update -> C_hid has rank <= 2,
        # so W_up keeps free directions (columns 2,3) even under hard projection.
        ad.down_proj.weight[2:].zero_()
        ad.down_proj.bias[2:].fill_(-1.0)
    B = torch.linalg.qr(torch.randn(D, k))[0].t()           # [k, D] orthonormal rows
    X_old = torch.randn(2, 5, k) @ B                         # [2, 5, D] tokens in a k-dim subspace
    X_new = torch.randn(2, 5, D)                             # generic data
    return ad, X_old, X_new


def _random_updates(ad, n=5, seed=123, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    return [(scale * torch.randn(ad.down_proj.weight.shape, generator=g),
             scale * torch.randn(ad.up_proj.weight.shape, generator=g)) for _ in range(n)]


def _apply(ad, ctrl, updates):
    for dWd, dWu in updates:
        snaps = ctrl.snapshot([ad])
        with torch.no_grad():                                 # emulate optimizer.step()
            ad.down_proj.weight.add_(dWd)
            ad.up_proj.weight.add_(dWu)
        ctrl.project_updates([ad], snaps)


def _ctrl(ad, X_old, **cfg):
    C_in, C_hid, _ = collect_adapter_statistics(ad, [(None, X_old, None)], [ad], "cpu", token_mode="all")
    ctrl = SDRSIATController(cfg)
    ctrl.update_statistics(C_in, C_hid, 0)
    ctrl.build_projectors("cpu")
    return ctrl


def test_hard_projection_preserves_old_outputs():
    ad, X_old, X_new = _setup()
    ctrl = _ctrl(ad, X_old, hard_projection=True)
    ad.eval()
    with torch.no_grad():
        out_old0 = ad(X_old, add_residual=False).clone()
        out_new0 = ad(X_new, add_residual=False).clone()
        Wd0, Wu0 = ad.down_proj.weight.clone(), ad.up_proj.weight.clone()
    _apply(ad, ctrl, _random_updates(ad))
    with torch.no_grad():
        out_old1 = ad(X_old, add_residual=False)
        out_new1 = ad(X_new, add_residual=False)
    err = (out_old1 - out_old0).abs().max().item()
    assert err < 1e-4, "old-data output changed by {}".format(err)
    assert (out_new1 - out_new0).abs().max().item() > 1e-3, "adapter lost all plasticity on new data"
    assert (ad.down_proj.weight - Wd0).norm().item() > 1e-2
    assert (ad.up_proj.weight[:, 2:] - Wu0[:, 2:]).norm().item() > 1e-2, "W_up free columns did not move"
    # hidden units that are active on old data span the protected subspace of P_hid
    alive = [j for j in range(R) if ctrl.C_hid[0][j, j].item() > 1e-8]
    assert all(j < 2 for j in alive), alive
    for j in alive:
        assert (ad.up_proj.weight[:, j] - Wu0[:, j]).norm().item() < 1e-3, "protected W_up column {} moved".format(j)


def test_soft_projection_reduces_old_drift_monotonically():
    drifts = []
    for cfg in ({"sdrsiat_project": "none"}, {"alpha_in": 1.0, "alpha_hid": 1.0},
                {"alpha_in": 0.01, "alpha_hid": 0.01}, {"hard_projection": True}):
        ad, X_old, _ = _setup()
        ctrl = _ctrl(ad, X_old, **cfg)
        ad.eval()
        with torch.no_grad():
            out0 = ad(X_old, add_residual=False).clone()
        _apply(ad, ctrl, _random_updates(ad))
        with torch.no_grad():
            drifts.append((ad(X_old, add_residual=False) - out0).norm().item())
    assert all(drifts[i] > drifts[i + 1] for i in range(len(drifts) - 1)), drifts


if __name__ == "__main__":
    run_all(globals())
