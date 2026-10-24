"""Update projection: right-multiplication for nn.Linear layout, biases untouched,
weight decay / momentum cannot leak, leaf safety. Run: python tests/sdrsiat/test_project_updates.py"""
from _common import D, R, TinyAdapter, run_all, set_seed
import torch

from models.sdrsiat import SDRSIATController, build_soft_projector


def _rand_sym(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return 0.5 * (A + A.t())


def _ctrl_with(P_in, P_hid, **cfg):
    ctrl = SDRSIATController(cfg)
    ctrl.P_in, ctrl.P_hid = [P_in], [P_hid]
    return ctrl


def _sgd_step(ad, lr=0.1, wd=0.0, momentum=0.0, opt=None):
    x = torch.randn(2, 5, D)
    loss = ad(x, add_residual=False).pow(2).sum() + ad(x, add_residual=False).sum()
    if opt is None:
        opt = torch.optim.SGD(ad.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return opt


def test_layout_and_right_multiplication():
    set_seed(0)
    ad = TinyAdapter(D, R)
    assert tuple(ad.down_proj.weight.shape) == (R, D)          # nn.Linear(d, r): [out, in] = [r, d]
    assert tuple(ad.up_proj.weight.shape) == (D, R)            # nn.Linear(r, d): [d, r]
    P_in, P_hid = _rand_sym(D, 1), _rand_sym(R, 2)
    ctrl = _ctrl_with(P_in, P_hid)
    snaps = ctrl.snapshot([ad])
    Wd_old, Wu_old = snaps[0]
    _sgd_step(ad)
    Wd_new, Wu_new = ad.down_proj.weight.detach().clone(), ad.up_proj.weight.detach().clone()
    ctrl.project_updates([ad], snaps)
    assert torch.allclose(ad.down_proj.weight, Wd_old + (Wd_new - Wd_old) @ P_in, atol=1e-6)
    assert torch.allclose(ad.up_proj.weight, Wu_old + (Wu_new - Wu_old) @ P_hid, atol=1e-6)
    # wrong side would not even be shape-compatible (r != d)
    try:
        P_in @ (Wd_new - Wd_old)
        raise AssertionError("left multiplication should be shape-incompatible")
    except RuntimeError:
        pass


def test_old_input_response_unchanged_under_hard_P():
    set_seed(1)
    ad = TinyAdapter(D, R)
    Q = torch.linalg.qr(torch.randn(D, 3))[0]
    P_in, _, _ = build_soft_projector(Q @ Q.t(), alpha=0.1, hard=True)
    ctrl = _ctrl_with(P_in, None)
    x_old = torch.randn(7, 3) @ Q.t()                           # rows in span(Q)
    pre0 = x_old @ ad.down_proj.weight.detach().t()
    snaps = ctrl.snapshot([ad])
    assert snaps[0][1] is None                                  # W_up not projected (P_hid None)
    _sgd_step(ad)
    ctrl.project_updates([ad], snaps)
    pre1 = x_old @ ad.down_proj.weight.detach().t()
    assert (pre1 - pre0).abs().max() < 1e-4


def test_biases_not_projected():
    set_seed(2)
    ad = TinyAdapter(D, R)
    ctrl = _ctrl_with(torch.zeros(D, D), torch.zeros(R, R))     # P = 0 freezes weights entirely
    snaps = ctrl.snapshot([ad])
    Wd0, Wu0 = ad.down_proj.weight.detach().clone(), ad.up_proj.weight.detach().clone()
    b0 = ad.down_proj.bias.detach().clone()
    _sgd_step(ad)
    b_after_step = ad.down_proj.bias.detach().clone()
    ctrl.project_updates([ad], snaps)
    assert torch.allclose(ad.down_proj.weight, Wd0) and torch.allclose(ad.up_proj.weight, Wu0)
    assert torch.equal(ad.down_proj.bias, b_after_step) and not torch.allclose(b_after_step, b0)


def test_project_mode_flags():
    set_seed(3)
    ad = TinyAdapter(D, R)
    ctrl = SDRSIATController({"sdrsiat_project": "down"})
    assert ctrl.project_down and not ctrl.project_up
    ctrl.P_in, ctrl.P_hid = [torch.zeros(D, D)], [None]
    snaps = ctrl.snapshot([ad])
    Wd0 = ad.down_proj.weight.detach().clone()
    _sgd_step(ad)
    Wu_step = ad.up_proj.weight.detach().clone()
    ctrl.project_updates([ad], snaps)
    assert torch.allclose(ad.down_proj.weight, Wd0)
    assert torch.equal(ad.up_proj.weight, Wu_step)
    ctrl_none = SDRSIATController({"sdrsiat_project": "none"})
    assert not ctrl_none.project_down and not ctrl_none.project_up


def test_weight_decay_and_momentum_cannot_leak():
    set_seed(4)
    ad = TinyAdapter(D, R)
    Q = torch.linalg.qr(torch.randn(D, 3))[0]
    P_in, _, _ = build_soft_projector(Q @ Q.t(), alpha=0.1, hard=True)
    ctrl = _ctrl_with(P_in, None)
    x_old = torch.randn(5, 3) @ Q.t()
    pre0 = x_old @ ad.down_proj.weight.detach().t()
    opt = torch.optim.SGD(ad.parameters(), lr=0.1, weight_decay=0.5, momentum=0.9)
    for _ in range(4):
        snaps = ctrl.snapshot([ad])
        opt.zero_grad(set_to_none=True)
        for p in ad.parameters():
            p.grad = torch.zeros_like(p)                        # pure weight-decay (+momentum) update
        opt.step()
        ctrl.project_updates([ad], snaps)
    pre1 = x_old @ ad.down_proj.weight.detach().t()
    assert (pre1 - pre0).abs().max() < 1e-4
    # but decay still acted in the free complement
    comp = (torch.eye(D) - Q @ Q.t())
    assert (ad.down_proj.weight.detach() @ comp).norm() < 0.99 * 1e9  # sanity: finite
    for _ in range(2):                                          # real gradients + momentum
        snaps = ctrl.snapshot([ad])
        _sgd_step(ad, opt=opt)
        ctrl.project_updates([ad], snaps)
    pre2 = x_old @ ad.down_proj.weight.detach().t()
    assert (pre2 - pre0).abs().max() < 1e-4


def test_leaf_safety_and_energy_stats():
    set_seed(5)
    ad = TinyAdapter(D, R)
    ctrl = _ctrl_with(0.5 * torch.eye(D), 0.5 * torch.eye(R))
    snaps = ctrl.snapshot([ad])
    _sgd_step(ad)
    ctrl.project_updates([ad], snaps)
    w = ad.down_proj.weight
    assert w.is_leaf and w.requires_grad and isinstance(w, torch.nn.Parameter)
    _sgd_step(ad)                                              # autograd still works afterwards
    kept = float(ctrl._step_stats["down_kept"]) / float(ctrl._step_stats["down_raw"])
    assert abs(kept - 0.25) < 1e-5                              # ||0.5 delta||^2 / ||delta||^2


if __name__ == "__main__":
    run_all(globals())
