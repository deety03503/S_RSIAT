"""Gradient routing: W_up never receives L_align; groups/rho correct.
Run: python tests/sdrsiat/test_grad_routing.py"""
from _common import compute_losses, make_old_new, run_all
import torch
import torch.nn as nn

from models.sdrsiat import SDRSIATController, get_adapters, route_gradients

BETA, GAMMA = 0.7, 0.4


def _setup(cfg=None):
    net, old, proj, protos, x, y = make_old_new(seed=0)
    adapters = get_adapters(net.convnet)
    ctrl = SDRSIATController(cfg or {})
    roles = ctrl.split_roles(adapters)
    fc = [p for p in net.fc.parameters() if p.requires_grad]
    pp = list(proj.parameters())
    return net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp


def test_routing_matches_per_loss_references():
    net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp = _setup()
    st, pl = roles["stable"], roles["plastic"]
    assert len(adapters) == 2 and len(st) == 4 and len(pl) == 4   # (W, b) x 2 layers each
    # references from a fresh forward
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    ref_cos = torch.autograd.grad(lc, st + pl + fc, retain_graph=True)
    ref_align = torch.autograd.grad(BETA * la, st + pl, retain_graph=True)
    ref_proj = torch.autograd.grad(BETA * la + GAMMA * lo, pp)
    # routed
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    route_gradients(lc, la, lo, BETA, GAMMA, st, pl, pp, fc)
    ns, npl = len(st), len(pl)
    for i, p in enumerate(st):
        assert torch.allclose(p.grad, ref_cos[i] + ref_align[i], atol=1e-6)
    assert any((ref_align[i].abs().max() > 1e-7) for i in range(ns)), "align should reach stable params"
    for j, p in enumerate(pl):
        assert torch.allclose(p.grad, ref_cos[ns + j], atol=1e-6), "plastic got more than L_cos"
    # routing is non-trivial: L_align WOULD have reached W_up through the graph
    assert any((ref_align[ns + j].abs().max() > 1e-7) for j in range(npl))
    for k, p in enumerate(fc):
        assert torch.allclose(p.grad, ref_cos[ns + npl + k], atol=1e-6)
    for k, p in enumerate(pp):
        assert torch.allclose(p.grad, ref_proj[k], atol=1e-6)


def test_matches_full_backward_except_plastic():
    net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp = _setup()
    st, pl = roles["stable"], roles["plastic"]
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    (lc + BETA * la + GAMMA * lo).backward()
    full = {id(p): p.grad.clone() for p in st + pl + fc + pp}
    for p in st + pl + fc + pp:
        p.grad = None
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    route_gradients(lc, la, lo, BETA, GAMMA, st, pl, pp, fc)
    for p in st + fc + pp:
        assert torch.allclose(p.grad, full[id(p)], atol=1e-6)
    assert any(not torch.allclose(p.grad, full[id(p)], atol=1e-7) for p in pl)


def test_unused_param_and_frozen_projector_are_safe():
    net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp = _setup()
    dummy = nn.Parameter(torch.randn(3))                       # not in the graph
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    route_gradients(lc, la, lo, BETA, GAMMA, roles["stable"], roles["plastic"] + [dummy], pp, fc)
    assert dummy.grad is None
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    route_gradients(lc, la, lo, BETA, GAMMA, roles["stable"], roles["plastic"], [], fc)   # no projector


def test_duplicate_role_raises():
    net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp = _setup()
    _, lc, la, lo = compute_losses(net, old, proj, protos, x, y)
    try:
        route_gradients(lc, la, lo, BETA, GAMMA, roles["stable"], roles["stable"][:1], pp, fc)
    except ValueError:
        return
    raise AssertionError("duplicate parameter across roles must raise")


def test_swap_roles_and_bias_freeze():
    net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp = _setup({"sdrsiat_swap_roles": True})
    assert roles["stable_w"][0] is adapters[0].up_proj.weight
    assert roles["plastic_w"][0] is adapters[0].down_proj.weight
    _, _, _, _, _, _, adapters, ctrl, roles, fc, pp = _setup({"freeze_bias_incremental": True})
    assert roles["stable_b"] == [] and roles["plastic_b"] == []
    assert len(roles["frozen"]) == 4
    assert all(p.dim() == 1 for p in roles["frozen"])


def test_param_groups_rho_complete_no_duplicates():
    net, old, proj, protos, x, y, adapters, ctrl, roles, fc, pp = _setup({"rho": 0.25})
    groups = ctrl.build_param_groups(roles, base_lr=0.1, base_wd=1e-4, fc_params=fc, fc_lr=0.1, fc_wd=1e-4,
                                     proj_params=pp, proj_lr=0.01, proj_wd=1e-3)
    by = {g["name"]: g for g in groups}
    assert abs(by["sd_stable_w"]["lr"] - 0.025) < 1e-12 and abs(by["sd_plastic_w"]["lr"] - 0.1) < 1e-12
    assert by["projector"]["lr"] == 0.01 and by["projector"]["weight_decay"] == 1e-3
    required = [p for p in roles["all_adapter"] if p.requires_grad] + fc + pp
    assert ctrl.check_param_groups(groups, required)
    torch.optim.SGD(groups, momentum=0.9)                       # constructible
    bad = groups + [{"params": [fc[0]], "lr": 0.1, "weight_decay": 0.0, "name": "dup"}]
    try:
        ctrl.check_param_groups(bad, required)
        raise AssertionError("duplicate not detected")
    except RuntimeError:
        pass
    try:
        ctrl.check_param_groups([g for g in groups if g["name"] != "fc"], required)
        raise AssertionError("missing not detected")
    except RuntimeError:
        pass
    # frozen biases: excluded from groups, accepted by the check
    _, _, _, _, _, _, _, ctrl2, roles2, fc2, pp2 = _setup({"freeze_bias_incremental": True})
    g2 = ctrl2.build_param_groups(roles2, 0.1, 1e-4, fc2, 0.1, 1e-4, pp2, 0.01, 1e-3)
    req2 = [p for p in roles2["all_adapter"] if p.requires_grad] + fc2 + pp2
    assert ctrl2.check_param_groups(g2, req2, excluded=roles2["frozen"])


if __name__ == "__main__":
    run_all(globals())
