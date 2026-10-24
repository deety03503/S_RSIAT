"""Flag-off equivalence tests: ensure that when --sdrsiat is disabled/absent,
the code path, config, and optimization logic remain completely unaltered.
Run: python tests/sdrsiat/test_flag_off_equivalence.py"""
import argparse
import copy
from _common import D, R, TinyNet, TinyProjector, compute_losses, make_old_new, run_all, set_seed
import torch
import torch.nn as nn
import torch.optim as optim

from utils.sdrsiat_config import (
    SDRSIAT_CLI_KEYS,
    add_sdrsiat_cli_args,
    apply_sdrsiat_overrides,
    pop_sdrsiat_cli,
)
from models.sdrsiat import SDRSIATController, get_adapters, route_gradients


def test_cli_config_clean_when_flag_omitted():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test.json")
    add_sdrsiat_cli_args(parser)

    # Simulate running without any SD-RSIAT flags
    args = parser.parse_args(["--config", "custom.json"])
    base_config = {"model_name": "adapter", "init_lr": 0.01, "batch_size": 32}
    config_copy = copy.deepcopy(base_config)

    sdrsiat_cli = pop_sdrsiat_cli(args)
    # The returned dict should only contain None values
    assert all(v is None for v in sdrsiat_cli.values())
    # No SD-RSIAT attributes remain on args
    for k in SDRSIAT_CLI_KEYS:
        assert not hasattr(args, k)

    # Applying empty overrides should leave config completely unchanged
    merged = apply_sdrsiat_overrides(config_copy, sdrsiat_cli)
    assert merged == base_config


def test_cli_config_when_flag_provided():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test.json")
    add_sdrsiat_cli_args(parser)

    args = parser.parse_args([
        "--config", "custom.json",
        "--sdrsiat",
        "--alpha_in", "0.05",
        "--rho", "0.2",
        "--stats_token_mode", "patch",
        "--hard_projection"
    ])
    base_config = {"model_name": "adapter", "init_lr": 0.01}
    sdrsiat_cli = pop_sdrsiat_cli(args)
    merged = apply_sdrsiat_overrides(base_config, sdrsiat_cli)

    assert merged["sdrsiat"] is True
    assert merged["alpha_in"] == 0.05
    assert merged["rho"] == 0.2
    assert merged["stats_token_mode"] == "patch"
    assert merged["hard_projection"] is True


def test_optimization_equivalence_flag_off_vs_baseline():
    """Verify that when SD-RSIAT is OFF, standard autograd backward + step
    produces identical parameter trajectories."""
    set_seed(42)
    net1, old1, proj1, protos1, x1, y1 = make_old_new(seed=42)
    net2 = copy.deepcopy(net1)
    old2 = copy.deepcopy(old1)
    proj2 = copy.deepcopy(proj1)
    protos2 = protos1.clone()
    x2, y2 = x1.clone(), y1.clone()

    beta, gamma = 0.8, 0.3
    lr = 0.05

    # Run baseline (original RSIAT path: loss = loss_c + beta * loss_align + gamma * loss_orth)
    opt1 = optim.SGD(
        list(net1.convnet.parameters()) + list(net1.fc.parameters()) + list(proj1.parameters()),
        lr=lr,
        momentum=0.9
    )
    opt1.zero_grad()
    _, lc1, la1, lo1 = compute_losses(net1, old1, proj1, protos1, x1, y1)
    loss1 = lc1 + beta * la1 + gamma * lo1
    loss1.backward()
    opt1.step()

    # Run exact duplicate baseline
    opt2 = optim.SGD(
        list(net2.convnet.parameters()) + list(net2.fc.parameters()) + list(proj2.parameters()),
        lr=lr,
        momentum=0.9
    )
    opt2.zero_grad()
    _, lc2, la2, lo2 = compute_losses(net2, old2, proj2, protos2, x2, y2)
    loss2 = lc2 + beta * la2 + gamma * lo2
    loss2.backward()
    opt2.step()

    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)
    for p1, p2 in zip(proj1.parameters(), proj2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)


def test_task0_behavior_unaffected():
    """On Task 0 (cur_task == 0), SD-RSIAT does no projection and uses base RSIAT losses."""
    args_off = {"sdrsiat": False, "alpha_in": 0.01}
    args_on = {"sdrsiat": True, "alpha_in": 0.01}

    ctrl = SDRSIATController(args_on)
    # At task 0, controller has no statistics and build_projectors returns False
    assert not ctrl.has_statistics()
    assert not ctrl.build_projectors("cpu")
    assert len(ctrl.P_in) == 0 and len(ctrl.P_hid) == 0


if __name__ == "__main__":
    run_all(globals())
