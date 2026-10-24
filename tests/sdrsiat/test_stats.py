"""Statistics collection tests: hooks, accumulation, token modes, trace normalization,
effective rank. Run: python tests/sdrsiat/test_stats.py"""
from _common import D, R, TinyAdapter, TinyBackbone, TinyNet, run_all, set_seed
import torch
import torch.nn as nn

from models.sdrsiat import (
    accumulate_normalized,
    collect_adapter_statistics,
    effective_rank,
    get_adapters,
    select_tokens,
    subspace_energy_fraction,
    SDRSIATController,
)


def test_select_tokens_modes():
    set_seed(0)
    B, N, C = 2, 5, D
    x = torch.randn(B, N, C)
    # 'all' -> (B*N, C)
    t_all = select_tokens(x, "all")
    assert t_all.shape == (B * N, C)
    assert torch.allclose(t_all[0], x[0, 0])
    assert torch.allclose(t_all[4], x[0, 4])
    assert torch.allclose(t_all[5], x[1, 0])

    # 'cls' -> (B, C)
    t_cls = select_tokens(x, "cls")
    assert t_cls.shape == (B, C)
    assert torch.allclose(t_cls[0], x[0, 0])
    assert torch.allclose(t_cls[1], x[1, 0])

    # 'patch' -> (B*(N-1), C)
    t_patch = select_tokens(x, "patch")
    assert t_patch.shape == (B * (N - 1), C)
    assert torch.allclose(t_patch[0], x[0, 1])

    # 2D tensor handling
    x_2d = torch.randn(B, C)
    assert select_tokens(x_2d, "all").shape == (B, C)
    assert select_tokens(x_2d, "cls").shape == (B, C)


def test_collect_statistics_shapes_and_tokens():
    set_seed(0)
    net = TinyNet(d=D, r=R, n_layers=2)
    adapters = get_adapters(net.convnet)
    assert len(adapters) == 2

    # Fake dataloader: 3 batches of (None, inputs, targets)
    B, N = 4, 6
    batches = [(None, torch.randn(B, N, D), torch.zeros(B, dtype=torch.long)) for _ in range(3)]

    C_in, C_hid, info = collect_adapter_statistics(
        net, batches, adapters, "cpu", token_mode="all", forward_fn=net.extract_vector
    )

    assert len(C_in) == 2 and len(C_hid) == 2
    for Cin, Chid in zip(C_in, C_hid):
        assert Cin.shape == (D, D) and Cin.dtype == torch.float32 and Cin.device.type == "cpu"
        assert Chid.shape == (R, R) and Chid.dtype == torch.float32 and Chid.device.type == "cpu"
        # Must be symmetric
        assert torch.allclose(Cin, Cin.t(), atol=1e-5)
        assert torch.allclose(Chid, Chid.t(), atol=1e-5)
        # Eigenvalues must be non-negative (PSD)
        assert torch.linalg.eigvalsh(Cin).min().item() >= -1e-5
        assert torch.linalg.eigvalsh(Chid).min().item() >= -1e-5

    # Check total token counts: 3 batches * 4 samples * 6 tokens = 72 tokens
    assert info["tokens_in"] == [72, 72]
    assert info["tokens_hid"] == [72, 72]


def test_collect_statistics_cls_mode():
    set_seed(0)
    net = TinyNet(d=D, r=R, n_layers=2)
    adapters = get_adapters(net.convnet)
    B, N = 4, 6
    batches = [(None, torch.randn(B, N, D), torch.zeros(B, dtype=torch.long)) for _ in range(2)]

    C_in, C_hid, info = collect_adapter_statistics(
        net, batches, adapters, "cpu", token_mode="cls", forward_fn=net.extract_vector
    )
    # 2 batches * 4 samples = 8 tokens
    assert info["tokens_in"] == [8, 8]
    assert info["tokens_hid"] == [8, 8]


def test_collect_statistics_max_batches():
    set_seed(0)
    net = TinyNet(d=D, r=R, n_layers=2)
    adapters = get_adapters(net.convnet)
    B, N = 2, 5
    batches = [(None, torch.randn(B, N, D), torch.zeros(B, dtype=torch.long)) for _ in range(5)]

    C_in, C_hid, info = collect_adapter_statistics(
        net, batches, adapters, "cpu", token_mode="all", max_batches=2, forward_fn=net.extract_vector
    )
    # Only 2 batches evaluated: 2 * 2 * 5 = 20 tokens
    assert info["tokens_in"] == [20, 20]


def test_accumulate_normalized_and_trace_scaling():
    set_seed(0)
    # Matrix with known trace
    A = torch.randn(D, D)
    C1 = A @ A.t()
    tr1 = torch.trace(C1).item()
    assert tr1 > 0

    C_acc, ok = accumulate_normalized(None, C1)
    assert ok
    assert torch.allclose(torch.trace(C_acc), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(C_acc, C1 / tr1, atol=1e-5)

    # Accumulate a second task matrix
    B = torch.randn(D, D)
    C2 = B @ B.t()
    tr2 = torch.trace(C2).item()
    C_acc, ok2 = accumulate_normalized(C_acc, C2)
    assert ok2
    assert torch.allclose(torch.trace(C_acc), torch.tensor(2.0), atol=1e-5)
    assert torch.allclose(C_acc, (C1 / tr1) + (C2 / tr2), atol=1e-5)

    # Zero matrix trace handled safely
    C_zero = torch.zeros(D, D)
    C_acc_new, ok_zero = accumulate_normalized(C_acc, C_zero)
    assert not ok_zero
    assert torch.equal(C_acc, C_acc_new)


def test_effective_rank():
    set_seed(0)
    # Rank 1 matrix: only 1 non-zero eigenvalue -> effective rank = exp(0) = 1.0
    v = torch.randn(D, 1)
    C_rank1 = v @ v.t()
    er1 = effective_rank(C_rank1)
    assert abs(er1 - 1.0) < 1e-4

    # Identity matrix: D equal eigenvalues -> effective rank = D
    C_eye = torch.eye(D)
    er_eye = effective_rank(C_eye)
    assert abs(er_eye - float(D)) < 1e-4

    # Zero matrix -> 0.0
    assert effective_rank(torch.zeros(D, D)) == 0.0


def test_controller_statistics_lifecycle():
    set_seed(0)
    args = {"alpha_in": 0.01, "alpha_hid": 0.01, "rho": 0.1}
    ctrl = SDRSIATController(args)
    assert not ctrl.has_statistics()

    C_hat_in = [torch.eye(D), torch.eye(D)]
    C_hat_hid = [torch.eye(R), torch.eye(R)]

    entry = ctrl.update_statistics(C_hat_in, C_hat_hid, task_id=0)
    assert ctrl.has_statistics()
    assert ctrl.n_stat_tasks == 1
    assert "erank_C_hid" in entry and len(entry["erank_C_hid"]) == 2
    assert abs(entry["erank_C_hid"][0] - float(R)) < 1e-4

    # Projectors can now be built
    ok = ctrl.build_projectors("cpu")
    assert ok
    assert len(ctrl.P_in) == 2 and len(ctrl.P_hid) == 2
    assert ctrl.P_in[0].shape == (D, D)
    assert ctrl.P_hid[0].shape == (R, R)


if __name__ == "__main__":
    run_all(globals())
