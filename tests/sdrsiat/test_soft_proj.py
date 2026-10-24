"""Soft projector: symmetry, spectrum, closed form, guards. Run: python tests/sdrsiat/test_soft_proj.py"""
from _common import D, run_all, set_seed  # noqa: F401  (also sets sys.path)
import torch

from models.sdrsiat import build_soft_projector


def _spd(d, rank=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, rank or d, generator=g)
    return A @ A.t()


def test_shape_dtype_symmetry_spectrum():
    C = _spd(D)
    P, U, lam = build_soft_projector(C, alpha=0.05)
    assert P.shape == (D, D) and P.dtype == torch.float32 and P.device.type == "cpu"
    assert torch.allclose(P, P.t(), atol=1e-6)
    ev = torch.linalg.eigvalsh(P)
    assert ev.min() >= -1e-5 and ev.max() <= 1 + 1e-5
    assert lam.min() >= 0 and lam.max() < 1


def test_closed_form_eigenvalues():
    alpha = 0.05
    C = _spd(D, seed=1)
    P, U, lam = build_soft_projector(C, alpha=alpha, eig_floor=0.0)
    e = torch.linalg.eigvalsh(0.5 * (C + C.t())).clamp_min(0)
    et = e / e.max()
    expected = alpha / (et + alpha)                  # eigenvalue of P along u_i = 1 - lambda_i
    got = torch.diagonal(U.t() @ P @ U)
    assert torch.allclose(got, expected, atol=1e-4), (got, expected)
    u_max = U[:, -1]
    assert torch.allclose(P @ u_max, (alpha / (1 + alpha)) * u_max, atol=1e-4)


def test_zero_matrix_gives_identity():
    P, _, lam = build_soft_projector(torch.zeros(D, D), alpha=0.1)
    assert torch.allclose(P, torch.eye(D)) and float(lam.abs().max()) == 0.0


def test_negative_eigenvalues_clamped():
    C = torch.diag(torch.tensor([-1.0, 0.0, 2.0, 4.0]))
    a = 0.1
    P, _, lam = build_soft_projector(C, alpha=a)
    expected = torch.tensor([1.0, 1.0, 1 - 0.5 / (0.5 + a), 1 - 1.0 / (1.0 + a)])
    assert torch.allclose(torch.diagonal(P), expected, atol=1e-5)
    assert torch.allclose(P, torch.diag(torch.diagonal(P)), atol=1e-6)


def test_asymmetric_input_is_symmetrised():
    C = _spd(D, seed=2)
    S = torch.randn(D, D)
    skew = S - S.t()
    P1, _, _ = build_soft_projector(C, alpha=0.05)
    P2, _, _ = build_soft_projector(C + 0.3 * skew, alpha=0.05)
    assert torch.allclose(P1, P2, atol=1e-4)


def test_alpha_monotonic_and_limits():
    C = _spd(D, seed=3)
    norms = [build_soft_projector(C, alpha=a)[0].norm().item() for a in (1e-3, 1e-2, 1e-1, 1.0)]
    assert all(norms[i] < norms[i + 1] for i in range(len(norms) - 1)), norms
    P_big, _, _ = build_soft_projector(C, alpha=1e6)
    assert torch.allclose(P_big, torch.eye(D), atol=1e-4)


def test_hard_projection_is_complement_projector():
    k = 3
    Q = torch.linalg.qr(torch.randn(D, k))[0]                # [D, k] orthonormal
    C = Q @ torch.diag(torch.tensor([3.0, 1.0, 0.5])) @ Q.t()
    P, _, _ = build_soft_projector(C, alpha=0.1, hard=True, hard_alpha=1e-8)
    expected = torch.eye(D) - Q @ Q.t()
    assert torch.allclose(P, expected, atol=1e-4)
    assert (P @ Q).abs().max() < 1e-4
    v = expected @ torch.randn(D)
    assert torch.allclose(P @ v, v, atol=1e-4)


def test_invalid_alpha_raises():
    for bad in (0.0, -1.0):
        try:
            build_soft_projector(torch.eye(D), alpha=bad)
        except ValueError:
            continue
        raise AssertionError("alpha={} should raise".format(bad))


if __name__ == "__main__":
    run_all(globals())
