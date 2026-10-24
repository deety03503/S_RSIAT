# SD-RSIAT Static Verification Log

> **STATUS: STATIC REVIEW ONLY - NOT EXECUTED**  
> All checks performed via formal mathematical review, dimensional tracing, and control-flow inspection. No PyTorch runtime or GPU environment was invoked.

---

## 1. Dimensional Trace (d=16, r=4, batch=2, tokens=5)

| Tensor / Variable | Mathematical Meaning | Shape | Notes on Layout & Broadcasting |
| :--- | :--- | :--- | :--- |
| `x` | ViT Block Input | `[2, 5, 16]` | Batch 2, 5 tokens (1 CLS + 4 patch), dim 16 |
| `select_tokens(x, 'all')` | Flattened Tokens | `[10, 16]` | `10 = 2 * 5`. Matrix mult `t.t() @ t` produces `[16, 16]` |
| `select_tokens(x, 'cls')` | CLS Token Only | `[2, 16]` | Matrix mult `t.t() @ t` produces `[16, 16]` |
| `select_tokens(x, 'patch')` | Patch Tokens | `[8, 16]` | `8 = 2 * 4`. Matrix mult `t.t() @ t` produces `[16, 16]` |
| `down_proj.weight` ($W_{down}$) | nn.Linear Weight | `[4, 16]` | Stored `[out_features, in_features]`. Computes $x W^T$ |
| `down_proj.bias` ($b_{down}$) | nn.Linear Bias | `[4]` | Added with automatic 1D broadcast across tokens |
| `h` | Hidden Activation | `[2, 5, 4]` | $h = \text{ReLU}(x W_{down}^T + b_{down})$ |
| `up_proj.weight` ($W_{up}$) | nn.Linear Weight | `[16, 4]` | Stored `[out_features, in_features]`. Computes $h W^T$ |
| `up_proj.bias` ($b_{up}$) | nn.Linear Bias | `[16]` | Added with automatic 1D broadcast across tokens |
| `C_in` ($\hat{C}_{in}$) | Second Moment (Input) | `[16, 16]` | Symmetric positive semi-definite |
| `C_hid` ($\hat{C}_{hid}$) | Second Moment (Hidden) | `[4, 4]` | Symmetric positive semi-definite |
| `P_in` | Soft Projector (Input) | `[16, 16]` | $I - U \text{diag}(\lambda) U^T \in \mathbb{R}^{16 \times 16}$ |
| `P_hid` | Soft Projector (Hidden) | `[4, 4]` | $I - U \text{diag}(\lambda) U^T \in \mathbb{R}^{4 \times 4}$ |
| $\Delta W_{down}$ | Update of $W_{down}$ | `[4, 16]` | $\Delta W_{down} \cdot P_{in} = [4, 16] \times [16, 16] = [4, 16]$. No broadcasting. |
| $\Delta W_{up}$ | Update of $W_{up}$ | `[16, 4]` | $\Delta W_{up} \cdot P_{hid} = [16, 4] \times [4, 4] = [16, 4]$. No broadcasting. |
| `U` ($\text{eig}_{in}$) | Eigenvectors of $C_{in}$ | `[16, 16]` | Orthonormal column basis |
| `lambda` ($\text{eig}_{in}$) | Filter Factors | `[16]` | Damped factors $\tilde{e} / (\tilde{e} + \alpha) \in [0, 1)$ |

---

## 2. Global Safety Checklist

| Verification Question | Status | Rationale / Proof |
| :--- | :---: | :--- |
| **Flag OFF leaves original path unchanged?** | **YES** | If `--sdrsiat` is omitted/false, `self._sdrsiat` is `False`. `_train()` takes the original `else` branch, exactly preserving the baseline param groups, losses, optimizer calls, and drift compensation without any hooks or projections. |
| **Projection applied to the UPDATE, correct side for layout?** | **YES** | `delta = W - W_old` captured around `optimizer.step()`. Since PyTorch `nn.Linear` weights are `[out, in]`, the input dimension is on the right; projection is strictly `delta @ P`. |
| **$W_{up}$ unreachable from $L_{align}$?** | **YES** | In `route_gradients()`, $W_{up}$ and $b_{up}$ are only included in the `cos_targets` target list. The `torch.autograd.grad(beta * loss_align, align_targets)` call explicitly specifies `align_targets = stable + other`. Plastic parameters are omitted, preventing any $L_{align}$ gradient flow. |
| **Any in-place edit of a leaf outside `no_grad`?** | **YES** (None exists) | In `models/sdrsiat.py`, weight modifications happen inside `torch.no_grad()` via `.copy_()`, preserving leaf parameter identity. |
| **Optimizer groups complete, no duplicates?** | **YES** | Checked via `check_param_groups()` before training begins: tests that all requires-grad adapter, FC, and projector parameters are present in exactly one group and frozen biases are excluded. |
| **`autograd.grad`: `allow_unused`, `retain_graph`, `None` handling correct?** | **YES** | All backward grad calls pass `allow_unused=True`; `retain_graph=True` is used for step (1) and (2), then freed on step (3); `None` gradients are converted to `0` or left as `None` (for unused params) to be safely skipped by optimizers. |
| **Device/dtype consistent?** | **YES** | Second-moment accumulation is performed on CPU in float64 / float32. Projectors are cast to `torch.float32` on the target model device when built. Projections convert float tensors matching parameter dtypes. |
| **$C$ symmetrized, eigenvalues clamped, zero-division guarded?** | **YES** | $C = 0.5 \cdot (C + C^T)$; eigenvalues clamped with `.clamp_min(0.0)`; max eigenvalue normalization checks `emax.item() <= 1e-12` and falls back to $\lambda = 0 \implies P = I$. Small noise eigenvalues below `eig_floor=1e-5` are zeroed. |
| **Stats collected in eval, `no_grad`, correct tokens, detached?** | **YES** | Forward pass for statistics runs under `model.eval()`, `torch.no_grad()`. Activations are explicitly `.detach()`. Model train/eval state is restored in a `try...finally` block. |
| **DDP implications noted?** | **YES** | Documented in `models/sdrsiat.py` and `README_SDRSIAT.md`. Code supports `nn.DataParallel` out of the box. For DDP, manual all-reduce of gradients and statistics would be required due to `torch.autograd.grad`. |
| **Weight decay cannot leak past the projection?** | **YES** | Weight decay acts during `optimizer.step()`, changing $W_{new}$. Since $\Delta W = W_{new} - W_{old}$ is calculated after `optimizer.step()`, weight decay update is contained within $\Delta W$ and projected by $P$. |

---

## 3. Phase Log

### Phase 1: Statistics Collection
- **Files Modified / Created**: `models/sdrsiat.py`
- **Functions**: `collect_adapter_statistics`, `accumulate_normalized`, `select_tokens`
- **Shape Review**: $x \in \mathbb{R}^{B \times N \times d} \to t \in \mathbb{R}^{TN \times d}$; $t^T t \in \mathbb{R}^{d \times d}$. Symmetric PSD matrices. Trace normalization division safe via epsilon check.
- **Review Result**: **STATIC REVIEW: OK**

### Phase 2: Soft Projector Builder
- **Files Modified / Created**: `models/sdrsiat.py`
- **Functions**: `build_soft_projector`, `effective_rank`, `subspace_energy_fraction`
- **Spectral Review**: $P = I - U \text{diag}(\lambda) U^T$. Since $\lambda_i \in [0, 1)$, eigenvalues of $P$ lie in $(0, 1]$. For $\alpha \to \infty$, $\lambda \to 0 \implies P \to I$. For $\alpha \to 0$ (`hard_projection`), $\lambda \to 1 \implies P \to I - U U^T$ (orthogonal complement).
- **Review Result**: **STATIC REVIEW: OK**

### Phase 3: Gradient Routing & Param Groups
- **Files Modified / Created**: `models/sdrsiat.py`, `models/RSIAT_adapter.py`
- **Functions**: `route_gradients`, `SDRSIATController.split_roles`, `SDRSIATController.build_param_groups`, `SDRSIATController.check_param_groups`
- **Routing Review**:
  $$\nabla_{W_{down}} = \nabla_{L_{cos}} + \beta \nabla_{L_{align}}$$
  $$\nabla_{W_{up}} = \nabla_{L_{cos}}$$
  $$\nabla_{P_t} = \beta \nabla_{L_{align}} + \gamma \nabla_{L_{orth}}$$
  $$\nabla_{FC} = \nabla_{L_{cos}}$$
  No parameter is duplicated across groups. $W_{up}$ is provably absent from $L_{align}$ graph call.
- **Review Result**: **STATIC REVIEW: OK**

### Phase 4: Update Projection & Bias Handling
- **Files Modified / Created**: `models/sdrsiat.py`, `models/RSIAT_adapter.py`
- **Functions**: `SDRSIATController.snapshot`, `SDRSIATController.project_updates`, `_project_one`
- **Layout Review**:
  $W_{down} \in \mathbb{R}^{r \times d} \implies \Delta W_{down} \cdot P_{in}$ where $P_{in} \in \mathbb{R}^{d \times d}$.
  $W_{up} \in \mathbb{R}^{d \times r} \implies \Delta W_{up} \cdot P_{hid}$ where $P_{hid} \in \mathbb{R}^{r \times r}$.
  Right-multiplication matches PyTorch $[out, in]$ weight convention for $x W^T$.
  `W.copy_(...)` used under `torch.no_grad()`, preserving PyTorch leaf status.
- **Review Result**: **STATIC REVIEW: OK**

### Phase 5 & 6: Integration, Config Flags, and Diagnostics
- **Files Modified / Created**: `models/RSIAT_adapter.py`, `utils/sdrsiat_config.py`, `main.py`, `exps/sdrsiat_cifar224.json`
- **Functions**: `_sdrsiat_inc_train`, `_sdrsiat_compute_losses`, `_sdrsiat_collect_statistics`, `add_sdrsiat_cli_args`, `pop_sdrsiat_cli`, `apply_sdrsiat_overrides`
- **Integration Review**: Cleanly flag-gated under `self._sdrsiat`. When flag is false, code executes unmodified baseline code path. CLI flags override JSON values if and only if provided.
- **Review Result**: **STATIC REVIEW: OK**

### Phase 7: Static Test Suites
- **Files Created**:
  - `tests/sdrsiat/_common.py`
  - `tests/sdrsiat/test_soft_proj.py`
  - `tests/sdrsiat/test_hard_invariance.py`
  - `tests/sdrsiat/test_grad_routing.py`
  - `tests/sdrsiat/test_project_updates.py`
  - `tests/sdrsiat/test_stats.py`
  - `tests/sdrsiat/test_flag_off_equivalence.py`
- **Review Result**: **STATIC REVIEW: OK**

---

## 4. Remaining Hardware / Runtime Risks to Verify on Colab/Kaggle

1. **GPU Memory Footprint During Statistics Pass**:
   - $C_{in}$ is $768 \times 768 \times 4 \text{ bytes} \approx 2.36\text{ MB}$ per layer ($28.3\text{ MB}$ for 12 layers). CPU memory accumulation is lightweight.
   - Forward pass uses batch size from config (`batch_size=64`) without gradients (`no_grad()`). On GPUs with limited VRAM (e.g. 15GB T4), this is well within memory limits.
2. **Eigenvalue Decomposition ($d=768$) Speed**:
   - Running `torch.linalg.eigh` on twelve $768 \times 768$ symmetric matrices happens once per incremental task. On CPU/GPU this takes $<0.1\text{ s}$ per task.
3. **Multi-GPU / DistributedDataParallel (if ever enabled)**:
   - Current repo uses `DataParallel` or single GPU. If user moves to `DistributedDataParallel`, manual all-reduce of gradients in `route_gradients` would be needed because `torch.autograd.grad` does not trigger DDP backward hooks.
