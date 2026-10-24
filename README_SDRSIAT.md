# SD-RSIAT: Stable-Plastic Decoupled Representation-Steered Incremental Adapter Tuning

<!--
================================================================================
STEP 0 REPOSITORY ARCHITECTURE TABLE
================================================================================
| Component | File | Class / Function | Tensor Shapes | Weight Type / Storage Layout |
| :--- | :--- | :--- | :--- | :--- |
| Adapter Module | `network/vision_transformer_adapter.py` | `Adapter` | `down_proj.weight`: [r, d] = [64, 768]<br>`down_proj.bias`: [r] = [64]<br>`up_proj.weight`: [d, r] = [768, 64]<br>`up_proj.bias`: [d] = [768]<br>`scale`: scalar float or [1] | `nn.Linear` (weights stored [out_features, in_features]; computes $x W^T + b$) |
| Adapter Insertion in ViT | `network/vision_transformer_adapter.py` | `Block.forward` (layers 0..11) | `x`: [B, N, d] = [B, 197, 768]<br>`adapt_x`: [B, N, d] | Parallel bypass: `adapt_x = self.adaptmlp(x, add_residual=False)`, added back `x = x + adapt_x` |
| Feature Extraction / Backbone | `network/vision_transformer_adapter.py` | `VisionTransformer.forward_features` | Input: [B, 3, 224, 224]<br>Tokens: [B, 197, 768]<br>Output CLS: [B, 768] | CLS token extracted at index 0 after LayerNorm (`norm(x)[:, 0]`) |
| Classification Head | `network/classifier.py` | `SimpleContinualLinear` | `heads[i][0].weight`: [task_cls, 768]<br>logits: [B, total_classes] | `nn.ModuleList` of cosine linear heads; normalized forward $x W^T$ |
| Residual Autoencoder (Projector) | `utils/toolkit.py` | `AutoencoderSigmoid` | Encoder: [64, 768], [256, 64]<br>Decoder: [64, 256], [768, 64]<br>Output: [B, 768] | `nn.Sequential` of `nn.Linear` + `GELU` / `Sigmoid` + identity skip $f(x) + x$ |
| Losses ($L_{cos}, L_{align}, L_{orth}$) | `utils/loss.py`<br>`models/RSIAT_adapter.py` | `AngularPenaltySMLoss`<br>`Learner._inc_loss` / `_sdrsiat_compute_losses` | $L_{cos}$: scalar (CosFace, $s=40, m=0.1$)<br>$L_{align}$: scalar (MSE on features)<br>$L_{orth}$: scalar (cosine similarity with old prototypes) | Scalar losses. Weighted: $L_{inc} = \beta L_{align} + \gamma L_{orth}$ |
| SSIAT Prototypes, Drift & CA | `models/RSIAT_adapter.py`<br>`models/base.py` | `_compute_class_mean`<br>`displacement`<br>`_stage2_compact_classifier` | `_class_means`: [total_classes, 768]<br>`_class_covs`: [total_classes, 768, 768]<br>displacement gap: [known_classes, 768] | Class-conditional feature centroids and covariances; sampled Gaussian replay for head calibration |
| Training Loop & Optimizer Construction | `models/RSIAT_adapter.py` | `Learner._train`<br>`Learner._sdrsiat_inc_train` | Parameter groups: stable ($W_{down}$), plastic ($W_{up}$), FC, projector | SGD (momentum=0.9) or AdamW, cosine annealing LR scheduler |
================================================================================
-->

## 1. Overview of SD-RSIAT

SD-RSIAT extends RSIAT with **Stable-Plastic Decoupled Adapter Tuning** and **Soft Projection of Weight Updates**:
- **Decoupled Roles**:
  - $W_{down}$ is designated as the **stable** weight: learns at a reduced learning rate ($\rho \cdot \text{lr}_{up}$, default $\rho = 0.1$) and is steered by both classification loss $L_{cos}$ and representation alignment loss $\beta L_{align}$.
  - $W_{up}$ is designated as the **plastic** weight: learns at the full learning rate $\text{lr}_{up}$ and receives **only** classification gradient from $L_{cos}$. Representation alignment gradient $L_{align}$ is strictly prevented from reaching $W_{up}$.
  - Projector ($P_t$, residual autoencoder) receives gradients from $\beta L_{align} + \gamma L_{orth}$.
- **Uncentered Second-Moment Statistics**:
  - After each task $t \ge 1$, in evaluation mode (`eval()`, `torch.no_grad()`), forward hooks collect:
    $$C_{in} = \mathbb{E}[x^T x] \in \mathbb{R}^{d \times d}, \quad C_{hid} = \mathbb{E}[h^T h] \in \mathbb{R}^{r \times r}$$
    where $x$ is the input seen by $W_{down}$ and $h = \text{ReLU}(\text{down}(x))$ is the input seen by $W_{up}$.
  - Statistics are accumulated on CPU with trace normalization: $C \leftarrow C + \hat{C} / \text{trace}(\hat{C})$.
- **Soft Projector**:
  - At the start of task $t > 1$, soft projectors are constructed via eigenvalue decomposition:
    $$C = U \text{diag}(e) U^T, \quad \tilde{e} = e / \max(e), \quad \lambda_i = \frac{\tilde{e}_i}{\tilde{e}_i + \alpha}$$
    $$P = I - U \text{diag}(\lambda) U^T$$
    with separate regularization strengths $\alpha_{in}$ (for $W_{down}$) and $\alpha_{hid}$ (for $W_{up}$).
- **Update Projection (Not Gradient Projection)**:
  - Updates are projected **after** `optimizer.step()`, capturing momentum, weight decay, and gradient:
    $$\Delta W = W_{new} - W_{old}, \quad W \leftarrow W_{old} + \Delta W \cdot P$$
  - **Layout**: Since `nn.Linear` stores weights as $[\text{out}, \text{in}]$, the input dimension is on the right. Hence the projection is a **right-multiplication**:
    - $W_{down} \in \mathbb{R}^{r \times d} \implies \Delta W_{down} \cdot P_{in}$ where $P_{in} \in \mathbb{R}^{d \times d}$.
    - $W_{up} \in \mathbb{R}^{d \times r} \implies \Delta W_{up} \cdot P_{hid}$ where $P_{hid} \in \mathbb{R}^{r \times r}$.
  - Biases are not projected (optionally frozen via `--freeze_bias_incremental`).

---

## 2. Configuration Options

All SD-RSIAT options can be provided in JSON config files or passed directly via CLI flags. CLI flags take precedence over JSON configs:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--sdrsiat` | flag | `false` | Enable SD-RSIAT method. If absent/false, code executes pure original RSIAT. |
| `--alpha_in` | float | `0.01` | Soft projection damping $\alpha$ for $W_{down}$ ($P_{in} \in \mathbb{R}^{d \times d}$). |
| `--alpha_hid` | float | `0.01` | Soft projection damping $\alpha$ for $W_{up}$ ($P_{hid} \in \mathbb{R}^{r \times r}$). |
| `--rho` | float | `0.1` | Learning rate multiplier for stable parameters: $\text{lr}_{down} = \rho \cdot \text{lr}_{up}$. |
| `--freeze_bias_incremental` | flag | `false` | Freeze adapter biases during incremental tasks ($t > 1$). |
| `--hard_projection` | flag | `false` | Near-hard orthogonal complement projection ($\alpha \to 10^{-8}$). |
| `--stats_token_mode` | str | `"all"` | Tokens used for statistics: `"all"` (all 197 tokens), `"cls"` (CLS token only), `"patch"` (196 patch tokens). |
| `--sdrsiat_project` | str | `"both"` | Which weight updates to project: `"both"`, `"down"`, `"up"`, or `"none"`. |
| `--sdrsiat_swap_roles` | flag | `false` | Ablation: swap roles ($W_{up}$ stable, $W_{down}$ plastic). |
| `--sdrsiat_stats_max_batches`| int | `null` | Cap maximum number of batches for statistics pass (speeds up large datasets). |

---

## 3. Running Training & Ablations

### Standard Run with SD-RSIAT Config File:
```bash
# CIFAR-224
python main.py --config ./exps/sdrsiat_cifar224.json

# ImageNet-A
python main.py --config ./exps/sdrsiat_imageneta.json

# ImageNet-R
python main.py --config ./exps/sdrsiat_imagenetr.json
```

### Enabling SD-RSIAT via CLI over Original Config:
```bash
python main.py --config ./exps/adapter_cifar224.json --sdrsiat --alpha_in 0.01 --alpha_hid 0.01 --rho 0.1
python main.py --config ./exps/adapter_imageneta.json --sdrsiat --alpha_in 0.01 --alpha_hid 0.01 --rho 0.1
python main.py --config ./exps/adapter_imagenetr.json --sdrsiat --alpha_in 0.01 --alpha_hid 0.01 --rho 0.1
```

### Ablation Switches:

1. **Project Only $W_{down}$ (Ablate $W_{up}$ Projection)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --sdrsiat_project down
   ```

2. **Project Only $W_{up}$ (Ablate $W_{down}$ Projection)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --sdrsiat_project up
   ```

3. **No Update Projection (Decoupled LR + Routing Only)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --sdrsiat_project none
   ```

4. **Hard Projection ($\alpha \to 10^{-8}$)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --hard_projection
   ```

5. **Ablate LR Decoupling ($\rho = 1.0$)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --rho 1.0
   ```

6. **Freeze $W_{down}$ on Incremental Tasks ($\rho = 0.0$)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --rho 0.0
   ```

7. **Swapped Roles ($W_{up}$ Stable, $W_{down}$ Plastic)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --sdrsiat_swap_roles
   ```

8. **Token Selection Mode (CLS only vs Patch only)**:
   ```bash
   python main.py --config ./exps/sdrsiat_cifar224.json --stats_token_mode cls
   python main.py --config ./exps/sdrsiat_cifar224.json --stats_token_mode patch
   ```

---

## 4. Diagnostics & Logging

At the end of each task, SD-RSIAT automatically logs:
- `norm_dW_down` and `norm_dW_up`: Total Frobenius norms of updates $\|\Delta W_{down}\|$ and $\|\Delta W_{up}\|$.
- `ratio_down_over_up`: $\frac{\|\Delta W_{down}\|}{\|\Delta W_{up}\|}$.
- `norm_db_down` and `norm_db_up`: Total update norms of biases.
- `frac_energy_in_old_subspace_down` / `_up`: Fraction of update energy that fell into the protected subspace of old tasks.
- `stepwise_kept_energy_down` / `_up`: Ratio $\frac{\|\Delta W \cdot P\|^2}{\|\Delta W\|^2}$ across all optimizer steps.
- `erank_C_hid` / `erank_C_in`: Effective ranks $\exp(H(p))$ of the covariance / second-moment spectra per layer.

---

## 5. Colab / Kaggle Command Block to Run Tests

Paste and execute this block in a Colab or Kaggle cell:

```bash
# ==============================================================================
# Run all SD-RSIAT static unit tests (CPU, tiny dimensions, fixed seeds)
# ==============================================================================
python tests/sdrsiat/test_soft_proj.py
python tests/sdrsiat/test_hard_invariance.py
python tests/sdrsiat/test_grad_routing.py
python tests/sdrsiat/test_project_updates.py
python tests/sdrsiat/test_stats.py
python tests/sdrsiat/test_flag_off_equivalence.py
```
