"""
SD-RSIAT helpers (stable/plastic decoupled adapter tuning with soft projection
of weight UPDATES).  Only imported when the ``sdrsiat`` config flag is ON.

This module depends on ``torch`` only (no timm / data code) so that it can be
unit-tested on CPU with tiny dimensions (see tests/sdrsiat/).

Adapter (network/vision_transformer_adapter.py::Adapter), per ViT block:
    h = ReLU(x W_down^T + b_down)          W_down = down_proj.weight  [r, d]
    a = s * (h W_up^T + b_up)              W_up   = up_proj.weight    [d, r]
Both are ``nn.Linear`` => weights are stored [out, in] and compute x @ W^T.
Hence the INPUT dimension of each weight is its LAST axis and every update
projection in this file is a RIGHT multiplication:  delta[out, in] @ P[in, in].

Roles for incremental tasks (t > 1, i.e. ``_cur_task >= 1``):
    stable  = W_down (+ b_down)  : lr = rho * lr_up, receives L_cos + beta*L_align
    plastic = W_up   (+ b_up)    : lr = lr_up,       receives L_cos only
    (``sdrsiat_swap_roles`` swaps stable/plastic for the ablation.)
After optimizer.step() the UPDATE of W_down is projected with P_in (d x d) and
the UPDATE of W_up with P_hid (r x r), built from accumulated, trace-normalised
uncentered second moments of the inputs each weight actually sees.

DDP note: the repository uses (at most) nn.DataParallel, and training calls the
bare module (``_network_module_ptr``) anyway.  ``route_gradients`` uses
``torch.autograd.grad`` which BYPASSES DistributedDataParallel's gradient
all-reduce hooks.  Under DDP one would have to all-reduce the routed grads
manually (and all-reduce the statistics sums) before the optimizer step.
"""
import logging
import math

import torch
import torch.nn as nn

_EPS = 1e-12
TOKEN_MODES = ("all", "cls", "patch")
PROJECT_MODES = ("both", "down", "up", "none")
_LOG = "[SD-RSIAT]"


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def get_adapters(module):
    """Return adapter modules (anything with nn.Linear ``down_proj`` and
    ``up_proj``) in deterministic ``module.modules()`` order."""
    adapters = []
    for m in module.modules():
        if isinstance(getattr(m, "down_proj", None), nn.Linear) and \
                isinstance(getattr(m, "up_proj", None), nn.Linear):
            adapters.append(m)
    return adapters


def select_tokens(x, token_mode="all"):
    """Flatten an activation to [num_tokens, dim] according to ``token_mode``.

    x: [B, N, D] (ViT tokens, token 0 = CLS since VPT is off) or [B, D].
      'all'   -> every token the adapter processes (B*N rows)  [DEFAULT]
      'cls'   -> CLS token only (B rows)
      'patch' -> patch tokens only (B*(N-1) rows)
    """
    if token_mode not in TOKEN_MODES:
        raise ValueError("stats_token_mode must be one of {}, got {}".format(TOKEN_MODES, token_mode))
    if x.dim() == 2:
        return x
    if x.dim() != 3:
        return x.reshape(-1, x.shape[-1])
    if token_mode == "all":
        return x.reshape(-1, x.shape[-1])
    if token_mode == "cls":
        return x[:, 0, :]
    return x[:, 1:, :].reshape(-1, x.shape[-1])


def _default_input_getter(batch):
    # PILOT loaders yield (idx, inputs, targets)
    if isinstance(batch, (list, tuple)):
        return batch[1] if len(batch) >= 2 else batch[0]
    return batch


def _sym(C):
    return 0.5 * (C + C.transpose(0, 1))


def effective_rank(C):
    """exp(entropy) of the normalised (clamped) eigenvalue spectrum of C."""
    C = _sym(C.detach().float())
    e = torch.linalg.eigvalsh(C).clamp_min(0.0)
    s = e.sum()
    if s.item() <= _EPS:
        return 0.0
    p = e / s
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def subspace_energy_fraction(delta, U, lam):
    """Fraction of the energy of ``delta`` ([out, in], nn.Linear layout) that
    lies inside the (soft) old-task input subspace: sum_i lam_i ||delta u_i||^2
    / ||delta||^2, with (U, lam) from ``build_soft_projector``."""
    delta = delta.detach().float()
    den = delta.pow(2).sum()
    if den.item() <= _EPS:
        return 0.0
    coeff = delta @ U.to(device=delta.device, dtype=torch.float32)        # [out, in]
    num = (coeff.pow(2) * lam.to(device=delta.device, dtype=torch.float32).unsqueeze(0)).sum()
    return float((num / den).item())


# ----------------------------------------------------------------------------
# Phase 1: statistics collection
# ----------------------------------------------------------------------------
def collect_adapter_statistics(model, loader, adapters, device, token_mode="all", max_batches=None,
                               forward_fn=None, input_getter=None):
    """Collect per-adapter uncentered second moments with forward hooks.

    * ``C_hat_in[l]  = mean_tokens x^T x``  where x = INPUT of down_proj (what W_down sees)
    * ``C_hat_hid[l] = mean_tokens h^T h``  where h = INPUT of up_proj = ReLU(down) after
      dropout; in eval mode dropout is identity, so h is exactly what W_up sees.
    Runs in eval mode under no_grad; everything is detached; per-batch sums are
    moved to CPU and accumulated in float64, returned as CPU float32.
    Returns (C_hat_in list[d x d], C_hat_hid list[r x r], info dict).
    """
    if token_mode not in TOKEN_MODES:
        raise ValueError("stats_token_mode must be one of {}".format(TOKEN_MODES))
    forward_fn = model if forward_fn is None else forward_fn
    input_getter = _default_input_getter if input_getter is None else input_getter
    n = len(adapters)
    sums_in, sums_hid = [None] * n, [None] * n
    cnt_in, cnt_hid = [0] * n, [0] * n

    def _make_hook(idx, sums, cnts):
        def _hook(module, inputs, output):
            x = inputs[0].detach()
            t = select_tokens(x, token_mode).float()          # [T, D]
            xtx = (t.transpose(0, 1) @ t).double().cpu()       # [D, D]
            if sums[idx] is None:
                sums[idx] = xtx
            else:
                sums[idx] += xtx                                # plain CPU buffer, not a leaf param
            cnts[idx] += int(t.shape[0])
        return _hook

    was_training = model.training
    handles = []
    try:
        model.eval()
        for i, ad in enumerate(adapters):
            handles.append(ad.down_proj.register_forward_hook(_make_hook(i, sums_in, cnt_in)))
            handles.append(ad.up_proj.register_forward_hook(_make_hook(i, sums_hid, cnt_hid)))
        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                if max_batches is not None and b_idx >= int(max_batches):
                    break
                forward_fn(input_getter(batch).to(device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    def _finish(s, c, dim):
        if s is None or c == 0:
            return torch.zeros(dim, dim, dtype=torch.float32)
        return (s / float(c)).float()

    C_in = [_finish(sums_in[i], cnt_in[i], ad.down_proj.in_features) for i, ad in enumerate(adapters)]
    C_hid = [_finish(sums_hid[i], cnt_hid[i], ad.up_proj.in_features) for i, ad in enumerate(adapters)]
    return C_in, C_hid, {"tokens_in": cnt_in, "tokens_hid": cnt_hid}


def accumulate_normalized(C_acc, C_hat):
    """C_acc += C_hat / trace(C_hat)  (CPU float32). Skips (returns False) if trace ~ 0."""
    C_hat = C_hat.detach().float().cpu()
    if C_acc is None:
        C_acc = torch.zeros_like(C_hat)
    tr = torch.trace(C_hat)
    if (not torch.isfinite(tr)) or tr.item() <= _EPS:
        return C_acc, False
    return C_acc + C_hat / tr, True


# ----------------------------------------------------------------------------
# Phase 2: soft projector
# ----------------------------------------------------------------------------
def build_soft_projector(C, alpha, device=None, hard=False, hard_alpha=1e-8, eig_floor=1e-5):
    """P = I - U diag(lambda) U^T,  lambda_i = e~_i / (e~_i + alpha),  e~ = e / max(e).

    * C is symmetrised, eigh in float32, eigenvalues clamped >= 0.
    * max(e) ~ 0  -> lambda = 0 -> P = I (nothing to protect).
    * ``eig_floor``: relative eigenvalues below it are treated as exactly 0
      (guards float32 eigh noise, which otherwise is fully protected when alpha is tiny).
    * ``hard=True`` uses ``hard_alpha`` (tiny) -> near-orthogonal-complement projector.
    Returns (P [n,n] float32 on device, U, lambda).
    """
    a = float(hard_alpha if hard else alpha)
    if a <= 0:
        raise ValueError("alpha must be > 0")
    C = C.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(C).all():
        raise ValueError("non-finite statistics matrix")
    C = _sym(C)
    n = C.shape[0]
    eye = torch.eye(n, device=C.device, dtype=torch.float32)
    e, U = torch.linalg.eigh(C)
    e = e.clamp_min(0.0)
    emax = e.max()
    if emax.item() <= _EPS:
        lam = torch.zeros_like(e)
    else:
        et = e / emax
        if eig_floor is not None and eig_floor > 0:
            et = torch.where(et < eig_floor, torch.zeros_like(et), et)
        lam = et / (et + a)
    P = eye - (U * lam.unsqueeze(0)) @ U.transpose(0, 1)   # U diag(lam) U^T
    P = _sym(P)
    return P, U, lam


# ----------------------------------------------------------------------------
# Phase 3: gradient routing
# ----------------------------------------------------------------------------
def _safe_grad(loss, params, retain_graph):
    if len(params) == 0:
        return []
    if loss is None or not loss.requires_grad:
        return [None] * len(params)
    return list(torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True))


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def route_gradients(loss_cos, loss_align, loss_orth, beta, gamma, stable_params, plastic_params,
                    projector_params, classifier_params, other_params=()):
    """Compute per-loss gradients with torch.autograd.grad and write ``.grad``.

        stable     <- grad(L_cos) + grad(beta * L_align)
        plastic    <- grad(L_cos)                     (L_align is NEVER differentiated w.r.t. it)
        classifier <- grad(L_cos)
        projector  <- grad(beta * L_align + gamma * L_orth)
        other      <- grad(L_cos) + grad(beta * L_align)   (original RSIAT behaviour; expected empty)

    Call ``optimizer.zero_grad(set_to_none=True)`` before.  Params with no path
    get ``.grad = None`` (allow_unused=True), so the optimizer skips them.
    """
    stable, plastic = list(stable_params), list(plastic_params)
    proj, cls, other = list(projector_params), list(classifier_params), list(other_params)
    all_lists = stable + plastic + proj + cls + other
    if len({id(p) for p in all_lists}) != len(all_lists):
        raise ValueError("route_gradients: a parameter appears in more than one role")

    # (1) L_cos -> stable, plastic, classifier, other. Graph is retained for (2)/(3).
    cos_targets = stable + plastic + cls + other
    g_cos = _safe_grad(loss_cos, cos_targets, retain_graph=True)

    # (2) beta*L_align -> stable (+other) ONLY. Plastic params are not in the input list,
    #     so no align gradient is ever computed for / applied to them.
    align_targets = stable + other
    align_loss = None if loss_align is None else beta * loss_align
    g_align = _safe_grad(align_loss, align_targets, retain_graph=True)

    # (3) beta*L_align + gamma*L_orth -> projector only (last call: graph may be freed).
    terms = []
    if loss_align is not None:
        terms.append(beta * loss_align)
    if loss_orth is not None:
        terms.append(gamma * loss_orth)
    proj_loss = None
    for t in terms:
        proj_loss = t if proj_loss is None else proj_loss + t
    g_proj = _safe_grad(proj_loss, proj, retain_graph=False)

    grads = {}
    for p, g in zip(cos_targets, g_cos):
        grads[id(p)] = g
    for p, g in zip(align_targets, g_align):
        grads[id(p)] = _add(grads.get(id(p)), g)
    for p, g in zip(proj, g_proj):
        grads[id(p)] = g
    for p in all_lists:
        g = grads.get(id(p))
        p.grad = None if g is None else g.detach()
    return grads


# ----------------------------------------------------------------------------
# Controller: config, roles/param groups, projection, statistics, diagnostics
# ----------------------------------------------------------------------------
def _cfg(args, key, default, cast):
    v = args.get(key, None) if hasattr(args, "get") else None
    if v is None:
        return default
    if cast is bool and isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return cast(v)


class SDRSIATController(object):
    def __init__(self, args):
        self.alpha_in = _cfg(args, "alpha_in", 0.01, float)
        self.alpha_hid = _cfg(args, "alpha_hid", 0.01, float)
        self.rho = _cfg(args, "rho", 0.1, float)
        self.freeze_bias_incremental = _cfg(args, "freeze_bias_incremental", False, bool)
        self.hard_projection = _cfg(args, "hard_projection", False, bool)
        self.hard_alpha = _cfg(args, "sdrsiat_hard_alpha", 1e-8, float)
        self.eig_floor = _cfg(args, "sdrsiat_eig_floor", 1e-5, float)
        self.stats_token_mode = _cfg(args, "stats_token_mode", "all", str)
        self.stats_max_batches = args.get("sdrsiat_stats_max_batches", None) if hasattr(args, "get") else None
        project = _cfg(args, "sdrsiat_project", "both", str)
        self.swap_roles = _cfg(args, "sdrsiat_swap_roles", False, bool)
        self.adapter_weight_decay = args.get("sdrsiat_adapter_weight_decay", None) if hasattr(args, "get") else None

        if self.alpha_in <= 0 or self.alpha_hid <= 0 or self.hard_alpha <= 0:
            raise ValueError("alpha_in / alpha_hid / sdrsiat_hard_alpha must be > 0")
        if self.rho < 0:
            raise ValueError("rho must be >= 0")
        if self.stats_token_mode not in TOKEN_MODES:
            raise ValueError("stats_token_mode must be one of {}".format(TOKEN_MODES))
        if project not in PROJECT_MODES:
            raise ValueError("sdrsiat_project must be one of {}".format(PROJECT_MODES))
        self.project_down = project in ("both", "down")
        self.project_up = project in ("both", "up")

        self.C_in, self.C_hid = None, None      # lists of CPU float32 matrices
        self.n_stat_tasks = 0
        self.P_in, self.P_hid = [], []          # device float32, built at start of task t>1
        self.eig_in, self.eig_hid = [], []      # (U, lam) for diagnostics
        self._task_start = None
        self._step_stats = {}
        self.history = []
        logging.info("{} config: alpha_in={} alpha_hid={} rho={} freeze_bias_incremental={} hard_projection={} "
                     "(hard_alpha={}) eig_floor={} stats_token_mode={} stats_max_batches={} project={} "
                     "swap_roles={} adapter_weight_decay={}".format(
                         _LOG, self.alpha_in, self.alpha_hid, self.rho, self.freeze_bias_incremental,
                         self.hard_projection, self.hard_alpha, self.eig_floor, self.stats_token_mode,
                         self.stats_max_batches, project, self.swap_roles, self.adapter_weight_decay))

    # ------------------------------------------------------------------ roles
    def split_roles(self, adapters):
        down_w = [a.down_proj.weight for a in adapters]
        down_b = [a.down_proj.bias for a in adapters if a.down_proj.bias is not None]
        up_w = [a.up_proj.weight for a in adapters]
        up_b = [a.up_proj.bias for a in adapters if a.up_proj.bias is not None]
        known = {id(p) for p in down_w + down_b + up_w + up_b}
        # any other adapter parameter (learnable scale / adapter LayerNorm; none in the
        # default config) is treated as plastic: L_cos only, never projected.
        extra = [p for a in adapters for p in a.parameters() if id(p) not in known]
        if self.swap_roles:
            s_w, s_b, p_w, p_b = up_w, up_b, down_w, down_b
        else:
            s_w, s_b, p_w, p_b = down_w, down_b, up_w, up_b
        frozen = []
        if self.freeze_bias_incremental:
            frozen = list(s_b) + list(p_b)
            s_b, p_b = [], []
        rg = lambda ps: [p for p in ps if p.requires_grad]
        roles = {"stable_w": rg(s_w), "stable_b": rg(s_b), "plastic_w": rg(p_w), "plastic_b": rg(p_b),
                 "plastic_extra": rg(extra), "frozen": frozen,
                 "all_adapter": down_w + down_b + up_w + up_b + extra}
        roles["stable"] = roles["stable_w"] + roles["stable_b"]
        roles["plastic"] = roles["plastic_w"] + roles["plastic_b"] + roles["plastic_extra"]
        return roles

    def build_param_groups(self, roles, base_lr, base_wd, fc_params, fc_lr, fc_wd,
                           proj_params, proj_lr, proj_wd, other_params=()):
        """Stable lr = rho * base_lr; plastic lr = base_lr; fc / projector as original.
        Adapter weight decay defaults to the original value: it is part of the
        optimizer update delta and is therefore projected too (cannot leak)."""
        aw = base_wd if self.adapter_weight_decay is None else float(self.adapter_weight_decay)
        specs = [
            ("sd_stable_w", roles["stable_w"], self.rho * base_lr, aw),
            ("sd_stable_b", roles["stable_b"], self.rho * base_lr, base_wd),
            ("sd_plastic_w", roles["plastic_w"], base_lr, aw),
            ("sd_plastic_b", roles["plastic_b"] + roles["plastic_extra"], base_lr, base_wd),
            ("convnet_other", list(other_params), base_lr, base_wd),
            ("fc", list(fc_params), fc_lr, fc_wd),
            ("projector", list(proj_params), proj_lr, proj_wd),
        ]
        return [{"params": list(ps), "lr": lr, "weight_decay": wd, "name": name}
                for name, ps, lr, wd in specs if len(ps) > 0]

    @staticmethod
    def check_param_groups(groups, required, excluded=()):
        seen = set()
        for g in groups:
            for p in g["params"]:
                if id(p) in seen:
                    raise RuntimeError("{} parameter duplicated across optimizer groups ({})".format(_LOG, g.get("name")))
                seen.add(id(p))
        exc = {id(p) for p in excluded}
        missing = [p for p in required if id(p) not in seen and id(p) not in exc]
        if missing:
            raise RuntimeError("{} {} trainable parameter(s) missing from optimizer groups".format(_LOG, len(missing)))
        if seen & exc:
            raise RuntimeError("{} an intentionally frozen parameter is in the optimizer".format(_LOG))
        return True

    # ------------------------------------------------------------- statistics
    def has_statistics(self):
        return self.C_in is not None and self.n_stat_tasks > 0

    def update_statistics(self, C_hat_in, C_hat_hid, task_id=None):
        if self.C_in is None:
            self.C_in = [None] * len(C_hat_in)
            self.C_hid = [None] * len(C_hat_hid)
        skipped = 0
        for i in range(len(C_hat_in)):
            self.C_in[i], ok1 = accumulate_normalized(self.C_in[i], C_hat_in[i])
            self.C_hid[i], ok2 = accumulate_normalized(self.C_hid[i], C_hat_hid[i])
            skipped += int(not ok1) + int(not ok2)
        self.n_stat_tasks += 1
        er_hid = [effective_rank(C) for C in self.C_hid]
        er_in = [effective_rank(C) for C in self.C_in]
        entry = {"task": task_id, "type": "stats", "erank_C_hid": er_hid, "erank_C_in": er_in, "skipped_zero_trace": skipped}
        self.history.append(entry)
        logging.info("{} task {} stats accumulated ({} task(s)); skipped zero-trace: {}".format(
            _LOG, task_id, self.n_stat_tasks, skipped))
        logging.info("{} task {} effective rank C_hid per layer: {}".format(_LOG, task_id, [round(v, 2) for v in er_hid]))
        logging.info("{} task {} effective rank C_in  per layer: {}".format(_LOG, task_id, [round(v, 2) for v in er_in]))
        return entry

    def build_projectors(self, device):
        """Move stats to ``device`` only here; build P_in / P_hid per layer."""
        self.P_in, self.P_hid, self.eig_in, self.eig_hid = [], [], [], []
        if not self.has_statistics():
            logging.info("{} no statistics yet -> no projection".format(_LOG))
            return False
        for C_in, C_hid in zip(self.C_in, self.C_hid):
            if self.project_down:
                P, U, lam = build_soft_projector(C_in, self.alpha_in, device, self.hard_projection,
                                                 self.hard_alpha, self.eig_floor)
                self.P_in.append(P); self.eig_in.append((U, lam))
            else:
                self.P_in.append(None); self.eig_in.append(None)
            if self.project_up:
                P, U, lam = build_soft_projector(C_hid, self.alpha_hid, device, self.hard_projection,
                                                 self.hard_alpha, self.eig_floor)
                self.P_hid.append(P); self.eig_hid.append((U, lam))
            else:
                self.P_hid.append(None); self.eig_hid.append(None)
        sum_in = [round(float(e[1].sum().item()), 2) for e in self.eig_in if e is not None]
        sum_hid = [round(float(e[1].sum().item()), 2) for e in self.eig_hid if e is not None]
        logging.info("{} projectors built (hard={}): sum(lambda_in) per layer {}".format(_LOG, self.hard_projection, sum_in))
        logging.info("{} projectors built (hard={}): sum(lambda_hid) per layer {}".format(_LOG, self.hard_projection, sum_hid))
        return True

    # ------------------------------------------------------- update projection
    def snapshot(self, adapters):
        """Clone (detached) the weights whose update will be projected. Call BEFORE optimizer.step()."""
        snaps = []
        with torch.no_grad():
            for i, a in enumerate(adapters):
                wd = a.down_proj.weight.detach().clone() if (i < len(self.P_in) and self.P_in[i] is not None) else None
                wu = a.up_proj.weight.detach().clone() if (i < len(self.P_hid) and self.P_hid[i] is not None) else None
                snaps.append((wd, wu))
        return snaps

    def _project_one(self, W, W_old, P, key):
        # caller holds torch.no_grad(); W is a leaf nn.Parameter -> only .copy_() is used.
        delta = (W - W_old).float()                               # [out, in]
        delta_p = delta @ P.to(device=delta.device)               # RIGHT-multiply: input dim is last axis
        W.copy_((W_old.float() + delta_p).to(W.dtype))
        st = self._step_stats
        st[key + "_raw"] = st.get(key + "_raw", 0.0) + delta.pow(2).sum()
        st[key + "_kept"] = st.get(key + "_kept", 0.0) + delta_p.pow(2).sum()

    def project_updates(self, adapters, snaps):
        """W = W_old + (W_new - W_old) @ P  for every snapshotted weight. Biases untouched.
        Call AFTER optimizer.step(); everything the optimizer did (momentum, weight decay,
        AdamW decoupled decay) is inside delta and is therefore projected."""
        with torch.no_grad():
            for i, (a, (wd_old, wu_old)) in enumerate(zip(adapters, snaps)):
                if wd_old is not None:
                    # LAYOUT: down_proj = nn.Linear(d, r) -> weight [r, d] (out, in); h_pre = x @ W^T.
                    # Input dim d is the LAST axis => delta[r, d] @ P_in[d, d]  (right side).
                    self._project_one(a.down_proj.weight, wd_old, self.P_in[i], "down")
                if wu_old is not None:
                    # LAYOUT: up_proj = nn.Linear(r, d) -> weight [d, r] (out, in); a = h @ W^T.
                    # Input (hidden) dim r is the LAST axis => delta[d, r] @ P_hid[r, r]  (right side).
                    self._project_one(a.up_proj.weight, wu_old, self.P_hid[i], "up")

    # ------------------------------------------------------------ diagnostics
    def begin_task(self, adapters):
        with torch.no_grad():
            self._task_start = [{
                "dw": a.down_proj.weight.detach().float().cpu().clone(),
                "uw": a.up_proj.weight.detach().float().cpu().clone(),
                "db": None if a.down_proj.bias is None else a.down_proj.bias.detach().float().cpu().clone(),
                "ub": None if a.up_proj.bias is None else a.up_proj.bias.detach().float().cpu().clone(),
            } for a in adapters]
        self._step_stats = {}

    def end_task(self, adapters, task_id):
        if self._task_start is None:
            return None
        sq = {"dw": 0.0, "uw": 0.0, "db": 0.0, "ub": 0.0}
        inside = {"dw": 0.0, "uw": 0.0}
        has_eig = {"dw": False, "uw": False}
        with torch.no_grad():
            for i, (a, s) in enumerate(zip(adapters, self._task_start)):
                dW_d = a.down_proj.weight.detach().float().cpu() - s["dw"]
                dW_u = a.up_proj.weight.detach().float().cpu() - s["uw"]
                sq["dw"] += float(dW_d.pow(2).sum()); sq["uw"] += float(dW_u.pow(2).sum())
                if s["db"] is not None:
                    sq["db"] += float((a.down_proj.bias.detach().float().cpu() - s["db"]).pow(2).sum())
                if s["ub"] is not None:
                    sq["ub"] += float((a.up_proj.bias.detach().float().cpu() - s["ub"]).pow(2).sum())
                if i < len(self.eig_in) and self.eig_in[i] is not None:
                    U, lam = self.eig_in[i]
                    inside["dw"] += subspace_energy_fraction(dW_d, U.cpu(), lam.cpu()) * float(dW_d.pow(2).sum())
                    has_eig["dw"] = True
                if i < len(self.eig_hid) and self.eig_hid[i] is not None:
                    U, lam = self.eig_hid[i]
                    inside["uw"] += subspace_energy_fraction(dW_u, U.cpu(), lam.cpu()) * float(dW_u.pow(2).sum())
                    has_eig["uw"] = True
        n_dw, n_uw = math.sqrt(sq["dw"]), math.sqrt(sq["uw"])
        entry = {
            "task": task_id, "type": "update",
            "norm_dW_down": n_dw, "norm_dW_up": n_uw,
            "ratio_down_over_up": n_dw / n_uw if n_uw > _EPS else float("inf"),
            "norm_db_down": math.sqrt(sq["db"]), "norm_db_up": math.sqrt(sq["ub"]),
            "frac_energy_in_old_subspace_down": (inside["dw"] / sq["dw"] if sq["dw"] > _EPS else 0.0) if has_eig["dw"] else None,
            "frac_energy_in_old_subspace_up": (inside["uw"] / sq["uw"] if sq["uw"] > _EPS else 0.0) if has_eig["uw"] else None,
        }
        for key in ("down", "up"):
            raw = self._step_stats.get(key + "_raw", None)
            kept = self._step_stats.get(key + "_kept", None)
            if raw is not None:
                raw_v, kept_v = float(raw), float(kept)
                entry["stepwise_kept_energy_" + key] = kept_v / raw_v if raw_v > _EPS else 0.0
        self.history.append(entry)
        logging.info("{} task {} update diagnostics: {}".format(_LOG, task_id, entry))
        self._task_start = None
        return entry
