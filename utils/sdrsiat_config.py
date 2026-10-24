"""
CLI plumbing for SD-RSIAT config flags (no torch import; unit-testable).

Precedence: CLI value (if given) > JSON value > learner default.
When no SD-RSIAT CLI option is passed, the merged config is byte-for-byte the
same dict the original main.py produced (the keys are removed from the
argparse namespace before merging), so the flag-OFF path is unchanged.
"""

SDRSIAT_CLI_KEYS = (
    "sdrsiat",
    "alpha_in",
    "alpha_hid",
    "rho",
    "freeze_bias_incremental",
    "hard_projection",
    "stats_token_mode",
    "sdrsiat_project",
    "sdrsiat_swap_roles",
    "sdrsiat_stats_max_batches",
)


def add_sdrsiat_cli_args(parser):
    g = parser.add_argument_group("SD-RSIAT (all default to None = use JSON / learner default)")
    g.add_argument("--sdrsiat", action="store_true", default=None, help="enable SD-RSIAT")
    g.add_argument("--alpha_in", type=float, default=None, help="soft-projection alpha for P_in (W_down)")
    g.add_argument("--alpha_hid", type=float, default=None, help="soft-projection alpha for P_hid (W_up)")
    g.add_argument("--rho", type=float, default=None, help="lr(stable) = rho * lr(plastic)")
    g.add_argument("--freeze_bias_incremental", action="store_true", default=None,
                   help="freeze adapter biases for tasks t>1")
    g.add_argument("--hard_projection", action="store_true", default=None, help="alpha -> tiny (hard projection)")
    g.add_argument("--stats_token_mode", type=str, default=None, choices=["all", "cls", "patch"])
    g.add_argument("--sdrsiat_project", type=str, default=None, choices=["both", "down", "up", "none"],
                   help="which weight updates are projected (ablation)")
    g.add_argument("--sdrsiat_swap_roles", action="store_true", default=None,
                   help="ablation: W_up stable (rho, L_align), W_down plastic")
    g.add_argument("--sdrsiat_stats_max_batches", type=int, default=None,
                   help="cap the number of batches used for statistics")
    return parser


def pop_sdrsiat_cli(namespace):
    """Remove SD-RSIAT keys from the argparse Namespace and return them as a dict."""
    cli = {}
    for k in SDRSIAT_CLI_KEYS:
        if hasattr(namespace, k):
            cli[k] = getattr(namespace, k)
            delattr(namespace, k)
    return cli


def apply_sdrsiat_overrides(config, cli):
    for k, v in cli.items():
        if v is not None:
            config[k] = v
    return config
