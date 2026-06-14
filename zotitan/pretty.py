"""ANSI colour helpers for training output."""
import dataclasses
import sys

BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YEL   = "\033[33m"
MAG   = "\033[35m"
BLUE  = "\033[34m"
RED   = "\033[31m"


_BANNER_ART = r"""
 ________   _____    ______        __
/\_____  \ /\  __`\ /\__  _\  __  /\ \__
\/____//'/'\ \ \/\ \\/_/\ \/ /\_\ \ \ ,_\     __       ___
     //'/'  \ \ \ \ \  \ \ \ \/\ \ \ \ \/   /'__`\   /' _ `\
    //'/'__  \ \ \_\ \  \ \ \ \ \ \ \ \ \_ /\ \L\.\_ /\ \/\ \
    /\_______\\ \_____\  \ \_\ \ \_\ \ \__\\ \__/.\_\\ \_\ \_\
    \/_______/ \/_____/   \/_/  \/_/  \/__/ \/__/\/_/ \/_/\/_/
      A Training Codebase for Evolutionary Strategies
"""

# bright green for the figlet, bright blue for the tagline prose.
_BGREEN, _BBLUE = "\033[92m", "\033[94m"
_TAGLINE = "A Training Codebase for Evolutionary Strategies"


def _colorize_banner(art: str) -> str:
    """Bright-green the art; recolor the tagline prose bright blue."""
    lines = []
    for line in art.split("\n"):
        if _TAGLINE in line:
            start = line.index(_TAGLINE)
            end = start + len(_TAGLINE)
            line = f"{line[:start]}{_BBLUE}{_TAGLINE}{_BGREEN}{line[end:]}"
        lines.append(line)
    return f"{_BGREEN}{chr(10).join(lines)}{RESET}"


# Color only when writing to a terminal; piped/redirected --help stays plain.
BANNER = _colorize_banner(_BANNER_ART) if sys.stdout.isatty() else _BANNER_ART


def fmt_config(cfg, indent: int = 0) -> str:
    """Recursively format a dataclass as indented, coloured key: value lines."""
    lines = []
    pad = "  " * indent
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if dataclasses.is_dataclass(val):
            lines.append(f"{pad}{CYAN}{BOLD}{f.name}{RESET}")
            lines.append(fmt_config(val, indent + 1))
        else:
            lines.append(f"{pad}{CYAN}{f.name}{RESET}  {val}")
    return "\n".join(lines)


def print_config(cfg):
    print(f"\n{BOLD}{type(cfg).__name__}{RESET}")
    print(fmt_config(cfg))
    print()


# Optimizer-diagnostic keys shown explicitly (or deliberately hidden) by print_zo_step;
# every other float key in the metrics dict is an objective metric and printed generically.
_ZO_SHOWN  = {"loss", "proj_grad", "lr", "clip_frac", "step_time"}
_ZO_HIDDEN = {"proj_grad_clipped", "clip_tau"}


def _fmt_step_time(dt) -> str:
    return f"sec {DIM}{dt:.1f}s{RESET}"


def print_eval_results(results: dict, **extra):
    """Print objective.evaluate() metrics as space-separated ``k=v`` pairs, with
    floats fixed to 4 decimals and any ``extra`` context (e.g. split) appended."""
    fields = {**results, **extra}
    print("  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in fields.items()
    ))


def print_zo_step(step: int, total: int, metrics: dict, show_lr: bool = True):
    loss     = metrics["loss"]
    pg       = metrics["proj_grad"]
    step_str = f"{DIM}{step + 1:>6}/{total}{RESET}"
    loss_str = f"loss {GREEN}{loss:.4f}{RESET}"
    pg_str   = f"proj_grad {YEL}{pg:+.3e}{RESET}"
    parts    = [step_str, loss_str, pg_str]
    clip_frac = metrics.get("clip_frac")
    if clip_frac is not None:
        # Red once the clamp is firing on most pairs (magnitude info being discarded).
        clip_col = RED if clip_frac > 0.5 else YEL
        parts.append(f"clip {clip_col}{clip_frac:.0%}{RESET}")
    # Objective metrics (ce, z_loss, acc, … or name-prefixed for a mixture).
    for k, v in sorted(metrics.items()):
        if k in _ZO_SHOWN or k in _ZO_HIDDEN or v is None:
            continue
        parts.append(f"{k} {MAG}{v:.4f}{RESET}")
    if show_lr:                                  # suppressed when the schedule is constant
        parts.append(f"lr {DIM}{metrics['lr']:.2e}{RESET}")
    if "step_time" in metrics:
        parts.append(_fmt_step_time(metrics["step_time"]))
    print("  " + f"  {DIM}|{RESET}  ".join(parts))


def print_fo_step(step: int, total: int, loss: float, grad_norm: float, lr: float,
                  extra: dict[str, float] | None = None, show_lr: bool = True,
                  step_time: float | None = None):
    step_str = f"{DIM}{step + 1:>6}/{total}{RESET}"
    loss_str = f"loss {GREEN}{loss:.4f}{RESET}"
    gn_str   = f"gnorm {BLUE}{grad_norm:.3f}{RESET}"
    parts    = [step_str, loss_str, gn_str]
    if extra is not None:
        for k, v in sorted(extra.items()):
            parts.append(f"{k} {MAG}{v:.4f}{RESET}")
    if show_lr:                                  # suppressed when the schedule is constant
        parts.append(f"lr {DIM}{lr:.2e}{RESET}")
    if step_time is not None:
        parts.append(_fmt_step_time(step_time))
    print("  " + f"  {DIM}|{RESET}  ".join(parts))
