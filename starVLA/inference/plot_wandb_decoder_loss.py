"""
Plot all decoder_loss_* metrics from a WandB run on a single figure for comparison.

Two modes:
  1. --log-file: Parse output.log (offline, no wandb login needed)
  2. --run-id:   Fetch directly from WandB API (requires `wandb login`)

Usage:
    # From local output.log (offline):
    python starVLA/inference/plot_wandb_decoder_loss.py \
        --log-file results/Checkpoints/<run>/wandb/wandb/latest-run/files/output.log

    # From WandB API (online):
    python starVLA/inference/plot_wandb_decoder_loss.py \
        --run-id ezt41crw \
        --wandb-project latent_vla_51 \
        --wandb-entity leledaidai-harbin-institute-of-technology
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# ANSI escape code stripping
# ======================================================================

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


# ======================================================================
# Parsing output.log
# ======================================================================

def parse_output_log(log_path: Path) -> dict[str, dict[int, float]]:
    """Parse output.log and extract step → value for every decoder_loss_* metric.

    Returns
    -------
    dict  key (e.g. "decoder_loss_task") → {step: value}
    """
    raw = log_path.read_text(encoding="utf-8")
    cleaned = strip_ansi(raw)

    # Split into blocks by ">> Step" markers
    blocks = re.split(r">> Step\s+(\d+)", cleaned)
    # blocks[0] = preamble, blocks[1] = step1, blocks[2] = metrics1, blocks[3] = step2, ...

    result: dict[str, dict[int, float]] = {}

    for i in range(1, len(blocks), 2):
        step = int(blocks[i])
        metrics_text = blocks[i + 1]

        # Find the metrics dict boundary: from "metrics={" to "}"
        # But the dict may have nested braces/URLs, so find the dict that starts after 'metrics='
        match = re.search(r"metrics=(\{.*?\})\s*$", metrics_text, re.DOTALL)
        if not match:
            # Fallback: find all key-value pairs in this block
            pass

        # Extract all decoder_loss_* values from the entire block text
        for m in re.finditer(r"'?(decoder_loss_\w*)'?\s*:\s*([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)", metrics_text):
            key = m.group(1)
            val = float(m.group(2))
            result.setdefault(key, {})[step] = val

        # Also capture the main 'decoder_loss' (not _suffix)
        for m in re.finditer(r"'?(decoder_loss)'?\s*:\s*([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)", metrics_text):
            key = m.group(1)
            val = float(m.group(2))
            result.setdefault(key, {})[step] = val

    return result


# ======================================================================
# WandB API fetching
# ======================================================================

def fetch_from_wandb(run_id: str, project: str, entity: str) -> dict[str, dict[int, float]]:
    """Fetch all decoder_loss_* metrics from a WandB run via the API."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(samples=None)  # all steps

    result: dict[str, dict[int, float]] = {}
    decoder_keys = [k for k in history.columns if k.startswith("decoder_loss")]

    for key in decoder_keys:
        series = history[["_step", key]].dropna()
        result[key] = dict(zip(series["_step"].astype(int), series[key].astype(float)))

    return result


# ======================================================================
# Plotting
# ======================================================================

def plot_decoder_losses(
    data: dict[str, dict[int, float]],
    output_path: Path,
    title: str = "Decoder Loss Comparison",
    smooth_window: int = 0,
    log_y: bool = False,
) -> None:
    """Plot all decoder_loss curves on a single figure.

    Parameters
    ----------
    data : dict  key → {step: value}
    output_path : Path  where to save the PNG
    title : str  plot title
    smooth_window : int  if > 1, apply running-mean smoothing
    log_y : bool  use log scale on y-axis
    """
    n_curves = len(data)
    if n_curves == 0:
        print("[!] No decoder_loss_* metrics found.")
        return

    # Sort keys: put 'decoder_loss' (overall) first, then alphabetical
    sorted_keys = sorted(data.keys(), key=lambda k: ("" if k == "decoder_loss" else k))

    # Color palette — distinguishable for up to ~10 curves
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(sorted_keys))]

    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, key in enumerate(sorted_keys):
        steps_vals = sorted(data[key].items())
        if not steps_vals:
            continue
        steps, vals = zip(*steps_vals)
        steps = np.array(steps)
        vals = np.array(vals)

        # Optional smoothing
        if smooth_window > 1 and len(vals) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            vals_smooth = np.convolve(vals, kernel, mode="valid")
            steps_smooth = steps[smooth_window // 2 : smooth_window // 2 + len(vals_smooth)]
            label = f"{key} (smooth={smooth_window})"
            ax.plot(steps_smooth, vals_smooth, color=colors[idx], linewidth=1.5, label=label, alpha=0.9)
            # Also plot faint raw
            ax.plot(steps, vals, color=colors[idx], linewidth=0.3, alpha=0.25)
        else:
            ax.plot(steps, vals, color=colors[idx], linewidth=1.2, label=key, alpha=0.85)

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=2 if n_curves > 5 else 1)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Plot saved to {output_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot all decoder_loss_* metrics from WandB on one figure"
    )
    # Mode 1: parse local output.log
    p.add_argument("--log-file", type=Path, default=None,
                   help="Path to wandb output.log for offline parsing")
    # Mode 2: fetch from wandb API
    p.add_argument("--run-id", type=str, default=None,
                   help="WandB run ID (e.g. ezt41crw)")
    p.add_argument("--wandb-project", type=str, default="latent_vla_51")
    p.add_argument("--wandb-entity", type=str,
                   default="leledaidai-harbin-institute-of-technology")
    # Plot options
    p.add_argument("--output", "-o", type=Path,
                   default=Path("results/decoder_loss_comparison.png"),
                   help="Output PNG path (default: results/decoder_loss_comparison.png)")
    p.add_argument("--title", type=str, default="Decoder Loss Comparison")
    p.add_argument("--smooth", type=int, default=0,
                   help="Running-mean window size (0 = no smoothing)")
    p.add_argument("--log-y", action="store_true",
                   help="Use log scale on y-axis")
    args = p.parse_args()

    if args.log_file is not None:
        print(f"[*] Parsing log file: {args.log_file}")
        data = parse_output_log(args.log_file)
    elif args.run_id is not None:
        print(f"[*] Fetching run {args.run_id} from WandB API...")
        data = fetch_from_wandb(args.run_id, args.wandb_project, args.wandb_entity)
    else:
        p.error("Either --log-file or --run-id must be specified")

    if not data:
        print("[!] No decoder_loss metrics found.")
        return

    print(f"[*] Found {len(data)} decoder_loss metrics:")
    for k in sorted(data.keys()):
        n_pts = len(data[k])
        vals = list(data[k].values())
        print(f"      {k}: {n_pts} points, range [{min(vals):.4f}, {max(vals):.4f}]")

    plot_decoder_losses(
        data, args.output, title=args.title,
        smooth_window=args.smooth, log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
