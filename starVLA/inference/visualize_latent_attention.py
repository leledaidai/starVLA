"""
Visualize DiT cross-attention weights from action tokens to latent think tokens.

Usage:
    python starVLA/inference/visualize_latent_attention.py \\
        --checkpoint-path results/Checkpoints/<run_id> \\
        --step 40000 \\
        --num-samples 3 \\
        --output-file results/attention.txt

The script loads a QwenGR00TImplicitCoT checkpoint, runs predict_action on
training-set samples, and prints the cross-attention weights from DiT action
tokens (query) to VLM latent think token positions (key).  Output is organised
around **comparing different latent think tokens**.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Optional, TextIO

import matplotlib
matplotlib.use("Agg")  # headless – no X11 needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from starVLA.inference.test_latent_cot_decode import (
    load_dataset_examples,
    load_latent_cot_debug_checkpoint,
    save_extracted_samples,
    set_debug_seed,
    _resolve_checkpoint,
)
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import read_mode_config
from omegaconf import OmegaConf


# ======================================================================
# Tee helper – duplicate stdout to a file
# ======================================================================

@contextmanager
def tee_output(filepath: Path):
    """Context manager that mirrors ``sys.stdout`` writes to *filepath*."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fh = filepath.open("w", encoding="utf-8")

    class _Tee(TextIO):
        def write(self, s):
            sys.__stdout__.write(s)
            fh.write(s)

        def flush(self):
            sys.__stdout__.flush()
            fh.flush()

        def __getattr__(self, name):
            return getattr(sys.__stdout__, name)

    _orig = sys.stdout
    try:
        sys.stdout = _Tee()
        yield
    finally:
        sys.stdout = _orig
        fh.close()
        print(f"[*] Output saved to {filepath}")


# ======================================================================
# Custom diffusers Attention processor – capture cross-attention weights
# ======================================================================

class CaptureCrossAttnProcessor:
    """Wrap a diffusers Attention processor to capture cross-attention weights.

    When ``encoder_hidden_states is not None`` (cross-attention), attention
    weights [B, heads, q_seq, kv_seq] are computed explicitly and stored in
    ``self.attn_weights``.  Actual forward computation is delegated to the
    *original* processor so model outputs are unchanged.
    """

    def __init__(self, original_processor):
        self.original = original_processor
        self.attn_weights: Optional[torch.Tensor] = None  # [B, heads, q_seq, kv_seq]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        if encoder_hidden_states is not None:
            self.attn_weights = _compute_cross_attn_weights(
                attn, hidden_states, encoder_hidden_states, attention_mask,
            )
        return self.original(attn, hidden_states, encoder_hidden_states,
                             attention_mask, **kwargs)


def _compute_cross_attn_weights(attn, hidden_states, encoder_hidden_states,
                                attention_mask) -> torch.Tensor:
    """Manual Q @ K^T * scale → softmax.  Returns [B, heads, q_seq, kv_seq] on CPU."""
    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)

    B = query.shape[0]
    q = attn.head_to_batch_dim(query)          # [B*heads, q_seq, dim_head]
    k = attn.head_to_batch_dim(key)            # [B*heads, kv_seq, dim_head]

    scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale  # [B*heads, q_seq, kv_seq]

    if attention_mask is not None:
        scores_4d = scores.view(B, attn.heads, scores.shape[1], scores.shape[2])
        if attention_mask.dim() == 2:
            mask_add = torch.where(
                attention_mask.bool(),
                torch.tensor(0.0, device=scores.device, dtype=scores.dtype),
                torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype),
            )
            mask_add = mask_add[:, None, None, :]       # [B, 1, 1, kv_seq]
            scores_4d = scores_4d + mask_add
        elif attention_mask.dim() == 4:
            scores_4d = scores_4d + attention_mask.to(dtype=scores.dtype)
        scores = scores_4d.view(B * attn.heads, scores_4d.shape[2], scores_4d.shape[3])

    attn_weights = scores.softmax(dim=-1)
    attn_weights = attn_weights.view(B, attn.heads, attn_weights.shape[1], attn_weights.shape[2])
    return attn_weights.detach().cpu()


# ======================================================================
# Hook installation
# ======================================================================

def install_attention_hooks(model) -> list[tuple[int, CaptureCrossAttnProcessor]]:
    """Replace cross-attention processors in the DiT with capture wrappers."""
    transformer_blocks = model.action_model.model.transformer_blocks
    capture_procs: list[tuple[int, CaptureCrossAttnProcessor]] = []

    for idx, block in enumerate(transformer_blocks):
        if idx % 2 == 1:          # self-attention block → skip
            continue
        cap = CaptureCrossAttnProcessor(block.attn1.processor)
        block.attn1.set_processor(cap)
        capture_procs.append((idx, cap))

    print(f"[*] Installed cross-attn capture on {len(capture_procs)} DiT layers "
          f"(indices: {[p[0] for p in capture_procs]})")
    return capture_procs


def install_thinking_position_hook(model):
    """Patch ``_forward_latent`` to capture ``thinking_positions``."""
    _orig = model._forward_latent

    def _patched(_self, *, thinking_positions, **kwargs):
        _self._captured_thinking_positions = thinking_positions
        return _orig(thinking_positions=thinking_positions, **kwargs)

    model._forward_latent = MethodType(_patched, model)


# ======================================================================
# Attention extraction
# ======================================================================

def extract_think_attention(
    capture_procs: list[tuple[int, CaptureCrossAttnProcessor]],
    thinking_positions: torch.Tensor,  # [B, n_think]
) -> dict[int, dict[str, np.ndarray]]:
    """Extract per-layer attention from action tokens → thinking tokens.

    Returns
    -------
    dict  layer_idx → {
        "per_head":    [heads, a_seq, n_think]  raw per-head weights,
        "mean_head":   [a_seq, n_think]          averaged over heads,
    }
    """
    result: dict[int, dict[str, np.ndarray]] = {}
    think_pos = thinking_positions[0].cpu().tolist()

    for layer_idx, cap in capture_procs:
        w = cap.attn_weights
        if w is None:
            print(f"  [WARN] Layer {layer_idx}: no attention weights captured")
            continue
        w0 = w[0]                                    # [heads, sa_seq, vl_seq]
        think_w = w0[:, :, think_pos].numpy()        # [heads, sa_seq, n_think]
        result[layer_idx] = {
            "per_head": think_w,
            "mean_head": think_w.mean(axis=0),       # [sa_seq, n_think]
        }
    return result


# ======================================================================
# Printing — think-token-centric
# ======================================================================

def _bar(val: float, max_width: int = 40, scale: float = 1.0) -> str:
    w = max(1, int(val * scale * max_width))
    return "█" * w


def _build_action_labels(info: dict[str, int]) -> list[str]:
    """Return list of action token labels like ['state_0', ..., 'future_0', ..., 'action_0', ...]."""
    n_state = info.get("n_state", 0)
    n_future = info.get("n_future", 0)
    n_action = info.get("n_action", 0)
    labels = []
    for i in range(n_state):
        labels.append(f"state_{i}")
    for i in range(n_future):
        labels.append(f"future_{i}")
    for i in range(n_action):
        labels.append(f"action_{i}")
    return labels


def print_think_token_comparison(
    sample_idx: int,
    example: dict,
    capture_procs: list[tuple[int, CaptureCrossAttnProcessor]],
    thinking_positions: torch.Tensor,
    field_names: list[str],
    action_seq_info: dict[str, int],
) -> None:
    """Print cross-attention results focused on comparing different think tokens."""

    attn_data = extract_think_attention(capture_procs, thinking_positions)

    think_pos = thinking_positions[0].cpu().tolist()
    n_think = len(think_pos)
    field_labels = field_names[:n_think] if len(field_names) >= n_think else [f"think_{i}" for i in range(n_think)]

    n_state = action_seq_info.get("n_state", 0)
    n_future = action_seq_info.get("n_future", 0)
    n_action = action_seq_info.get("n_action", 0)
    sa_seq = n_state + n_future + n_action
    sorted_layers = sorted(attn_data.keys())

    # --- Build action token labels ---
    action_labels: list[str] = []
    for i in range(n_state):   action_labels.append(f"state_{i}")
    for i in range(n_future):  action_labels.append(f"future_{i}")
    for i in range(n_action):  action_labels.append(f"action_{i}")

    # Aggregate:  [n_layers, sa_seq, n_think]  (head-mean)
    all_layers = np.stack([attn_data[l]["mean_head"] for l in sorted_layers], axis=0)

    # ---- 样本概览 ----
    print(f"\n{'=' * 100}")
    print(f"样本 {sample_idx}: {example.get('lang', 'N/A')[:120]}")
    print(f"  思考 Token 数量: {n_think}  字段名: {field_labels}")
    print(f"  思考 Token 在 VLM 序列中的位置: {think_pos}")
    print(f"  DiT 输入序列: {n_state} 个状态 + {n_future} 个 future + {n_action} 个动作 = {sa_seq} tokens")
    print(f"  DiT 交叉注意力层: {sorted_layers}")
    # 打印 CoT 文本（GT vs 模型解码）
    decoded = example.get("_decoded_cot", None)
    gt_cot = example.get("gt_cot", None)
    visible_cot = example.get("visible_cot_text", "")
    if decoded or gt_cot or visible_cot:
        print(f"  CoT 对比 (GT=Ground Truth, PD=模型从 latent 解码):")
        if gt_cot:
            # 有逐字段 GT
            for slot_name in field_labels:
                gt_text = gt_cot.get(slot_name, "")
                pd_text = decoded.get(slot_name, "") if decoded else ""
                if gt_text or pd_text:
                    if gt_text == pd_text:
                        print(f"    [{slot_name}] GT=PD: {gt_text}")
                    else:
                        print(f"    [{slot_name}] GT: {gt_text}")
                        print(f"    [{slot_name}] PD: {pd_text}")
        elif visible_cot:
            # 仅有完整 visible CoT 文本作为 GT
            print(f"    GT (完整): {visible_cot}")
            if decoded:
                pd_full = " | ".join(f"{k}={v}" for k, v in decoded.items() if v)
                print(f"    PD (解码): {pd_full}")
        elif decoded:
            for slot_name in field_labels:
                text = decoded.get(slot_name, "")
                if text:
                    print(f"    [{slot_name}] PD: {text}")

    # ==================================================================
    # Section 1: 思考 Token 注意力占比对比
    # ==================================================================
    print(f"\n{'─' * 100}")
    print("  1. 思考 Token 注意力占比对比（对所有层、所有头、所有动作 token 取均值）")
    print(f"{'─' * 100}")
    print(f"{'思考 Token':>18s} │ {'总注意力':>10s} │ {'占全体%':>8s} │ "
          f"{'状态%':>7s} {'未来%':>7s} {'动作%':>7s} │ "
          f"最关注该思考 Token 的 Top-3 动作 Token")
    print(f"{'─' * 18}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 7}─{'─' * 7}─{'─' * 7}─┼─{'─' * 50}")

    # Per-think-token: mean attention over [layers, action_tokens]
    think_total = all_layers.mean(axis=(0, 1))   # [n_think]
    total_all = think_total.sum()

    # Per action-token-type breakdown
    if n_state > 0:
        state_slice = slice(0, n_state)
    else:
        state_slice = slice(0, 0)
    future_slice = slice(n_state, n_state + n_future) if n_future > 0 else slice(0, 0)
    action_slice = slice(n_state + n_future, sa_seq) if n_action > 0 else slice(0, 0)

    # token_mean: [sa_seq, n_think]  avg over layers
    token_mean = all_layers.mean(axis=0)

    for i, fl in enumerate(field_labels):
        total = think_total[i]
        pct = 100.0 * total / total_all if total_all > 0 else 0.0

        # Action-type breakdown for this think token
        state_attn  = token_mean[state_slice, i].sum() if n_state > 0 else 0.0
        future_attn = token_mean[future_slice, i].sum() if n_future > 0 else 0.0
        action_attn = token_mean[action_slice, i].sum() if n_action > 0 else 0.0
        s_pct = 100.0 * state_attn / total if total > 0 else 0.0
        f_pct = 100.0 * future_attn / total if total > 0 else 0.0
        a_pct = 100.0 * action_attn / total if total > 0 else 0.0

        # Top-3 action tokens
        top_indices = np.argsort(token_mean[:, i])[::-1][:3]
        top_strs = []
        for ti in top_indices:
            lbl = action_labels[ti] if ti < len(action_labels) else f"tok_{ti}"
            top_strs.append(f"{lbl}({token_mean[ti, i]:.4f})")
        top_str = ", ".join(top_strs)

        print(f"{fl:>18s} │ {total:10.6f} │ {pct:7.1f}% │ "
              f"{s_pct:6.1f}% {f_pct:6.1f}% {a_pct:6.1f}% │ {top_str}")

    print(f"{'─' * 18}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 7}─{'─' * 7}─{'─' * 7}─┼─{'─' * 50}")
    print(f"{'合计（全部思考 Token）':>18s} │ {total_all:10.6f} │ {'100.0%':>8s} │")

    # ==================================================================
    # Section 2: 逐层注意力分布
    # ==================================================================
    print(f"\n{'─' * 100}")
    print("  2. 逐层注意力分布（各层各思考 Token 获得的注意力）")
    print(f"{'─' * 100}")

    # layer_think: [n_layers, n_think]
    layer_think = all_layers.mean(axis=1)   # avg over action tokens → [n_layers, n_think]
    max_val = layer_think.max()

    header = f"{'层':>6s} │"
    for fl in field_labels:
        header += f" {fl:>14s}"
    header += f" │ {'合计':>10s}"
    print(header)
    print(f"{'─' * 6}─┼─{'─' * (15 * n_think + 2)}─┼─{'─' * 10}")

    for li, layer_idx in enumerate(sorted_layers):
        vals = layer_think[li]                      # [n_think]
        row = f"{layer_idx:>4d}  │"
        for vi, v in enumerate(vals):
            bar = _bar(v, max_width=12, scale=1.0 / (max_val + 1e-8))
            row += f" {bar:>12s}{v:.4f}"
        row += f" │ {vals.sum():10.6f}"
        print(row)

    # ==================================================================
    # Section 3: 各思考 Token 在动作序列上的注意力分布
    # ==================================================================
    print(f"\n{'─' * 100}")
    print("  3. 各思考 Token 在动作序列上的注意力分布")
    print(f"     （每个思考 Token 的注意力在 56 个动作 token 位置上如何分布？█=高于均值, ▁=低于均值）")
    print(f"{'─' * 100}")

    # For each think token, show a compact "heatmap" of attention over action positions,
    # grouped by type (state / future / action)
    for i, fl in enumerate(field_labels):
        col = token_mean[:, i]                     # [sa_seq]  avg over layers
        # Summarise by segment
        state_vals  = col[state_slice] if n_state > 0 else np.array([])
        future_vals = col[future_slice] if n_future > 0 else np.array([])
        action_vals = col[action_slice] if n_action > 0 else np.array([])

        def _seg_str(name, vals, n):
            if len(vals) == 0:
                return ""
            mean_v = vals.mean()
            max_v = vals.max()
            argmax = int(np.argmax(vals))
            # Compact bar: each position one pixel
            bar = "".join("█" if v > mean_v else "▁" for v in vals)
            return (f"{name}(数量={n}) 均值={mean_v:.4f} 最大=t{argmax}({max_v:.4f}) [{bar}]")

        parts = []
        if n_state > 0:
            parts.append(_seg_str("state ", state_vals, n_state))
        if n_future > 0:
            parts.append(_seg_str("future", future_vals, n_future))
        if n_action > 0:
            parts.append(_seg_str("action", action_vals, n_action))

        print(f"  {fl:>16s}: {' | '.join(parts)}")

    # ==================================================================
    # Section 4: 逐层完整注意力矩阵
    # ==================================================================
    print(f"\n{'─' * 100}")
    print("  4. 逐层完整注意力矩阵（动作 Token → 思考 Token，对注意力头取均值，不省略任何行）")
    print(f"{'─' * 100}")

    # Pick representative layers (first / mid / last) — show ALL action token rows
    disp_layers = sorted_layers
    if len(disp_layers) > 4:
        mid = len(disp_layers) // 2
        disp_layers = [sorted_layers[0], sorted_layers[mid], sorted_layers[-1]]

    field_width = 12

    for layer_idx in disp_layers:
        mh = attn_data[layer_idx]["mean_head"]   # [sa_seq, n_think]
        sa_seq = mh.shape[0]
        print(f"\n  ── 第 {layer_idx} 层（{sa_seq} 个动作 Token × {n_think} 个思考 Token）──")

        # Header
        header = f"  {'动作 Token':>12s} │"
        for fl in field_labels:
            header += f" {fl:>{field_width}s}"
        header += f" │ {'合计':>{field_width}s}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        # 打印每行动作 token
        for pos in range(sa_seq):
            label = action_labels[pos] if pos < len(action_labels) else f"tok_{pos}"
            row_vals = mh[pos]                                     # [n_think]
            parts = []
            for v in row_vals:
                if abs(v) < 1e-5:
                    parts.append(f"{'‧':>{field_width}s}")
                else:
                    parts.append(f"{v:>{field_width}.{field_width - 3}f}")
            row_str = " ".join(parts)
            sum_val = row_vals.sum()
            sum_str = f"{sum_val:>{field_width}.{field_width - 3}f}" if sum_val >= 1e-5 else f"{'‧':>{field_width}s}"
            print(f"  {label:>12s} │{row_str} │{sum_str}")

    # Also print the all-layer average matrix
    print(f"\n  ── 全层均值（{sa_seq} 个动作 Token × {n_think} 个思考 Token，{len(sorted_layers)} 层平均）──")
    avg_mh = all_layers.mean(axis=0)   # [sa_seq, n_think]
    header = f"  {'动作 Token':>12s} │"
    for fl in field_labels:
        header += f" {fl:>{field_width}s}"
    header += f" │ {'合计':>{field_width}s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for pos in range(sa_seq):
        label = action_labels[pos] if pos < len(action_labels) else f"tok_{pos}"
        row_vals = avg_mh[pos]
        parts = []
        for v in row_vals:
            if abs(v) < 1e-5:
                parts.append(f"{'‧':>{field_width}s}")
            else:
                parts.append(f"{v:>{field_width}.{field_width - 3}f}")
        row_str = " ".join(parts)
        sum_val = row_vals.sum()
        sum_str = f"{sum_val:>{field_width}.{field_width - 3}f}" if sum_val >= 1e-5 else f"{'‧':>{field_width}s}"
        print(f"  {label:>12s} │{row_str} │{sum_str}")


# ======================================================================
# Action sequence info detection
# ======================================================================

# ======================================================================
# Heatmap image generation
# ======================================================================

def save_heatmaps(
    attn_data: dict[int, dict[str, np.ndarray]],
    action_labels: list[str],
    field_labels: list[str],
    sample_idx: int,
    output_dir: Path,
) -> None:
    """Save per-layer and all-layer-mean attention heatmaps as PNG images.

    Each heatmap has action tokens on the Y axis and think tokens on the X axis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_layers = sorted(attn_data.keys())
    all_mean_heads = np.stack([attn_data[l]["mean_head"] for l in sorted_layers], axis=0)
    # all_mean_heads: [n_layers, sa_seq, n_think]

    sa_seq = all_mean_heads.shape[1]
    n_think = all_mean_heads.shape[2]

    # --- Shared figure settings ---
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
    })

    # --- Per-layer heatmaps (ALL cross-attention layers) ---
    for layer_idx in sorted_layers:
        data = attn_data[layer_idx]["mean_head"]   # [sa_seq, n_think]
        _draw_heatmap(
            data=data,
            row_labels=action_labels,
            col_labels=field_labels,
            title=f"Sample {sample_idx} — DiT Cross-Attn Layer {layer_idx}\n"
                  f"Action tokens (Y) → Think tokens (X), head-mean",
            save_path=output_dir / f"sample_{sample_idx:02d}_layer_{layer_idx:02d}_heatmap.png",
        )

    # --- All-layer mean heatmap ---
    avg_data = all_mean_heads.mean(axis=0)   # [sa_seq, n_think]
    _draw_heatmap(
        data=avg_data,
        row_labels=action_labels,
        col_labels=field_labels,
        title=f"Sample {sample_idx} — DiT Cross-Attn ALL-LAYER MEAN\n"
              f"Action tokens (Y) → Think tokens (X), avg over {len(sorted_layers)} layers",
        save_path=output_dir / f"sample_{sample_idx:02d}_all_layers_mean_heatmap.png",
    )

    # --- Per-think-token attention share bar chart ---
    think_total = avg_data.sum(axis=0)       # [n_think]  total attn each think token receives
    think_pct = 100.0 * think_total / think_total.sum()

    fig, ax = plt.subplots(figsize=(max(6, n_think * 1.2), 4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_think))
    bars = ax.bar(range(n_think), think_total, color=colors, edgecolor="white", linewidth=0.5)
    for i, (bar, pct) in enumerate(zip(bars, think_pct)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + think_total.max() * 0.02,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(n_think))
    ax.set_xticklabels(field_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Total attention received (sum over action tokens)", fontsize=9)
    ax.set_title(f"Sample {sample_idx} — Think Token Attention Share\n"
                 f"(avg over {len(sorted_layers)} layers & all heads)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{sample_idx:02d}_think_token_share.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Heatmaps saved to {output_dir}")


def _draw_heatmap(
    data: np.ndarray,        # [n_rows, n_cols]
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    save_path: Path,
) -> None:
    """Draw a single heatmap image with clear row/column labels."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = data.shape

    # Determine figure size: enough pixels per cell for readability
    fig_width = max(6, n_cols * 0.9)
    fig_height = max(5, n_rows * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Attention weight", fontsize=8)

    # --- Axis ticks ---
    # Y axis: show all action token labels, but skip some if too dense
    ax.set_yticks(range(n_rows))
    if n_rows <= 40:
        ax.set_yticklabels(row_labels, fontsize=6)
    else:
        # Show every Nth label to avoid crowding
        step = max(1, n_rows // 30)
        shown = [row_labels[i] if i % step == 0 else "" for i in range(n_rows)]
        ax.set_yticklabels(shown, fontsize=5)

    # X axis: think token labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)

    # --- Grid lines between cells ---
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- Annotate cells if matrix is small enough ---
    if n_rows <= 30 and n_cols <= 12:
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                text_color = "white" if v > data.max() * 0.6 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=5, color=text_color)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Action tokens", fontsize=9)
    ax.set_xlabel("Think tokens", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def detect_action_seq_info(model) -> dict[str, int]:
    am = model.action_model
    config = getattr(am, "config", None)
    n_action = getattr(config, "action_horizon", 16) if config is not None else 16
    future_weight = getattr(am, "future_tokens", None)
    n_future = future_weight.weight.shape[0] if future_weight is not None else 0
    n_state = 1   # default; updated after first run
    return {"n_state": n_state, "n_future": n_future, "n_action": n_action}


def update_action_seq_info_from_weights(capture_procs, info: dict[str, int]) -> dict[str, int]:
    for _, cap in capture_procs:
        w = cap.attn_weights
        if w is not None:
            sa_seq = w.shape[2]
            info["sa_seq"] = sa_seq
            info["n_state"] = max(0, sa_seq - info.get("n_future", 0) - info.get("n_action", 16))
            break
    return info


# ======================================================================
# Main
# ======================================================================

# ======================================================================
# Trajectory loading
# ======================================================================

def load_trajectory_examples(
    checkpoint_path: Path,
    episode_idx: int,
    max_steps: int,
    attn_implementation: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Load all frames from a specific episode/trajectory.

    Returns a list of example dicts (ordered by frame_index) for the episode.
    """
    model_config, _ = read_mode_config(str(checkpoint_path))
    cfg = OmegaConf.create(model_config)
    cfg.datasets.vla_data.per_device_batch_size = 1
    if attn_implementation:
        cfg.framework.qwenvl.attn_implementation = attn_implementation

    set_debug_seed(seed)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, full_cfg=cfg, mode="train", seed=seed)

    # 如果是 MixtureDataset，取第一个底层 dataset
    if hasattr(dataset, "datasets"):
        inner = dataset.datasets[0]
    else:
        inner = dataset

    if not hasattr(inner, "all_steps"):
        raise RuntimeError(
            f"Dataset type {type(inner).__name__} does not have 'all_steps' attribute. "
            "Cannot iterate trajectory frames."
        )

    # Group flat indices by trajectory_id
    ep_indices: list[int] = []
    for flat_idx, (traj_id, base_idx) in enumerate(inner.all_steps):
        if int(traj_id) == episode_idx:
            ep_indices.append(flat_idx)
            if max_steps > 0 and len(ep_indices) >= max_steps:
                break

    if not ep_indices:
        allowed = getattr(inner, "_allowed_episode_ids", None)
        if allowed:
            allowed_sample = sorted(allowed)[:20]
            raise ValueError(
                f"Episode {episode_idx} not found in dataset. "
                f"CoT-annotated episodes (first 20): {allowed_sample}\n"
                f"Tip: use --trajectory-episode with one of these IDs."
            )
        available = sorted(set(int(tid) for tid, _ in inner.all_steps))
        raise ValueError(
            f"Episode {episode_idx} not found. Available episodes (first 20): {available[:20]}"
        )

    # 检查第一个样本的 GT 是否为空
    test_sample = inner[ep_indices[0]]
    test_cot = test_sample.get("cot_fields_raw", {})
    has_gt = any(str(v).strip() for v in test_cot.values())
    if not has_gt:
        allowed = getattr(inner, "_allowed_episode_ids", None)
        if allowed:
            allowed_sample = sorted(allowed)[:20]
        else:
            allowed_sample = "N/A"
        print(f"  ⚠ Episode {episode_idx} 的 CoT GT 全部为空！该 episode 可能缺少 CoT 标注。")
        print(f"  ⚠ 有 CoT 标注的 episode (前20): {allowed_sample}")
        print(f"  ⚠ 将仅显示模型解码的 CoT 文本。")

    print(f"[*] 轨迹 Episode {episode_idx}: 共 {len(ep_indices)} 步")

    examples: list[dict[str, Any]] = []
    for flat_idx in ep_indices:
        sample = inner[flat_idx]
        action_val = sample["action"]
        example: dict[str, Any] = {
            "image": sample["image"],
            "lang": sample["lang"],
            "action": action_val.detach().to(dtype=torch.float32).cpu().numpy()
            if torch.is_tensor(action_val)
            else np.asarray(action_val, dtype=np.float32),
        }
        for meta_key in ("episode_index", "frame_index", "task_index"):
            if meta_key in sample:
                val = sample[meta_key]
                example[meta_key] = int(val.detach().cpu().item()) if torch.is_tensor(val) else int(val)
        if "state" in sample:
            s = sample["state"]
            example["state"] = s.detach().to(dtype=torch.float32).cpu().numpy() if torch.is_tensor(s) else np.asarray(s, dtype=np.float32)
        # 保存 ground truth CoT 文本
        cot_raw = sample.get("cot_fields_raw", None)
        if cot_raw:
            example["gt_cot"] = dict(cot_raw)
        examples.append(example)

    return examples


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize DiT cross-attention from action tokens to latent think tokens"
    )
    p.add_argument("--checkpoint-path", type=Path, required=True,
                   help="Path to checkpoint directory (containing checkpoints/ subdir) or .pt file")
    p.add_argument("--step", type=int, default=None,
                   help="Specific training step to load (e.g. 20000). "
                        "Looks for checkpoints/steps_{step}_pytorch_model.pt under --checkpoint-path.")
    p.add_argument("--mode", choices=["train", "valid"], default="train")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cot-tokens", type=int, default=128)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--save-npy-dir", type=Path, default=None,
                   help="Save raw attention arrays as .npy files")
    p.add_argument("--output-file", type=Path, default=None,
                   help="Duplicate all printed output to this file")
    p.add_argument("--save-samples-dir", type=Path, default=None,
                   help="If set, save sample images and prompts to this directory")
    p.add_argument("--heatmap-dir", type=Path, default=None,
                   help="If set, save per-layer + mean attention heatmap images to this directory")
    p.add_argument("--trajectory-episode", type=int, default=None,
                   help="Process ALL frames of a specific episode instead of random samples. "
                        "Saves all-layers-mean heatmap for every step.")
    p.add_argument("--trajectory-max-steps", type=int, default=0,
                   help="Max steps in trajectory mode (0=unlimited). Default: 0")
    return p


def _resolve_checkpoint_with_step(checkpoint_path: Path, step: Optional[int]) -> Path:
    """Resolve checkpoint path, optionally selecting a specific step."""
    ckpt = Path(checkpoint_path)

    if step is not None:
        if ckpt.is_dir():
            specific = ckpt / "checkpoints" / f"steps_{step}_pytorch_model.pt"
            if not specific.exists():
                available = sorted(ckpt.glob("checkpoints/steps_*_pytorch_model.pt"))
                raise FileNotFoundError(
                    f"Checkpoint for step {step} not found at {specific}. "
                    f"Available steps: {[p.stem.replace('steps_', '').replace('_pytorch_model', '') for p in available]}"
                )
            return specific
        else:
            # ckpt is a .pt file — ignore --step
            print(f"[*] --step ignored (--checkpoint-path is a direct .pt file: {ckpt})")
            return ckpt

    return _resolve_checkpoint(ckpt)


def main() -> None:
    args = build_argparser().parse_args()
    ckpt_path = _resolve_checkpoint_with_step(args.checkpoint_path, args.step)
    attn_impl = args.attn_implementation or None
    set_debug_seed(args.seed)

    output_ctx = tee_output(args.output_file) if args.output_file else nullcontext()
    with output_ctx:
        print(f"[*] Checkpoint: {ckpt_path}")

        # --- Load data ---
        trajectory_mode = args.trajectory_episode is not None
        if trajectory_mode:
            examples = load_trajectory_examples(
                ckpt_path, episode_idx=args.trajectory_episode,
                max_steps=args.trajectory_max_steps,
                attn_implementation=attn_impl, seed=args.seed,
            )
        else:
            examples = load_dataset_examples(
                ckpt_path, mode=args.mode, num_samples=args.num_samples,
                num_workers=args.num_workers, attn_implementation=attn_impl,
                seed=args.seed,
            )

        # --- 保存样本图片和 prompt（非轨迹模式）---
        if args.save_samples_dir is not None and not trajectory_mode:
            save_extracted_samples(examples, args.save_samples_dir)

        # --- Load model ---
        model, norm_stats = load_latent_cot_debug_checkpoint(
            ckpt_path, attn_implementation=attn_impl, mmap=not args.no_mmap,
        )
        model = model.to(torch.device(args.device)).eval()

        field_names: list[str] = getattr(model, "field_names", [])
        action_seq_info = detect_action_seq_info(model)

        print(f"[*] field_names: {field_names}")
        print(f"[*] 样本数: {len(examples)}")
        print(f"[*] Device: {args.device}")
        if trajectory_mode:
            print(f"[*] 轨迹模式: Episode {args.trajectory_episode}, {len(examples)} 步")
            # 保存第一帧图片到 heatmap 目录
            if args.heatmap_dir is not None and len(examples) > 0:
                first_img = examples[0]["image"]
                if isinstance(first_img, list):
                    first_img = first_img[0]
                from PIL import Image
                img = first_img if isinstance(first_img, Image.Image) else Image.fromarray(np.asarray(first_img))
                img.convert("RGB").save(Path(args.heatmap_dir) / "1.jpg")
                print(f"[*] 第一帧图片已保存到 {args.heatmap_dir}/1.jpg")

        # --- Install hooks ---
        capture_procs = install_attention_hooks(model)
        install_thinking_position_hook(model)

        # --- Run inference ---
        all_mse: list[float] = []
        for idx, example in enumerate(examples):
            with torch.inference_mode():
                outputs = model.predict_action(
                    [example], decode_cot=True, max_cot_tokens=args.max_cot_tokens,
                )

            thinking_positions = getattr(model, "_captured_thinking_positions", None)
            if thinking_positions is None:
                print(f"  [WARN] No thinking positions captured for sample {idx}")
                continue

            if idx == 0:
                action_seq_info = update_action_seq_info_from_weights(capture_procs, action_seq_info)

            if trajectory_mode:
                # 轨迹模式: 紧凑输出 + 必定保存 all-layers-mean heatmap
                n_think = thinking_positions.shape[1]
                flabels = field_names[:n_think] if len(field_names) >= n_think else [f"think_{i}" for i in range(n_think)]
                attn_data = extract_think_attention(capture_procs, thinking_positions)
                sorted_layers = sorted(attn_data.keys())
                all_mean = np.stack([attn_data[l]["mean_head"] for l in sorted_layers], axis=0).mean(axis=0)
                think_total = all_mean.sum(axis=0)  # [n_think]
                top_think_idx = int(np.argmax(think_total))

                frame_info = example.get("frame_index", idx)
                lang = example.get("lang", "")
                print(f"  step {idx:04d} (frame {frame_info})  Task: {lang}")
                print(f"         top_think={flabels[top_think_idx]}({think_total[top_think_idx]:.4f}) "
                      f"total_attn={think_total.sum():.4f}")
                # 打印该步的 latent CoT 解码文本（与 GT 对比）
                decoded = outputs.get("decoded_cot_by_field", [{}])
                cot_dict = decoded[0] if isinstance(decoded, list) else decoded
                gt_cot = example.get("gt_cot", {})
                for slot_name in flabels:
                    dec_text = cot_dict.get(slot_name, "") if cot_dict else ""
                    gt_text = gt_cot.get(slot_name, "") if gt_cot else ""
                    if dec_text or gt_text:
                        print(f"         [{slot_name}] GT: {gt_text}")
                        if dec_text != gt_text:
                            print(f"         [{slot_name}] PD: {dec_text}")
                        # 如果 GT 和 PD 相同则只打印 GT 一行

                # Save all-layers-mean heatmap
                if args.heatmap_dir is not None:
                    action_labels = _build_action_labels(action_seq_info)
                    _draw_heatmap(
                        data=all_mean,
                        row_labels=action_labels,
                        col_labels=flabels,
                        title=f"Episode {args.trajectory_episode} Step {idx:04d} (frame {frame_info})\n"
                              f"DiT Cross-Attn ALL-LAYER MEAN — Action→Think",
                        save_path=args.heatmap_dir / f"ep{args.trajectory_episode:04d}_step{idx:04d}_all_layers_mean.png",
                    )
            else:
                # 注入解码的 CoT 文本到 example 中供打印
                decoded = outputs.get("decoded_cot_by_field", [{}])
                if decoded:
                    example["_decoded_cot"] = decoded[0] if isinstance(decoded, list) else decoded
                print_think_token_comparison(
                    sample_idx=idx,
                    example=example,
                    capture_procs=capture_procs,
                    thinking_positions=thinking_positions,
                    field_names=field_names,
                    action_seq_info=action_seq_info,
                )

            # --- 计算 Action Loss (MSE) ---
            pred_action = outputs["normalized_actions"][0]    # [action_horizon, action_dim]
            gt_action = example.get("action")                  # [T, action_dim]
            if gt_action is not None:
                gt_action = np.asarray(gt_action, dtype=np.float32)
                min_len = min(len(pred_action), len(gt_action))
                pred_aligned = pred_action[:min_len]
                gt_aligned = gt_action[:min_len]
                mse = ((pred_aligned - gt_aligned) ** 2).mean()
                per_dim_mse = ((pred_aligned - gt_aligned) ** 2).mean(axis=0)
                dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"][:per_dim_mse.shape[0]]
                all_mse.append(mse)
                if trajectory_mode:
                    print(f"         MSE={mse:.6f}")
                else:
                    print(f"\n  Action Loss (MSE): {mse:.6f}  (对齐长度={min_len})")
                    dim_str = " | ".join(f"{n}={v:.6f}" for n, v in zip(dim_names, per_dim_mse))
                    print(f"  逐维度 MSE: {dim_str}")
            else:
                if trajectory_mode:
                    print(f"         MSE=N/A (无 ground-truth)")
                else:
                    print(f"\n  [WARN] 样本 {idx} 无 ground-truth action，跳过 loss 计算")

            # --- Save heatmap images (non-trajectory mode: per-layer + all-layers-mean) ---
            if args.heatmap_dir is not None and not trajectory_mode:
                n_think = thinking_positions.shape[1]
                flabels = field_names[:n_think] if len(field_names) >= n_think else [f"think_{i}" for i in range(n_think)]
                attn_data = extract_think_attention(capture_procs, thinking_positions)
                save_heatmaps(
                    attn_data=attn_data,
                    action_labels=_build_action_labels(action_seq_info),
                    field_labels=flabels,
                    sample_idx=idx,
                    output_dir=args.heatmap_dir,
                )

            # --- Save .npy (non-trajectory mode only) ---
            if args.save_npy_dir is not None and not trajectory_mode:
                save_dir = Path(args.save_npy_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                for layer_idx, cap in capture_procs:
                    w = cap.attn_weights
                    if w is not None:
                        fpath = save_dir / f"sample_{idx:02d}_layer_{layer_idx:02d}_cross_attn.npy"
                        np.save(fpath, w.numpy())
                print(f"\n[*] Saved attention arrays to {save_dir}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 平均 Action Loss ---
        if all_mse:
            avg_mse = np.mean(all_mse)
            std_mse = np.std(all_mse)
            print(f"\n{'=' * 100}")
            print(f"平均 Action Loss (MSE): {avg_mse:.6f} ± {std_mse:.6f}  (共 {len(all_mse)} 个样本)")
            if trajectory_mode and len(all_mse) > 1:
                # 打印 MSE 走势
                print(f"MSE 走势: ", end="")
                for i, v in enumerate(all_mse):
                    bar = "█" if v > avg_mse else "▁"
                    print(bar, end="")
                print()
                print(f"  min={min(all_mse):.6f}  max={max(all_mse):.6f}")
            else:
                print(f"各样本 MSE: {', '.join(f'{v:.6f}' for v in all_mse)}")
            print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
