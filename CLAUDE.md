# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarVLA is a modular ("Lego-like") research codebase for Vision-Language-Action (VLA) models — multimodal models that output robot actions. It supports multiple VLA frameworks (FAST, OFT, PI, GR00T) and benchmark integrations (LIBERO, SimplerEnv, RoboCasa, Calvin, etc.).

**Branches:** `starVLA_dev` is the active development branch (may be unstable). `starVLA` is the stable branch with verified results.

## Build, Lint, Format

```bash
make check       # black --check + ruff check (read-only)
make autoformat  # black + ruff --fix-only (applies fixes in-place)
make clean       # remove .pyc and __pycache__
```

- Black line-length: **121**, target Python 3.10
- Ruff target version: py310, line-length 121
- Install package in editable mode: `pip install -e .`

## Running Tests / Smoke Tests

There is no traditional test suite. Each major module supports standalone smoke-testing:

```bash
# Smoke test a framework (runs a forward pass on fake data)
python starVLA/model/framework/VLM4A/QwenGR00T.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml

# Smoke test the dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
```

## Training

Training uses `accelerate` + DeepSpeed (ZeRO-2 or ZeRO-3). Launch pattern:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml path/to/config.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /path/to/data \
  --datasets.vla_data.data_mix bridge \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --run_root_dir ./results/Checkpoints \
  --run_id my_experiment
```

Three training entry points exist:
- `train_starvla.py` — VLA-only training
- `train_starvla_cotrain.py` — VLA + VLM co-training
- `train_starvlm.py` — VLM-only training

CLI args use dot-notation to override any YAML config key (e.g., `--trainer.max_train_steps 5000`).

## Architecture

### Top-level directory layout

```
starVLA/
  model/
    framework/        # Framework definitions (VLM4A/, WM4A/) — each is a complete model variant
      base_framework.py  # Abstract base + build_framework() factory
    modules/
      action_model/   # Action prediction heads (GR00T, MLP, DiT, FAST, etc.)
      vlm/            # VLM backbone interfaces (Qwen-VL, InternVL, etc.)
      projector/      # Feature projection (QFormer)
    tools.py          # Registry, FrameworkTools (normalization, trainable-module discovery)
  training/
    train_starvla.py      # Main VLA training loop
    train_starvla_cotrain.py  # Multi-dataloader VLA+VLM co-training
    trainer_utils/        # Metrics, LR groups, config tracking, freeze helpers
  dataloader/
    lerobot_datasets.py   # LeRobot-format VLA datasets
    vlm_datasets.py       # VLM co-training data (LLaVA-JSON format)
  config/
    training/         # YAML training configs
    deepseeds/        # DeepSpeed ZeRO-2 / ZeRO-3 configs
examples/             # Benchmark-specific pipelines (LIBERO, SimplerEnv, RoboCasa, Calvin, etc.)
deployment/           # Model server for real-robot inference
```

### Framework registry pattern

Each framework file (e.g., `QwenGR00T.py`) registers itself via a decorator:

```python
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("QwenGR00T")
class QwenGR00TModel(baseframework):
    ...
```

`build_framework(cfg)` in `base_framework.py` auto-imports all framework modules and dispatches to the class registered under `cfg.framework.name`. This is the single external entry point for model construction.

### baseframework contract

Every framework inherits from `baseframework(PreTrainedModel)` and must implement:

- `forward(examples: List[dict]) -> dict` — returns `{"action_loss": Tensor}` (training)
- `predict_action(examples: List[dict]) -> dict` — returns `{"normalized_actions": np.ndarray}` (inference)
- `compute_loss(tag, batch)` — dispatches `"vla"` → `forward()`, `"vlm"` → `forward_vlm()`

### Data contract

Dataloaders return model-agnostic dicts. A single sample includes:
- `image`: `list[PIL.Image]` or `np.ndarray`
- `lang`: `str`
- `action`: `np.ndarray` shape `[T, action_dim]`
- `state`: `Optional[np.ndarray]` shape `[..., state_dim]`

No model-specific preprocessing (tokenization, image encoding) happens in the dataloader.

### Framework variants

| Framework | Class | Action Head | Style |
|---|---|---|---|
| StarVLA-FAST | QwenFast | fast_ActionHeader | Autoregressive discrete tokens (π₀-fast) |
| StarVLA-OFT | QwenOFT | MLP_ActionHeader | Parallel continuous (OpenVLA-OFT) |
| StarVLA-PI | QwenPI | LayerwiseFM_ActionHeader | Flow-matching diffusion (π₀) |
| StarVLA-GR00T | QwenGR00T | FlowmatchingActionHead | Dual-system VLM+FlowMatching (GR00T N1.5) |

WM4A variants (`WM4A/`) use pretrained video-generation DiT models (Cosmos-Predict2, Wan2.2) as backbones instead of VLMs.

## Key Configuration Patterns

**Different learning rates per module** (in YAML):
```yaml
trainer:
  learning_rate:
    base: 1e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```

**Freeze specific modules** (CLI or YAML):
```bash
--trainer.freeze_modules "qwen_vl_interface.model.model.visual,dino_encoder"
```

**Resume from checkpoint** (in YAML):
```yaml
trainer:
  pretrained_checkpoint: path/to/steps_10000.pt
  reload_modules: "action_model"   # empty string = full model
```

## Git Conventions

- Directories named `bar` or matching `**/bar/` are git-ignored — place personal scripts there
- Conda environment name: `starVLA`
- PRs should follow `docs/PR_readme.md` and `docs/branching_strategy.md`
