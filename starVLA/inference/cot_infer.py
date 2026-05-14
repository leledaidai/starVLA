from __future__ import annotations

from typing import Any

import torch

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config


def run_implicit_cot_inference(model, examples: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    return model.predict_action(examples=examples, **kwargs)


def run_teacher_only_inference(model, examples: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    if not hasattr(model, "predict_action_teacher_only"):
        raise AttributeError(f"{type(model).__name__} does not implement predict_action_teacher_only().")
    return model.predict_action_teacher_only(examples=examples, **kwargs)


def load_implicit_cot_checkpoint(
    checkpoint_path: str,
    *,
    decode_cot_in_inference: bool = False,
):
    model_config, norm_stats = read_mode_config(checkpoint_path)
    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None
    cfg.cot.enable_teacher_cot_loss = False
    cfg.cot.enable_student_action_loss = True
    cfg.cot.enable_decoder_loss = False
    cfg.cot.enable_slot_distill_loss = False
    cfg.cot.enable_pool_distill_loss = False
    cfg.cot.decode_cot_in_inference = bool(decode_cot_in_inference)

    model = build_framework(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.norm_stats = norm_stats
    return model
