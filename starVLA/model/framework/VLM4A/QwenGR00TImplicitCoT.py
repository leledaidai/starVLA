from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.cot_formatter import (
    build_prefix_message,
    build_student_message,
    build_teacher_message,
    build_visible_cot_text,
    parse_cot_fields,
    parse_thinking_tokens,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)
IGNORE_INDEX = -100
_VALID_DISTILL_LOSS_TYPES = frozenset({"smooth_l1", "l2"})


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass
class QwenGR00TImplicitCoTDefaultConfig:
    name: str = "QwenGR00TImplicitCoT"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
        "vl_hidden_dim": 2048,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 7,
        "action_horizon": 8,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })
    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("QwenGR00TImplicitCoT")
class QwenGR00TImplicitCoT(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TImplicitCoTDefaultConfig, config)
        self.cot_cfg = self.config.cot
        self.field_specs = parse_cot_fields(self.cot_cfg)
        self.field_names = [spec.name for spec in self.field_specs]
        self.token_spec = parse_thinking_tokens(self.cot_cfg)

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.processor = self.qwen_vl_interface.processor
        self.tokenizer = self.processor.tokenizer
        self._add_thinking_tokens()

        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.latent_decode_max_tokens = int(_cfg_get(self.cot_cfg, "max_cot_length", 64))

        self.eval_mode = str(_cfg_get(self.cot_cfg, "eval_mode", "student"))
        self.enable_teacher_cot_loss = bool(_cfg_get(self.cot_cfg, "enable_teacher_cot_loss", False))
        self.enable_teacher_action_loss = bool(_cfg_get(self.cot_cfg, "enable_teacher_action_loss", False))
        self.enable_student_action_loss = bool(_cfg_get(self.cot_cfg, "enable_student_action_loss", False))
        self.enable_decoder_loss = bool(_cfg_get(self.cot_cfg, "enable_decoder_loss", False))
        self.decoder_type = str(_cfg_get(self.cot_cfg, "decoder_type", "qwen_text"))
        self.decode_cot_in_inference = bool(_cfg_get(self.cot_cfg, "decode_cot_in_inference", False))
        self.teacher_cot_loss_weight = float(_cfg_get(self.cot_cfg, "teacher_cot_loss_weight", 1.0))
        self.teacher_action_loss_weight = float(_cfg_get(self.cot_cfg, "teacher_action_loss_weight", 1.0))
        self.student_action_loss_weight = float(_cfg_get(self.cot_cfg, "student_action_loss_weight", 1.0))
        self.decoder_loss_weight = float(_cfg_get(self.cot_cfg, "decoder_loss_weight", 1.0))
        self.decoder_model_path = str(_cfg_get(self.cot_cfg, "decoder_model_path",
            "./playground/Pretrained_models/Qwen3-1.7B"))
        self.distill_loss_type = str(_cfg_get(self.cot_cfg, "distill_loss_type", "smooth_l1"))
        self.distill_loss_div_std = bool(_cfg_get(self.cot_cfg, "distill_loss_div_std", True))
        self.enable_slot_distill_loss = bool(_cfg_get(self.cot_cfg, "enable_slot_distill_loss", True))
        self.enable_pool_distill_loss = bool(_cfg_get(self.cot_cfg, "enable_pool_distill_loss", True))
        self.slot_distill_loss_weight = float(_cfg_get(self.cot_cfg, "slot_distill_loss_weight", 1.0))
        self.pool_distill_loss_weight = float(_cfg_get(self.cot_cfg, "pool_distill_loss_weight", 0.2))
        self.has_any_distill_loss = self.enable_slot_distill_loss or self.enable_pool_distill_loss
        self.latent_forward_use_cache = bool(_cfg_get(self.cot_cfg, "latent_forward_use_cache", False))
        if self.distill_loss_type not in _VALID_DISTILL_LOSS_TYPES:
            raise ValueError(
                f"Unsupported cot.distill_loss_type={self.distill_loss_type!r}. "
                f"Expected one of {sorted(_VALID_DISTILL_LOSS_TYPES)}."
            )

        self.decoder_language_model: Optional[nn.Module] = None
        self.decoder_lm_head: Optional[nn.Module] = None
        if self.enable_decoder_loss:
            self._build_text_decoder()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _add_thinking_tokens(self) -> None:
        existing = set(self.tokenizer.get_vocab().keys())
        to_add = []
        for token_text in (
            self.token_spec.thinking_token,
            self.token_spec.start_token,
            self.token_spec.end_token,
        ):
            if token_text not in existing:
                to_add.append(token_text)

        if to_add:
            additional = list(getattr(self.tokenizer, "additional_special_tokens", []) or [])
            for token_text in to_add:
                if token_text not in additional:
                    additional.append(token_text)
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional})

        old_vocab_size = self.qwen_vl_interface.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(self.tokenizer)
        if new_vocab_size > old_vocab_size:
            self.qwen_vl_interface.model.resize_token_embeddings(new_vocab_size)

        self.thinking_token_id = self.tokenizer.convert_tokens_to_ids(self.token_spec.thinking_token)
        self.start_thinking_id = self.tokenizer.convert_tokens_to_ids(self.token_spec.start_token)
        self.end_thinking_id = self.tokenizer.convert_tokens_to_ids(self.token_spec.end_token)
        if any(
            token_id is None or token_id == self.tokenizer.unk_token_id
            for token_id in (self.thinking_token_id, self.start_thinking_id, self.end_thinking_id)
        ):
            raise ValueError("Failed to register CoT thinking tokens in tokenizer.")

        input_embeddings = self.qwen_vl_interface.model.get_input_embeddings()
        reference_id = 0
        while reference_id < input_embeddings.weight.shape[0]:
            token_text = self.tokenizer.convert_ids_to_tokens(reference_id)
            if token_text and not (token_text.startswith("<") and token_text.endswith(">")):
                break
            reference_id += 1
        if reference_id >= input_embeddings.weight.shape[0]:
            reference_id = 0

        reference_embedding = input_embeddings.weight.data[reference_id].clone()
        for token_id in (self.thinking_token_id, self.start_thinking_id, self.end_thinking_id):
            input_embeddings.weight.data[token_id] = reference_embedding

        lm_head = getattr(self.qwen_vl_interface.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            reference_lm = lm_head.weight.data[reference_id].clone()
            for token_id in (self.thinking_token_id, self.start_thinking_id, self.end_thinking_id):
                lm_head.weight.data[token_id] = reference_lm

    def _move_qwen_inputs(self, qwen_inputs: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in qwen_inputs.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device)
            else:
                out[key] = value
        return out

    def _forward_standard(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        model = self.qwen_vl_interface.model
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self._masked_lm_loss(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden_states,))

    def _masked_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shifted_labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous()
        valid_mask = shifted_labels.ne(IGNORE_INDEX)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            safe_labels.reshape(-1),
            reduction="none",
        ).view_as(shifted_labels)
        valid_mask_f = valid_mask.to(token_losses.dtype)
        return (token_losses * valid_mask_f).sum() / valid_mask_f.sum().clamp_min(1.0)

    def _build_text_decoder(self) -> None:
        if self.decoder_type not in {"qwen_text", "qwen_vl_text"}:
            raise ValueError(
                f"Unsupported cot.decoder_type={self.decoder_type!r}. "
                "Expected one of: `qwen_text`, `qwen_vl_text`."
            )
        if self.decoder_language_model is not None and self.decoder_lm_head is not None:
            return

        main_dtype = next(self.qwen_vl_interface.parameters()).dtype
        if self.decoder_type == "qwen_text":
            decoder_model = AutoModelForCausalLM.from_pretrained(
                self.decoder_model_path,
                torch_dtype=main_dtype,
                trust_remote_code=True,
            )
            if not hasattr(decoder_model, "model"):
                raise ValueError(
                    "`cot.decoder_type=qwen_text` expects a causal LM checkpoint with a `.model` module. "
                    f"Got decoder_model_path={self.decoder_model_path!r}."
                )
            decoder_hidden = getattr(decoder_model.config, "hidden_size", None)
            self.decoder_language_model = decoder_model.model
        else:
            decoder_model = AutoModelForImageTextToText.from_pretrained(
                self.decoder_model_path,
                torch_dtype=main_dtype,
                trust_remote_code=True,
            )
            if not hasattr(decoder_model, "language_model") or not hasattr(decoder_model.config, "text_config"):
                raise ValueError(
                    "`cot.decoder_type=qwen_vl_text` expects a VLM checkpoint with `.language_model` "
                    f"and `config.text_config`. Got decoder_model_path={self.decoder_model_path!r}."
                )
            decoder_hidden = getattr(decoder_model.config.text_config, "hidden_size", None)
            self.decoder_language_model = decoder_model.language_model

        if not hasattr(decoder_model, "lm_head"):
            raise ValueError(f"Decoder checkpoint has no `lm_head`: {self.decoder_model_path!r}.")
        if decoder_hidden is None:
            raise ValueError(f"Failed to resolve decoder hidden size from {self.decoder_model_path!r}.")
        self.decoder_lm_head = decoder_model.lm_head
        self.decoder_language_model.to(self.device)
        self.decoder_lm_head.to(self.device)

        main_hidden = self.qwen_vl_interface.model.config.hidden_size
        if not hasattr(self, "decoder_projection") or self.decoder_projection is None:
            self.decoder_projection = nn.Linear(main_hidden, decoder_hidden)
            self.decoder_projection.to(self.device)

        for param in self.decoder_language_model.parameters():
            param.requires_grad = True
        for param in self.decoder_lm_head.parameters():
            param.requires_grad = True
        for param in self.decoder_projection.parameters():
            param.requires_grad = True

    def _require_text_decoder(self) -> tuple[nn.Module, nn.Module]:
        if self.decoder_language_model is None or self.decoder_lm_head is None:
            raise RuntimeError(
                "CoT decoder is not loaded. Set cot.enable_decoder_loss=true for training, "
                "or cot.decode_cot_in_inference=true / decode_cot=True for inference-time latent decoding."
            )
        return self.decoder_language_model, self.decoder_lm_head

    def _forward_latent(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thinking_positions: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if self.latent_forward_use_cache:
            return self._forward_latent_cached(
                input_ids=input_ids,
                attention_mask=attention_mask,
                thinking_positions=thinking_positions,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        return self._forward_latent_recompute(
            input_ids=input_ids,
            attention_mask=attention_mask,
            thinking_positions=thinking_positions,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _forward_latent_recompute(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thinking_positions: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        model = self.qwen_vl_interface.model
        if thinking_positions.numel() == 0:
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            return {
                "hidden_states": outputs.last_hidden_state,
                "num_reasoning_passes": 0,
            }

        max_n_latents = int(thinking_positions.shape[1])
        batch_indices = torch.arange(batch_size, device=input_ids.device)

        inputs_embeds = model.get_input_embeddings()(input_ids)
        position_ids, _ = model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        for pass_idx in range(max_n_latents):
            token_idx = thinking_positions[:, pass_idx]
            end_idx = int(token_idx.max().item())
            if pass_idx == 0:
                # First pass: use input_ids for reliable image-token detection
                # via integer token-ID comparison (not bf16 embedding comparison).
                outputs = model.model(
                    input_ids=input_ids[:, :end_idx],
                    attention_mask=attention_mask[:, :end_idx],
                    position_ids=position_ids[:, :, :end_idx],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                    return_dict=True,
                )
            else:
                outputs = model.model(
                    inputs_embeds=inputs_embeds[:, :end_idx, :],
                    attention_mask=attention_mask[:, :end_idx],
                    position_ids=position_ids[:, :, :end_idx],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                    return_dict=True,
                )
            hidden_states = outputs.last_hidden_state

            updated_embeds = inputs_embeds.clone()
            local_pos = token_idx - 1
            if bool((local_pos < 0).any()) or bool((local_pos >= hidden_states.shape[1]).any()):
                raise ValueError(
                    "Invalid recomputed latent update: "
                    f"token_idx range=({int(token_idx.min().item())}, {int(token_idx.max().item())}), "
                    f"local_pos range=({int(local_pos.min().item())}, {int(local_pos.max().item())}), "
                    f"hidden_len={hidden_states.shape[1]}."
                )
            source_hidden = hidden_states[batch_indices, local_pos, :]
            projected_hidden = self._project_latent_hidden(
                source_hidden,
                target_dtype=updated_embeds.dtype,
            )
            updated_embeds[batch_indices, token_idx, :] = projected_hidden
            inputs_embeds = updated_embeds

        # Final pass: reprocess full sequence with updated thinking-token embeddings.
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        return {
            "hidden_states": hidden_states,
            "num_reasoning_passes": max_n_latents + 1,
        }

    def _forward_latent_cached(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thinking_positions: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        model = self.qwen_vl_interface.model
        if thinking_positions.numel() == 0:
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            return {
                "hidden_states": outputs.last_hidden_state,
                "num_reasoning_passes": 0,
            }

        max_n_latents = int(thinking_positions.shape[1])
        batch_indices = torch.arange(batch_size, device=input_ids.device)

        inputs_embeds = model.get_input_embeddings()(input_ids)
        position_ids, _ = model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        if not torch.all(thinking_positions == thinking_positions[:1, :]):
            raise ValueError(
                "Cached latent forward requires aligned thinking_positions across the batch. "
                f"Got min={thinking_positions.min(dim=0).values.tolist()}, "
                f"max={thinking_positions.max(dim=0).values.tolist()}."
            )
        expected_thinking_positions = thinking_positions[:, :1] + torch.arange(
            max_n_latents,
            device=thinking_positions.device,
            dtype=thinking_positions.dtype,
        ).unsqueeze(0)
        if not torch.all(thinking_positions == expected_thinking_positions):
            raise ValueError(
                "Cached latent forward requires consecutive thinking tokens. "
                f"Got first row={thinking_positions[0].tolist()}."
            )
        earliest_latent_pos = int(thinking_positions[:, 0].min().item())
        next_compute_range = (0, earliest_latent_pos)
        kv_cache = None
        full_hidden_states = torch.zeros(
            (batch_size, seq_len, inputs_embeds.shape[-1]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        for pass_idx in range(max_n_latents):
            start_idx, end_idx = next_compute_range
            if kv_cache is None:
                # Pass input_ids (not inputs_embeds) so Qwen3VL uses integer
                # token-ID comparison to locate image tokens, avoiding bf16
                # embedding-comparison failures in get_placeholder_mask.
                outputs = model.model(
                    input_ids=input_ids[:, start_idx:end_idx],
                    attention_mask=attention_mask[:, :end_idx],
                    position_ids=position_ids[:, :, start_idx:end_idx],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_states_offset = 0
            else:
                cache_position = torch.arange(start_idx, end_idx, device=inputs_embeds.device)
                outputs = model.model(
                    inputs_embeds=inputs_embeds[:, start_idx:end_idx, :],
                    attention_mask=attention_mask[:, :end_idx],
                    position_ids=position_ids[:, :, start_idx:end_idx],
                    pixel_values=None,
                    image_grid_thw=None,
                    past_key_values=kv_cache,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_states_offset = start_idx

            hidden_states = outputs.last_hidden_state
            kv_cache = outputs.past_key_values
            full_hidden_states[:, start_idx:end_idx, :] = hidden_states

            updated_embeds = inputs_embeds.clone()
            token_idx = thinking_positions[:, pass_idx]
            local_pos = token_idx - 1 - hidden_states_offset
            if bool((local_pos < 0).any()) or bool((local_pos >= hidden_states.shape[1]).any()):
                raise ValueError(
                    "Invalid cached latent update: "
                    f"token_idx range=({int(token_idx.min().item())}, {int(token_idx.max().item())}), "
                    f"local_pos range=({int(local_pos.min().item())}, {int(local_pos.max().item())}), "
                    f"hidden_len={hidden_states.shape[1]}, hidden_states_offset={hidden_states_offset}."
                )
            source_hidden = hidden_states[batch_indices, local_pos, :]
            projected_hidden = self._project_latent_hidden(
                source_hidden,
                target_dtype=updated_embeds.dtype,
            )
            updated_embeds[batch_indices, token_idx, :] = projected_hidden
            inputs_embeds = updated_embeds

            if pass_idx + 1 >= max_n_latents:
                next_compute_range = (end_idx, seq_len)
            else:
                next_compute_range = (end_idx, end_idx + 1)
        #the final
        start_idx, end_idx = next_compute_range
        if kv_cache is None:
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            full_hidden_states = hidden_states
        else:
            cache_position = torch.arange(start_idx, end_idx, device=inputs_embeds.device)
            outputs = model.model(
                inputs_embeds=inputs_embeds[:, start_idx:end_idx, :],
                attention_mask=attention_mask[:, :end_idx],
                position_ids=position_ids[:, :, start_idx:end_idx],
                pixel_values=None,
                image_grid_thw=None,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            full_hidden_states[:, start_idx:end_idx, :] = hidden_states

        hidden_states = full_hidden_states
        return {
            "hidden_states": hidden_states,
            "num_reasoning_passes": max_n_latents + 1,
        }

    def _teacher_forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | Any]:
        qwen_inputs = {
            "input_ids": batch["teacher_input_ids"].to(self.device),
            "attention_mask": batch["teacher_attention_mask"].to(self.device),
        }
        for optional_key in ("pixel_values", "image_grid_thw"):
            if optional_key in batch:
                qwen_inputs[optional_key] = batch[optional_key].to(self.device)
        model = self.qwen_vl_interface.model
        outputs = model.model(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            pixel_values=qwen_inputs.get("pixel_values"),
            image_grid_thw=qwen_inputs.get("image_grid_thw"),
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.lm_head(hidden_states)
        teacher_cot_loss = hidden_states.new_zeros(())
        if self.enable_teacher_cot_loss and "labels" in batch:
            teacher_cot_loss = self._masked_lm_loss(logits, batch["labels"].to(self.device))
        field_hidden = None
        if self.has_any_distill_loss:
            if "teacher_field_positions" in batch:
                teacher_field_positions = batch["teacher_field_positions"].to(self.device)
                hidden_size = hidden_states.shape[-1]
                clamped_positions = teacher_field_positions.clamp(min=0)
                gather_index = clamped_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
                field_hidden = torch.gather(hidden_states, 1, gather_index)
                field_mask = (teacher_field_positions >= 0).unsqueeze(-1)
                field_hidden = field_hidden * field_mask.to(field_hidden.dtype)
            else:
                field_hidden = hidden_states.new_zeros((hidden_states.shape[0], len(self.field_names), hidden_states.shape[-1]))

        return {
            "teacher_cot_loss": teacher_cot_loss,
            "hidden_states": hidden_states,
            "field_hidden": field_hidden,
        }

    def _student_forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | Any]:
        qwen_inputs = {
            "input_ids": batch["student_input_ids"].to(self.device),
            "attention_mask": batch["student_attention_mask"].to(self.device),
        }
        for optional_key in ("pixel_values", "image_grid_thw"):
            if optional_key in batch:
                qwen_inputs[optional_key] = batch[optional_key].to(self.device)
        thinking_positions = batch["thinking_positions"].to(self.device)
        outputs = self._forward_latent(thinking_positions=thinking_positions, **qwen_inputs)
        hidden_states = outputs["hidden_states"]
        hidden_size = hidden_states.shape[-1]
        gather_index = thinking_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        latent_hidden = torch.gather(hidden_states, 1, gather_index)
        return {
            "hidden_states": hidden_states,
            "latent_hidden": latent_hidden,
            "num_reasoning_passes": outputs["num_reasoning_passes"],
        }

    def _prepare_state_tensor(self, batch: dict[str, Any], hidden_dtype: torch.dtype) -> Optional[torch.Tensor]:
        if "state" not in batch:
            return None
        state = batch["state"].to(self.device, dtype=hidden_dtype)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        return state

    def _action_loss_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        action_dtype = next(self.action_model.parameters()).dtype
        repeated_hidden = hidden_states.to(dtype=action_dtype).repeat(repeated_diffusion_steps, 1, 1)
        repeated_attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool).repeat(
            repeated_diffusion_steps, 1
        )
        repeated_actions = actions.to(dtype=action_dtype).repeat(repeated_diffusion_steps, 1, 1)
        repeated_state = state.to(dtype=action_dtype).repeat(repeated_diffusion_steps, 1, 1) if state is not None else None
        return self.action_model(
            repeated_hidden,
            repeated_actions,
            repeated_state,
            encoder_attention_mask=repeated_attention_mask,
        )

    def _predict_action_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_dtype = next(self.action_model.parameters()).dtype
        hidden_states = hidden_states.to(dtype=action_dtype)
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if state is not None:
            state = state.to(dtype=action_dtype)
        return self.action_model.predict_action(hidden_states, state, encoder_attention_mask=attention_mask)

    def _project_latent_hidden(self, hidden_state: torch.Tensor, *, target_dtype: torch.dtype) -> torch.Tensor:
        projection_dtype = self.latent_projection[0].weight.dtype
        projected = self.latent_projection(hidden_state.to(dtype=projection_dtype))
        return projected.to(dtype=target_dtype)

    def _gather_positions(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        gather_index = positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden_size)
        gathered = torch.gather(hidden_states, 1, gather_index)
        return gathered * (positions >= 0).unsqueeze(-1).to(gathered.dtype)

    def _embed_decoder_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp(min=0)
        return self.decoder_language_model.embed_tokens(token_ids)

    def _run_decoder_teacher_forcing(
        self,
        *,
        latent_hidden: torch.Tensor,
        token_targets: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_targets.shape[0] == 0 or token_targets.shape[-1] == 0:
            zero = latent_hidden.new_zeros(())
            return zero, zero

        token_mask = token_mask.to(device=latent_hidden.device, dtype=torch.bool)
        token_targets = token_targets.to(device=latent_hidden.device)
        valid_lengths = token_mask.long().sum(dim=1)
        # Keep a decoder call even for all-empty slots. With ZeRO/DeepSpeed, rank-local
        # skipping can make some ranks enter decoder collectives while others do not.
        max_valid_len = max(int(valid_lengths.max().item()), 1)

        token_targets = token_targets[:, :max_valid_len]
        token_mask = token_mask[:, :max_valid_len]

        lm, lm_head = self._require_text_decoder()
        decoder_input_ids = token_targets[:, :-1]

        prefix = self.decoder_projection(latent_hidden).unsqueeze(1)
        if decoder_input_ids.shape[1] > 0:
            text_embeds = self._embed_decoder_tokens(decoder_input_ids)
            inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
        else:
            # Keep embed_tokens in the graph on every rank. When a rank's local
            # batch contains only empty CoT slots, decoder_input_ids has length 0,
            # and skipping embed_tokens here can leave that parameter unused on
            # this rank while other ranks do use it, which may hang distributed
            # backward/gradient sync.
            dummy_token_ids = torch.zeros(
                (latent_hidden.shape[0], 1),
                dtype=torch.long,
                device=latent_hidden.device,
            )
            dummy_embeds = self._embed_decoder_tokens(dummy_token_ids)
            inputs_embeds = prefix + dummy_embeds * 0.0

        attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        decoder_outputs = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = decoder_outputs.last_hidden_state
        valid_token_mask = token_mask & token_targets.ge(0)
        token_count = valid_token_mask.sum()
        if bool(valid_token_mask.any()):
            selected_hidden_states = hidden_states[valid_token_mask]
            selected_targets = token_targets[valid_token_mask]
            logits = lm_head(selected_hidden_states)
            loss_sum = F.cross_entropy(
                logits.float(),
                selected_targets,
                reduction="sum",
            )
        else:
            # Still execute lm_head on every rank for ZeRO-3 module-call consistency.
            dummy_logits = lm_head(hidden_states[:, :1, :])
            loss_sum = dummy_logits.sum() * 0.0

        return loss_sum, token_count

    def _generate_from_latent(
        self,
        latent_hidden: torch.Tensor,
        *,
        max_new_tokens: int,
    ) -> list[str]:
        if max_new_tokens <= 0:
            return ["" for _ in range(latent_hidden.shape[0])]

        lm, lm_head = self._require_text_decoder()
        eos_token_id = self.tokenizer.eos_token_id
        decoder_latent = self.decoder_projection(latent_hidden)
        generated: list[list[int]] = [[] for _ in range(latent_hidden.shape[0])]
        generated_ids = torch.empty((latent_hidden.shape[0], 0), dtype=torch.long, device=latent_hidden.device)
        finished = torch.zeros(latent_hidden.shape[0], dtype=torch.bool, device=latent_hidden.device)

        for _ in range(max_new_tokens):
            if generated_ids.shape[1] > 0:
                text_embeds = self._embed_decoder_tokens(generated_ids)
                current_embeds = torch.cat([decoder_latent.unsqueeze(1), text_embeds], dim=1)
            else:
                current_embeds = decoder_latent.unsqueeze(1)
            step_outputs = lm(
                inputs_embeds=current_embeds,
                attention_mask=torch.ones(current_embeds.shape[:2], dtype=torch.long, device=current_embeds.device),
                use_cache=False,
                return_dict=True,
            )
            next_logits = lm_head(step_outputs.last_hidden_state[:, -1, :])
            next_token = torch.argmax(next_logits, dim=-1)
            if eos_token_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            for batch_idx, token_id in enumerate(next_token.tolist()):
                if finished[batch_idx]:
                    continue
                if eos_token_id is not None and token_id == eos_token_id:
                    finished[batch_idx] = True
                    continue
                generated[batch_idx].append(token_id)
            if finished.all():
                break
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

        return [self.tokenizer.decode(token_ids, skip_special_tokens=True).strip() for token_ids in generated]

    def _decoder_loss(
        self,
        latent_hidden: torch.Tensor,
        cot_labels: torch.Tensor,
        cot_label_mask: torch.Tensor,
        cot_slot_mask: torch.Tensor,
        *,
        decode_text: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[list[str]]]:
        batch_size, num_slots, _ = cot_labels.shape
        total_loss = latent_hidden.new_zeros(())
        total_weight = latent_hidden.new_zeros(())
        per_slot_losses: dict[str, torch.Tensor] = {}
        decoded_texts: list[list[str]] = [[] for _ in range(batch_size)]

        for slot_idx in range(num_slots):
            field_name = self.field_names[slot_idx] if slot_idx < len(self.field_names) else f"slot_{slot_idx}"
            slot_mask = cot_slot_mask[:, slot_idx].to(device=latent_hidden.device, dtype=torch.bool)

            slot_latent = latent_hidden[:, slot_idx, :]
            slot_labels = cot_labels[:, slot_idx, :].to(device=latent_hidden.device)
            slot_label_mask = cot_label_mask[:, slot_idx, :].to(device=latent_hidden.device) & slot_mask.unsqueeze(1)
            slot_loss_sum, slot_token_count = self._run_decoder_teacher_forcing(
                latent_hidden=slot_latent,
                token_targets=slot_labels,
                token_mask=slot_label_mask,
            )
            total_loss = total_loss + slot_loss_sum
            total_weight = total_weight + slot_token_count.to(total_loss.dtype)

            # Per-slot average decoder loss for logging (detached to avoid
            # holding unnecessary computation graph; does not affect training).
            per_slot_losses[f"decoder_loss_{field_name}"] = (
                slot_loss_sum / slot_token_count.clamp_min(1.0)
            ).detach()

            if decode_text:
                if bool(slot_mask.any()):
                    active_latent = slot_latent[slot_mask]
                    generated_slot_texts = self._generate_from_latent(
                        active_latent,
                        max_new_tokens=self.latent_decode_max_tokens,
                    )
                    active_index = 0
                    for sample_idx in range(batch_size):
                        if slot_mask[sample_idx]:
                            decoded_texts[sample_idx].append(generated_slot_texts[active_index])
                            active_index += 1
                        else:
                            decoded_texts[sample_idx].append("")
                else:
                    for sample_idx in range(batch_size):
                        decoded_texts[sample_idx].append("")
            else:
                continue

        return total_loss / total_weight.clamp_min(1.0), per_slot_losses, decoded_texts

    def _vector_distill_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if student.shape != teacher.shape:
            raise ValueError(f"Distill student/teacher shape mismatch: {student.shape} vs {teacher.shape}.")
        if mask.shape != student.shape[:-1]:
            raise ValueError(f"Distill mask shape mismatch: {mask.shape} vs expected {student.shape[:-1]}.")

        teacher = teacher.detach().to(dtype=student.dtype)
        mask_bool = mask.to(device=student.device, dtype=torch.bool)
        if not bool(mask_bool.any()):
            return student.sum() * 0.0
        mask = mask_bool.to(dtype=student.dtype)
        delta = student - teacher
        if self.distill_loss_type == "smooth_l1":
            abs_delta = delta.abs()
            per_dim = torch.where(abs_delta < 1.0, 0.5 * delta.square(), abs_delta - 0.5)
        else:
            per_dim = delta.square()

        denom = (mask.sum().clamp_min(1.0) * delta.shape[-1]).to(per_dim.dtype)
        loss = (per_dim * mask.unsqueeze(-1)).sum() / denom
        if self.distill_loss_div_std:
            teacher_selected = teacher[mask_bool]
            teacher_std = teacher_selected.std().clamp_min(torch.finfo(per_dim.dtype).eps)
            loss = loss / teacher_std
        return loss

    def _pool_hidden(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _distill_loss_from_slots(
        self,
        *,
        student_latent_hidden: torch.Tensor,
        teacher_field_hidden: torch.Tensor,
        cot_slot_mask: torch.Tensor,
        has_cot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_mask = cot_slot_mask.to(student_latent_hidden.device, dtype=torch.bool) & has_cot.to(
            student_latent_hidden.device, dtype=torch.bool
        )[:, None]
        zero = student_latent_hidden.sum() * 0.0

        slot_distill_loss = zero
        if self.enable_slot_distill_loss:
            slot_distill_loss = self._vector_distill_loss(
                student_latent_hidden,
                teacher_field_hidden,
                slot_mask,
            )

        pool_distill_loss = zero
        if self.enable_pool_distill_loss:
            sample_mask = slot_mask.any(dim=1)
            student_pool = self._pool_hidden(student_latent_hidden, slot_mask)
            teacher_pool = self._pool_hidden(teacher_field_hidden, slot_mask)
            pool_distill_loss = self._vector_distill_loss(student_pool, teacher_pool, sample_mask)

        return slot_distill_loss, pool_distill_loss

    def forward(self, batch: dict[str, Any] = None, **kwargs) -> dict[str, torch.Tensor | Any]:
        if batch is None:
            raise ValueError("QwenGR00TImplicitCoT.forward expects a collated batch dict.")
        #teacher
        teacher_outputs = None
        teacher_needs_grad = self.enable_teacher_cot_loss or self.enable_teacher_action_loss
        if teacher_needs_grad or self.has_any_distill_loss:
            teacher_forward_context = nullcontext() if teacher_needs_grad else torch.no_grad()
            with teacher_forward_context:
                teacher_outputs = self._teacher_forward(batch)
        #student
        student_outputs = None
        if self.enable_student_action_loss or self.enable_decoder_loss or self.has_any_distill_loss:
            student_outputs = self._student_forward(batch)
        #action
        action_labels = batch["action_labels"].to(self.device, dtype=torch.float32)
        if action_labels.ndim != 3:
            raise ValueError(
                f"Expected action_labels to have shape [B, T, D], got {tuple(action_labels.shape)}."
            )
        if action_labels.shape[1] != self.chunk_len:
            raise ValueError(
                "Action chunk length mismatch: "
                f"got action_labels.shape[1]={action_labels.shape[1]}, expected chunk_len={self.chunk_len} "
                f"(past_action_window_size={self.past_action_window_size}, "
                f"future_action_window_size={self.future_action_window_size}). "
                "Please keep dataset action_indices and action-model horizon aligned."
            )
        state_tensor = self._prepare_state_tensor(batch, hidden_dtype=torch.float32)
        loss_mask = batch.get("has_cot", None)
        if loss_mask is None:
            loss_mask = torch.ones(action_labels.shape[0], dtype=torch.bool, device=self.device)
        else:
            loss_mask = loss_mask.to(self.device, dtype=torch.bool)

        teacher_cot_loss = (
            teacher_outputs["teacher_cot_loss"] if teacher_outputs is not None else action_labels.new_zeros(())
        )
        teacher_action_loss = action_labels.new_zeros(())
        if teacher_outputs is not None and self.enable_teacher_action_loss:
            teacher_hidden = teacher_outputs["hidden_states"]
            teacher_action_loss = self._action_loss_from_hidden(
                teacher_hidden,
                batch["teacher_attention_mask"].to(self.device),
                action_labels,
                state_tensor.to(dtype=teacher_hidden.dtype) if state_tensor is not None else None,
            )
        
        student_action_loss = action_labels.new_zeros(())
        decoder_loss = action_labels.new_zeros(())
        slot_distill_loss = action_labels.new_zeros(())
        pool_distill_loss = action_labels.new_zeros(())
        sum_of_distill_loss = action_labels.new_zeros(())
        decoded_cot_texts: list[list[str]] = [[] for _ in range(action_labels.shape[0])]
        per_slot_decoder_losses: dict[str, torch.Tensor] = {}
        if student_outputs is not None:
            student_hidden = student_outputs["hidden_states"]
            latent_hidden = student_outputs["latent_hidden"]
            if self.enable_student_action_loss:
                student_action_loss = self._action_loss_from_hidden(
                    student_hidden,
                    batch["student_attention_mask"].to(self.device),
                    action_labels,
                    state_tensor.to(dtype=student_hidden.dtype) if state_tensor is not None else None,
                )
            if self.enable_decoder_loss:
                decoder_loss, per_slot_decoder_losses, decoded_cot_texts = self._decoder_loss(
                    latent_hidden=latent_hidden,
                    cot_labels=batch["cot_labels"].to(self.device),
                    cot_label_mask=batch["cot_label_mask"].to(self.device),
                    cot_slot_mask=batch["cot_slot_mask"].to(self.device),
                )
            if self.has_any_distill_loss and teacher_outputs is not None:
                slot_distill_loss, pool_distill_loss = self._distill_loss_from_slots(
                    student_latent_hidden=latent_hidden,
                    teacher_field_hidden=teacher_outputs["field_hidden"],
                    cot_slot_mask=batch["cot_slot_mask"].to(self.device),
                    has_cot=loss_mask,
                )
                sum_of_distill_loss = (
                    self.slot_distill_loss_weight * slot_distill_loss
                    + self.pool_distill_loss_weight * pool_distill_loss
                )

        total_loss = action_labels.new_zeros(())
        if self.enable_teacher_cot_loss:
            total_loss = total_loss + self.teacher_cot_loss_weight * teacher_cot_loss
        if self.enable_teacher_action_loss:
            total_loss = total_loss + self.teacher_action_loss_weight * teacher_action_loss
        if self.enable_student_action_loss:
            total_loss = total_loss + self.student_action_loss_weight * student_action_loss
        if self.enable_decoder_loss:
            total_loss = total_loss + self.decoder_loss_weight * decoder_loss
        if self.enable_slot_distill_loss:
            total_loss = total_loss + self.slot_distill_loss_weight * slot_distill_loss
        if self.enable_pool_distill_loss:
            total_loss = total_loss + self.pool_distill_loss_weight * pool_distill_loss

        # --- NaN diagnostics: log every component before returning ---
        #$modify for nan debug
        loss_items = {
            "teacher_cot_loss": teacher_cot_loss,
            "teacher_action_loss": teacher_action_loss,
            "student_action_loss": student_action_loss,
            "decoder_loss": decoder_loss,
            "slot_distill_loss": slot_distill_loss,
            "pool_distill_loss": pool_distill_loss,
            "total_loss": total_loss,
        }
        nan_losses = [name for name, val in loss_items.items() if torch.isnan(val).any() or torch.isinf(val).any()]
        if nan_losses:
            print(f"[NaN DIAG] NaN/Inf detected in losses: {nan_losses}", flush=True)
            for name, val in loss_items.items():
                v = float(val) if not (torch.isnan(val).any() or torch.isinf(val).any()) else "NaN/Inf"
                print(f"[NaN DIAG]   {name}: {v}", flush=True)
            # Scan intermediate tensors from teacher/student outputs
            if teacher_outputs is not None:
                for sub_key in ["hidden_states", "field_hidden", "teacher_cot_loss"]:
                    t = teacher_outputs.get(sub_key)
                    if t is not None and isinstance(t, torch.Tensor):
                        n = torch.isnan(t).sum().item()
                        i = torch.isinf(t).sum().item()
                        if n > 0 or i > 0:
                            print(f"[NaN DIAG]   teacher.{sub_key}: NaN={n}, Inf={i}, range=[{float(t.min()):.4e}, {float(t.max()):.4e}]", flush=True)
            if student_outputs is not None:
                for sub_key in ["hidden_states", "latent_hidden"]:
                    t = student_outputs.get(sub_key)
                    if t is not None and isinstance(t, torch.Tensor):
                        n = torch.isnan(t).sum().item()
                        i = torch.isinf(t).sum().item()
                        if n > 0 or i > 0:
                            print(f"[NaN DIAG]   student.{sub_key}: NaN={n}, Inf={i}, range=[{float(t.min()):.4e}, {float(t.max()):.4e}]", flush=True)
            # Scan which parameter gradients carry NaN/Inf
            for pname, param in self.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        n = torch.isnan(param.grad).sum().item()
                        i = torch.isinf(param.grad).sum().item()
                        print(f"[NaN GRAD] {pname}: grad NaN={n}, Inf={i}, grad_norm={param.grad.norm():.2e}", flush=True)
        #$modify for nan debug
        return {
            "loss": total_loss,
            "teacher_cot_loss": teacher_cot_loss,
            "student_action_loss": student_action_loss,
            "decoder_loss": decoder_loss,
            "slot_distill_loss": slot_distill_loss,
            "pool_distill_loss": pool_distill_loss,
            "sum_of_distill_loss": sum_of_distill_loss,
            "teacher_action_loss": teacher_action_loss.detach(),
            "num_reasoning_passes": torch.tensor(
                student_outputs["num_reasoning_passes"] if student_outputs is not None else 0,
                device=self.device,
                dtype=torch.float32,
            ),
            "decoded_cot_texts": decoded_cot_texts,
            **per_slot_decoder_losses,
        }

    def _build_batch_inputs(
        self,
        examples: list[dict[str, Any]],
        *,
        teacher_mode: bool,
    ) -> dict[str, Any]:
        images = [to_pil_preserve(example["image"]) for example in examples]
        if teacher_mode:
            field_names = self.field_names
            visible_texts = [
                build_visible_cot_text(field_names, example.get("cot_fields_raw", {}))
                for example in examples
            ]
            messages = [
                build_teacher_message(images[idx], examples[idx]["lang"], visible_texts[idx], self.config.datasets.vla_data)
                for idx in range(len(examples))
            ]
        else:
            messages = [
                build_student_message(
                    images[idx],
                    examples[idx]["lang"],
                    self.config.datasets.vla_data,
                    field_count=len(self.field_names),
                    token_spec=self.token_spec,
                )
                for idx in range(len(examples))
            ]

        limit_key = "teacher_max_prompt_length" if teacher_mode else "student_max_prompt_length"
        limit = int(_cfg_get(self.cot_cfg, limit_key, 0) or 0)
        if limit <= 0:
            raise ValueError(f"cot.{limit_key} must be a positive integer.")
        batch_inputs = self._apply_chat_template_checked(
            messages,
            add_generation_prompt=not teacher_mode,
            limit=limit,
            label="Teacher" if teacher_mode else "Student",
        )
        return self._move_qwen_inputs(batch_inputs)

    def _build_prefix_only_inputs(
        self,
        examples: list[dict[str, Any]],
        *,
        limit_key: str = "teacher_max_prompt_length",
        label: str = "TeacherPrefix",
    ) -> dict[str, Any]:
        images = [to_pil_preserve(example["image"]) for example in examples]
        messages = [
            build_prefix_message(images[idx], examples[idx]["lang"], self.config.datasets.vla_data)
            for idx in range(len(examples))
        ]
        limit = int(_cfg_get(self.cot_cfg, limit_key, 0) or 0)
        if limit <= 0:
            raise ValueError(f"cot.{limit_key} must be a positive integer.")
        batch_inputs = self._apply_chat_template_checked(
            messages,
            add_generation_prompt=True,
            limit=limit,
            label=label,
        )
        return self._move_qwen_inputs(batch_inputs)

    def _validate_prompt_lengths(
        self,
        attention_mask: torch.Tensor,
        *,
        limit: int,
        label: str,
    ) -> None:
        lengths = attention_mask.sum(dim=1)
        too_long = torch.nonzero(lengths > limit, as_tuple=False)
        if too_long.numel() == 0:
            return

        bad_index = int(too_long[0].item())
        bad_length = int(lengths[bad_index].item())
        raise ValueError(
            f"{label} prompt length {bad_length} exceeds {limit}. "
            "Increase the configured max prompt length; truncation is disabled."
        )

    def _apply_chat_template_checked(
        self,
        messages: list[list[dict[str, Any]]],
        *,
        add_generation_prompt: bool,
        limit: int,
        label: str,
    ) -> dict[str, torch.Tensor]:
        if limit <= 0:
            raise ValueError("Chat template max length must be positive.")
        batch_inputs = dict(
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
            )
        )
        self._validate_prompt_lengths(
            batch_inputs["attention_mask"],
            limit=limit,
            label=label,
        )
        return batch_inputs

    @torch.inference_mode()
    def predict_action(self, examples: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        if type(examples) is not list:
            examples = [examples]
        if self.eval_mode == "teacher":
            return self.predict_action_teacher_only(examples=examples, **kwargs)
        decode_cot = bool(kwargs.get("decode_cot", self.decode_cot_in_inference))
        if decode_cot and self.decoder_language_model is None:
            self._build_text_decoder()
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        
        #构建输入token，包括添加thinking token
        qwen_inputs = self._build_batch_inputs(
            [{"image": batch_images[idx], "lang": examples[idx]["lang"]} for idx in range(len(examples))],
            teacher_mode=False,
        )
        valid_mask = qwen_inputs["attention_mask"].bool()
        thinking_mask = (qwen_inputs["input_ids"] == self.thinking_token_id) & valid_mask
        latent_hidden_positions = thinking_mask.nonzero(as_tuple=False)
        if latent_hidden_positions.numel() == 0:
            raise ValueError("Student inference inputs do not contain any thinking tokens.")
        thinking_positions = []
        for batch_idx in range(qwen_inputs["input_ids"].shape[0]):
            positions = latent_hidden_positions[latent_hidden_positions[:, 0] == batch_idx][:, 1]
            if positions.numel() != len(self.field_names):
                raise ValueError(
                    f"Expected {len(self.field_names)} thinking tokens for sample {batch_idx}, got {positions.numel()}."
                )
            thinking_positions.append(positions)
        thinking_positions = torch.stack(thinking_positions, dim=0).to(qwen_inputs["input_ids"].device)
        outputs = self._forward_latent(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            thinking_positions=thinking_positions,
            pixel_values=qwen_inputs.get("pixel_values"),
            image_grid_thw=qwen_inputs.get("image_grid_thw"),
        )
        hidden_states = outputs["hidden_states"]
        state = None
        if "state" in examples[0]:
            state = torch.from_numpy(np.array([example["state"] for example in examples])).to(
                hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if state.ndim == 2:
                state = state.unsqueeze(1)
        pred_actions = self._predict_action_from_hidden(hidden_states, qwen_inputs["attention_mask"], state)

        result = {
            "normalized_actions": pred_actions.detach().to(dtype=torch.float32).cpu().numpy(),
            "cot_field_names": list(self.field_names),
            "num_reasoning_passes": outputs["num_reasoning_passes"],
        }
        if decode_cot:
            hidden_size = hidden_states.shape[-1]
            latent_hidden_tensor = torch.gather(
                hidden_states, 1,
                thinking_positions.unsqueeze(-1).expand(-1, -1, hidden_size),
            )
            decoded_cot_by_field: list[dict[str, str]] = []
            for slot_idx, field_name in enumerate(self.field_names):
                slot_texts = self._generate_from_latent(
                    latent_hidden_tensor[:, slot_idx, :],
                    max_new_tokens=int(kwargs.get("max_cot_tokens", self.latent_decode_max_tokens)),
                )
                for sample_idx, text in enumerate(slot_texts):
                    if slot_idx == 0:
                        decoded_cot_by_field.append({})
                    decoded_cot_by_field[sample_idx][field_name] = text
            result["decoded_cot_by_field"] = decoded_cot_by_field
        return result
    
    @torch.inference_mode()
    def predict_action_teacher_only(self, examples: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        if type(examples) is not list:
            examples = [examples]

        decode_cot_text = bool(kwargs.get("decode_cot_text", False))

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        inference_examples = [
            {"image": batch_images[idx], "lang": examples[idx]["lang"]}
            for idx in range(len(examples))
        ]

        prefix_inputs = self._build_prefix_only_inputs(inference_examples)

        generated_ids = self.qwen_vl_interface.generate(
            **prefix_inputs,
            max_new_tokens=int(kwargs.get("max_new_tokens", self.latent_decode_max_tokens)),
        )

        prompt_len = prefix_inputs["input_ids"].shape[1]
        teacher_token_ids = generated_ids[:, prompt_len:]

        teacher_texts = None
        if decode_cot_text:
            teacher_texts = self.tokenizer.batch_decode(teacher_token_ids, skip_special_tokens=True)

        prefix_mask = prefix_inputs["attention_mask"].to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            gen_mask = teacher_token_ids.ne(pad_token_id).to(dtype=prefix_mask.dtype, device=self.device)
        else:
            gen_mask = torch.ones(
                teacher_token_ids.shape[:2], dtype=prefix_mask.dtype, device=self.device
            )
        full_attention_mask = torch.cat([prefix_mask, gen_mask], dim=1)

        full_input_ids = torch.cat([prefix_inputs["input_ids"], teacher_token_ids], dim=1)
        backbone_outputs = self.qwen_vl_interface.model.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=prefix_inputs.get("pixel_values"),
            image_grid_thw=prefix_inputs.get("image_grid_thw"),
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state

        state = None
        if "state" in examples[0]:
            state = torch.from_numpy(np.array([example["state"] for example in examples])).to(
                hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if state.ndim == 2:
                state = state.unsqueeze(1)
        pred_actions = self._predict_action_from_hidden(hidden_states, full_attention_mask, state)

        result = {
            "normalized_actions": pred_actions.detach().to(dtype=torch.float32).cpu().numpy(),
            "teacher_token_ids": teacher_token_ids.detach().cpu().numpy(),
        }
        if decode_cot_text:
            result["teacher_texts"] = teacher_texts
            result["teacher_cot_texts"] = [text.strip() for text in teacher_texts]

        torch.cuda.empty_cache()
        return result

    def load_state_dict(self, state_dict, strict: bool = True):
        decoder_keys = tuple(key for key in state_dict if key.startswith(("decoder_language_model.", "decoder_lm_head.", "decoder_projection.")))
        if decoder_keys and self.decoder_language_model is None:
            if strict:
                logger.warning(
                    "Ignoring decoder keys because the CoT decoder is not loaded. "
                    "Set cot.enable_decoder_loss=true or cot.decode_cot_in_inference=true if decoder weights are needed."
                )
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith(("decoder_language_model.", "decoder_lm_head.", "decoder_projection."))
            }
            strict = False
        if not decoder_keys and self.decoder_language_model is not None:
            if strict:
                logger.warning("Decoder is loaded, but checkpoint has no decoder keys; loading checkpoint with strict=False.")
            strict = False
        return super().load_state_dict(state_dict, strict=strict)
