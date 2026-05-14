from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


SUPPORTED_COT_FIELDS = (
    "task",
    "plan",
    "bboxes",
    "subtask_reason",
    "subtask",
    "move_reason",
    "move",
    "gripper",
)

FIELD_TO_TAG = {
    "task": "TASK:",
    "plan": "PLAN:",
    "bboxes": "VISIBLE OBJECTS:",
    "subtask_reason": "SUBTASK REASONING:",
    "subtask": "SUBTASK:",
    "move_reason": "MOVE REASONING:",
    "move": "MOVE:",
    "gripper": "GRIPPER POSITION:",
}

_PRINTED_COT_FIELD_ORDER = False


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class CotFieldSpec:
    name: str


@dataclass(frozen=True)
class ThinkingTokenSpec:
    thinking_token: str
    start_token: str
    end_token: str


def parse_cot_fields(cot_cfg: Any) -> list[CotFieldSpec]:
    global _PRINTED_COT_FIELD_ORDER
    raw_fields = _cfg_get(cot_cfg, "fields", []) or []
    requested_names: list[str] = []
    seen: set[str] = set()
    for item in raw_fields:
        if isinstance(item, str):
            name = item
        else:
            name = _cfg_get(item, "name", None)
        if name is None:
            raise ValueError("Each `cot.fields` entry must define `name`.")
        name = str(name).strip()
        if name not in FIELD_TO_TAG:
            raise ValueError(
                f"Unsupported CoT field `{name}`. Supported fields: {list(SUPPORTED_COT_FIELDS)}"
            )
        if name in seen:
            raise ValueError(f"Duplicate CoT field `{name}` is not allowed.")
        seen.add(name)
        requested_names.append(name)

    # Align with Embodied-CoT / OpenPI canonical implicit-CoT slot order, while
    # still allowing users to request only a subset of fields.
    requested_set = set(requested_names)
    fields = [CotFieldSpec(name=name) for name in SUPPORTED_COT_FIELDS if name in requested_set]
    if not _PRINTED_COT_FIELD_ORDER:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        if rank in {"0", ""} and local_rank in {"0", ""}:
            print(f"[CoT] Loaded field order: {[field.name for field in fields]}")
        _PRINTED_COT_FIELD_ORDER = True
    return fields


def format_tagged_field(field_name: str, content: str) -> str:
    content = (content or "").strip()
    if not content:
        return ""
    return f"{FIELD_TO_TAG[field_name]} {content}"


def build_visible_cot_text(field_names: list[str], field_values: dict[str, str]) -> str:
    text, _ = build_visible_cot_text_with_spans(field_names, field_values)
    return text


def build_visible_cot_text_with_spans(
    field_names: list[str],
    field_values: dict[str, str],
) -> tuple[str, dict[str, tuple[int, int]]]:
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for field_name in field_names:
        tagged = format_tagged_field(field_name, field_values.get(field_name, ""))
        if tagged:
            if parts:
                cursor += 1
            start = cursor
            end = start + len(tagged)
            parts.append(tagged)
            spans[field_name] = (start, end)
            cursor = end
    return " ".join(parts), spans


def parse_visible_cot_text(text: str, field_names: list[str]) -> dict[str, str]:
    text = (text or "").strip()
    parsed = {field_name: "" for field_name in field_names}
    if not text:
        return parsed

    matches = []
    selected_fields = set(field_names)
    for field_name, tag in FIELD_TO_TAG.items():
        start = text.find(tag)
        if start >= 0:
            matches.append((start, field_name, tag))

    if not matches:
        return parsed

    matches.sort(key=lambda item: item[0])
    for idx, (start, field_name, tag) in enumerate(matches):
        if field_name not in selected_fields:
            continue
        content_start = start + len(tag)
        content_end = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        parsed[field_name] = text[content_start:content_end].strip()
    return parsed


def build_instruction_text(instruction: str, data_cfg: Any) -> str:
    instruction = instruction or ""
    cot_prompt = _cfg_get(data_cfg, "CoT_prompt", None)
    if cot_prompt:
        return str(cot_prompt).replace("{instruction}", instruction)
    return instruction


def build_user_content(images: list[Any], instruction: str, data_cfg: Any) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": build_instruction_text(instruction, data_cfg)})
    return content


def parse_thinking_tokens(cot_cfg: Any) -> ThinkingTokenSpec:
    return ThinkingTokenSpec(
        thinking_token=str(_cfg_get(cot_cfg, "thinking_token", "<|thinking|>")),
        start_token=str(_cfg_get(cot_cfg, "start_thinking_token", "<|start_of_thinking|>")),
        end_token=str(_cfg_get(cot_cfg, "end_thinking_token", "<|end_of_thinking|>")),
    )


def build_thinking_sequence(field_count: int, token_spec: ThinkingTokenSpec) -> str:
    if field_count <= 0:
        return ""
    body = token_spec.thinking_token * field_count
    return f"{token_spec.start_token}{body}{token_spec.end_token}"


def build_student_instruction_text(instruction: str, data_cfg: Any, field_count: int, token_spec: ThinkingTokenSpec) -> str:
    base_instruction = build_instruction_text(instruction, data_cfg).strip()
    thinking_sequence = build_thinking_sequence(field_count, token_spec)
    if not thinking_sequence:
        return base_instruction
    if base_instruction:
        return f"{base_instruction} {thinking_sequence}"
    return thinking_sequence


def build_prefix_message(images: list[Any], instruction: str, data_cfg: Any) -> list[dict[str, Any]]:
    return [{"role": "user", "content": build_user_content(images, instruction, data_cfg)}]


def build_student_message(
    images: list[Any],
    instruction: str,
    data_cfg: Any,
    field_count: int,
    token_spec: ThinkingTokenSpec,
) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": img} for img in images]
    content.append(
        {
            "type": "text",
            "text": build_student_instruction_text(instruction, data_cfg, field_count=field_count, token_spec=token_spec),
        }
    )
    return [{"role": "user", "content": content}]


def build_teacher_message(
    images: list[Any],
    instruction: str,
    visible_cot_text: str,
    data_cfg: Any,
) -> list[dict[str, Any]]:
    message = build_prefix_message(images, instruction, data_cfg)
    message.append({"role": "assistant", "content": [{"type": "text", "text": visible_cot_text or ""}]})
    return message
