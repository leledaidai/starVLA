from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IMPLICIT_COT_TAGS = [
    ("TASK:", "task"),
    ("PLAN:", "plan"),
    ("VISIBLE OBJECTS:", "bboxes"),
    ("SUBTASK REASONING:", "subtask_reason"),
    ("SUBTASK:", "subtask"),
    ("MOVE REASONING:", "move_reason"),
    ("MOVE:", "move"),
    ("GRIPPER POSITION:", "gripper"),
]


@dataclass(frozen=True)
class CotRecord:
    episode_index: int
    frame_index: int
    has_cot: bool
    reasoning_text: str
    task: str = ""
    plan: str = ""
    bboxes: str = ""
    subtask_reason: str = ""
    subtask: str = ""
    move_reason: str = ""
    move: str = ""
    gripper: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "has_cot": self.has_cot,
            "reasoning_text": self.reasoning_text,
            "task": self.task,
            "plan": self.plan,
            "bboxes": self.bboxes,
            "subtask_reason": self.subtask_reason,
            "subtask": self.subtask,
            "move_reason": self.move_reason,
            "move": self.move,
            "gripper": self.gripper,
        }


def format_reasoning_dict_to_string(reasoning_dict: dict[str, str]) -> str:
    parts: list[str] = []
    for tag, key in IMPLICIT_COT_TAGS:
        value = reasoning_dict.get(key, "")
        if value:
            parts.append(f"{tag} {value}")
    return " ".join(parts)


def augment_reasoning_dict(episode_data: dict[str, Any], frame_idx: int, reasoning_dict: dict[str, Any]) -> dict[str, str]:
    augmented = {str(key): _normalize_text(value) for key, value in reasoning_dict.items()}

    features = episode_data.get("features", {})
    gripper_positions = features.get("gripper_position")
    if gripper_positions is not None and frame_idx < len(gripper_positions):
        future_positions: list[Any] = []
        for offset in range(5):
            pos_idx = frame_idx + offset
            if pos_idx < len(gripper_positions):
                future_positions.extend(gripper_positions[pos_idx])
            elif future_positions:
                future_positions.extend(future_positions[-2:])
        augmented["gripper"] = str(future_positions)
    else:
        augmented.setdefault("gripper", "")

    bboxes = features.get("bboxes")
    if bboxes is not None and frame_idx < len(bboxes):
        boxes = bboxes[frame_idx]
        if boxes:
            augmented["bboxes"] = ", ".join(f"{name} {box}" for _, name, box in boxes)
        else:
            augmented.setdefault("bboxes", "")
    else:
        augmented.setdefault("bboxes", "")

    for _, key in IMPLICIT_COT_TAGS:
        augmented.setdefault(key, "")
    return augmented


def build_cot_record(
    *,
    episode_index: int,
    frame_index: int,
    episode_data: dict[str, Any] | None,
) -> CotRecord:
    if not episode_data:
        return CotRecord(episode_index=episode_index, frame_index=frame_index, has_cot=False, reasoning_text="")

    reasoning_by_frame = episode_data.get("reasoning") or {}
    raw_reasoning = reasoning_by_frame.get(str(frame_index))
    if not raw_reasoning:
        return CotRecord(episode_index=episode_index, frame_index=frame_index, has_cot=False, reasoning_text="")

    augmented = augment_reasoning_dict(episode_data, frame_index, raw_reasoning)
    reasoning_text = format_reasoning_dict_to_string(augmented)
    return CotRecord(
        episode_index=episode_index,
        frame_index=frame_index,
        has_cot=bool(reasoning_text),
        reasoning_text=reasoning_text,
        task=augmented["task"],
        plan=augmented["plan"],
        bboxes=augmented["bboxes"],
        subtask_reason=augmented["subtask_reason"],
        subtask=augmented["subtask"],
        move_reason=augmented["move_reason"],
        move=augmented["move"],
        gripper=augmented["gripper"],
    )


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

