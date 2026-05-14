#!/usr/bin/env python3
"""
Test whether the teacher_only trained model generates <|im_end|> (EOS)
at the end of CoT text, or keeps generating until max_new_tokens cutoff.

Usage:
    python test_cot_eos.py \
      --ckpt results/Checkpoints/teacher_only_cot_bridge_train_cot_qwen3vl4b \
      --max-cot-tokens 435 \
      --num-samples 5
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check if teacher_only model learns to emit EOS for CoT.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to run directory or .pt checkpoint.")
    parser.add_argument("--max-cot-tokens", type=int, default=435)
    parser.add_argument("--num-samples", type=int, default=5)
    return parser


def make_dummy_image(model) -> Image.Image:
    obs_size = getattr(model.config.framework, "obs_image_size", None)
    h, w = obs_size if obs_size else (224, 224)
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def run_single_test(
    model,
    image: Image.Image,
    instruction: str,
    max_new_tokens: int,
    sample_idx: int,
) -> dict[str, Any]:
    pil_image = image.convert("RGB") if hasattr(image, "convert") else image
    inference_examples = [{"image": [pil_image], "lang": instruction}]

    prefix_inputs = model._build_prefix_only_inputs(inference_examples)
    prompt_len = prefix_inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    generated_ids = model.qwen_vl_interface.generate(
        **prefix_inputs,
        max_new_tokens=max_new_tokens,
    )
    gen_time = time.perf_counter() - t0

    teacher_token_ids = generated_ids[:, prompt_len:]
    num_generated = int(teacher_token_ids.shape[1])

    eos_id = model.tokenizer.eos_token_id
    eos_positions = (teacher_token_ids[0] == eos_id).nonzero(as_tuple=True)[0]
    stopped_at_eos = len(eos_positions) > 0
    first_eos_pos = int(eos_positions[0].item()) if stopped_at_eos else -1

    cot_text = model.tokenizer.decode(teacher_token_ids[0], skip_special_tokens=False)

    return {
        "sample_idx": sample_idx,
        "instruction": instruction,
        "prompt_len": prompt_len,
        "num_generated": num_generated,
        "eos_token_id": eos_id,
        "stopped_at_eos": stopped_at_eos,
        "first_eos_pos": first_eos_pos,
        "eos_count": len(eos_positions),
        "tokens_after_eos": num_generated - first_eos_pos - 1 if stopped_at_eos else 0,
        "gen_time_sec": round(gen_time, 2),
        "cot_text_preview": cot_text[:500],
        "cot_text_full": cot_text,
    }


def main():
    args = build_argparser().parse_args()

    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_files = list(ckpt_path.glob("checkpoints/steps_*_pytorch_model.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint .pt files found under {ckpt_path}")
        ckpt_file = sorted(ckpt_files)[0]  # earliest step
        print(f"[*] Using checkpoint: {ckpt_file}")
    else:
        ckpt_file = ckpt_path

    print("[*] Loading model via from_pretrained ...")
    model = baseframework.from_pretrained(str(ckpt_file))
    model = model.to("cuda").eval()

    eos_id = model.tokenizer.eos_token_id
    eos_str = model.tokenizer.decode([eos_id]) if eos_id is not None else "N/A"
    print(f"[*] Model eval_mode: {getattr(model, 'eval_mode', 'N/A')}")
    print(f"[*] EOS token id: {eos_id}, EOS token str: {eos_str!r}")
    print(f"[*] Model enables: teacher_cot={getattr(model, 'enable_teacher_cot_loss', 'N/A')}, "
          f"teacher_action={getattr(model, 'enable_teacher_action_loss', 'N/A')}, "
          f"student_action={getattr(model, 'enable_student_action_loss', 'N/A')}")
    print(f"[*] max_cot_length config: {model.latent_decode_max_tokens}")
    print()

    test_instructions = [
        "pick up the red block and place it on the blue block",
        "open the drawer",
        "put the carrot on the plate",
        "move the spoon next to the bowl",
        "stack the green block on the yellow block",
    ]

    results = []
    for i in range(min(args.num_samples, len(test_instructions))):
        instruction = test_instructions[i]
        image = make_dummy_image(model)
        print(f"[{i+1}/{args.num_samples}] Instruction: {instruction}")
        result = run_single_test(model, image, instruction, args.max_cot_tokens, i)
        results.append(result)
        status = "✓ EOS FOUND" if result["stopped_at_eos"] else "✗ NO EOS (hit max_tokens!)"
        print(f"    Status: {status}")
        print(f"    Generated tokens: {result['num_generated']}, EOS at pos: {result['first_eos_pos']}")
        print(f"    Tokens after EOS: {result['tokens_after_eos']}")
        print(f"    Generation time: {result['gen_time_sec']}s")
        print(f"    CoT : {result['cot_text_full']}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    stopped = [r for r in results if r["stopped_at_eos"]]
    no_eos = [r for r in results if not r["stopped_at_eos"]]
    print(f"Total samples: {len(results)}")
    print(f"Stopped at EOS: {len(stopped)}/{len(results)} ({100*len(stopped)/len(results):.0f}%)")
    print(f"No EOS (max cutoff): {len(no_eos)}/{len(results)} ({100*len(no_eos)/len(results):.0f}%)")
    if stopped:
        avg_eos_pos = np.mean([r["first_eos_pos"] for r in stopped])
        avg_time = np.mean([r["gen_time_sec"] for r in stopped])
        print(f"Average EOS position: {avg_eos_pos:.0f} tokens")
        print(f"Average generation time: {avg_time:.2f}s")
    if no_eos:
        avg_time_no_eos = np.mean([r["gen_time_sec"] for r in no_eos])
        print(f"Average generation time (no EOS): {avg_time_no_eos:.2f}s")
        wasted_time = avg_time_no_eos - (avg_time if stopped else 0)
        print(f"Wasted time per sample without EOS: ~{wasted_time:.2f}s")

    if any(not r["stopped_at_eos"] for r in results):
        print()
        print("*** WARNING: Model does NOT reliably generate <|im_end|> for CoT text! ***")
        print("    This means generation runs to max_cot_length every time, wasting compute")
        print("    and making inference much slower than necessary.")
        print("    Fix: add <|im_end|> to CoT training labels in cot_dataset.py so the")
        print("    model learns to emit EOS at the end of CoT text.")

    # Print full CoT text for the first sample
    print()
    print("=" * 80)
    print("FULL CoT TEXT (sample 0)")
    print("=" * 80)
    print(results[0]["cot_text_full"])


if __name__ == "__main__":
    main()
