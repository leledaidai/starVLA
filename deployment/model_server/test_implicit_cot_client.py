#!/usr/bin/env python3
"""
Smoke-test client for :class:`ImplicitCoTPolicyServer`.

Connect to a running server, send a synthetic observation, and verify
end-to-end transport + inference for both latent and teacher modes.

Usage::

    # Test latent mode (default)
    python deployment/model_server/test_implicit_cot_client.py --port 10093

    # Test teacher mode
    python deployment/model_server/test_implicit_cot_client.py --port 10093 --mode teacher

    # Test latent mode with CoT decode
    python deployment/model_server/test_implicit_cot_client.py --port 10093 --decode_cot

    # Use a real image file
    python deployment/model_server/test_implicit_cot_client.py --image /path/to/obs.png \\
        --instruction "pick up the red block"
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test client for ImplicitCoT policy server")
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname/IP")
    parser.add_argument("--port", type=int, default=10093, help="Server port")
    parser.add_argument("--mode", default="latent", choices=["latent", "teacher"], help="Inference mode")
    parser.add_argument("--decode_cot", action="store_true", help="Request CoT text decoding")
    parser.add_argument("--image", default=None, help="Path to a real image file (optional)")
    parser.add_argument("--instruction", default="pick up the red block", help="Task instruction")
    parser.add_argument("--max_cot_tokens", type=int, default=128, help="Max CoT tokens (latent per-field / teacher total)")
    return parser


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    # -- connect ----------------------------------------------------------------
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    logging.info("Connected. Server metadata: %s", meta)

    # -- build observation ------------------------------------------------------
    if args.image is not None:
        from PIL import Image

        img_np = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    else:
        # Synthetic image: random noise, 224×224×3
        rng = np.random.default_rng(42)
        img_np = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)

    state = np.zeros((7,), dtype=np.float32)  # dummy 7-DoF state

    # IMPORTANT: the model expects "image" as a LIST (one entry per camera view),
    # matching the dataloader contract: image: list[PIL.Image] or list[np.ndarray].
    example = {
        "image": [img_np],            # list of (H, W, 3) uint8
        "lang": args.instruction,
        "state": state,               # optional
    }
    payload = {
        "type": "infer",
        "mode": args.mode,
        "payload": {
            "examples": [example],
            "decode_cot": args.decode_cot,
            "max_cot_tokens": args.max_cot_tokens,
            "max_new_tokens": args.max_cot_tokens,
        },
    }

    # -- run inference ----------------------------------------------------------
    t0 = time.time()
    response = client.predict_action(payload)
    elapsed = time.time() - t0

    # -- print results ----------------------------------------------------------
    logging.info("Inference took %.3f s", elapsed)

    if response.get("status") != "ok":
        logging.error("Inference returned error status: %s", response)
        client.close()
        return

    data = response.get("data", {})
    actions = data.get("normalized_actions")
    if actions is not None:
        logging.info("normalized_actions shape: %s", np.asarray(actions).shape)
        logging.info("normalized_actions:\n%s", actions)

    if args.decode_cot:
        if args.mode == "teacher":
            texts = data.get("teacher_cot_texts")
            if texts:
                logging.info("=== Teacher CoT text ===\n%s", texts[0])
        else:
            fields = data.get("decoded_cot_by_field")
            if fields:
                logging.info("=== Decoded latent CoT (by field) ===")
                for fname, text in fields[0].items():
                    logging.info("  [%s] %s", fname, text)

    client.close()
    logging.info("Smoke test done.")


if __name__ == "__main__":
    main()
