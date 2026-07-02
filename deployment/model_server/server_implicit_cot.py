#!/usr/bin/env python3
"""
Policy server for QwenGR00TImplicitCoT — serves latent CoT VLA over WebSocket.

Two inference modes
--------------------
- ``"latent"`` (default) — Student forward: VLM encodes observation + thinking
  tokens → latent slots → action head.  When *decode_cot=True* the latent slots
  are decoded through the text decoder into human-readable CoT fields.
- ``"teacher"`` — Teacher-forward: VLM autoregressively generates CoT text
  tokens → full hidden states → action head.  When *decode_cot=True* the
  generated token ids are decoded to string.

Launch
------
.. code-block:: bash

    python deployment/model_server/server_implicit_cot.py \\
        --ckpt_path results/Checkpoints/.../checkpoints/steps_50000.pt \\
        --port 10093 \\
        --use_bf16

Client protocol (WebSocket + MessagePack with NumPy)
----------------------------------------------------
Each request is a msgpack-encoded dict::

    {
        "type": "infer",            # "ping" | "infer"
        "mode": "latent",           # "latent" | "teacher"
        "payload": {
            "examples": [           # one dict per sample
                {
                    "image": [<np.ndarray (H, W, 3) uint8>, ...],  # list, one per camera
                    "lang": "pick up the red block",
                    "state": <np.ndarray (state_dim,) float32>,     # optional
                }
            ],
            "decode_cot": true,     # decode CoT text (latent) / return CoT text (teacher)
            "max_cot_tokens": 128,  # max tokens for latent CoT decode (per field)
            "max_new_tokens": 512,  # max tokens for teacher autoregressive generation
        }
    }

Response::

    {
        "status": "ok",
        "type": "inference_result",
        "data": {
            "normalized_actions": <np.ndarray (B, action_horizon, action_dim)>,
            "cot_field_names": [...],                     # latent mode
            "num_reasoning_passes": int,                  # latent mode
            "decoded_cot_by_field": [{field: str, ...}],  # latent + decode_cot
            "teacher_cot_texts": [str, ...],              # teacher + decode_cot
        }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import time
import traceback
from typing import Any

import torch
import websockets.asyncio.server
import websockets.frames

from deployment.model_server.tools import msgpack_numpy
from starVLA.model.framework.base_framework import baseframework


class ImplicitCoTPolicyServer:
    """Thin WebSocket wrapper around a QwenGR00TImplicitCoT model instance.

    The server unpacks client messages and dispatches to
    :meth:`QwenGR00TImplicitCoT.predict_action` (latent) or
    :meth:`QwenGR00TImplicitCoT.predict_action_teacher_only` (teacher).

    Parameters
    ----------
    model:
        An already-loaded, eval-mode QwenGR00TImplicitCoT model on the
        desired device.
    host / port:
        Bind address for the WebSocket server.
    idle_timeout:
        Seconds of inactivity before the server auto-shuts down (≤0 = never).
    """

    def __init__(
        self,
        model,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,
    ) -> None:
        self._model = model
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._last_active: float = time.time()
        self._metadata: dict[str, Any] = {
            "framework": type(model).__name__,
            "fields": getattr(model, "field_names", []),
            "eval_mode": getattr(model, "eval_mode", "unknown"),
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Block the current thread, running the WebSocket event loop."""
        asyncio.run(self._run())

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server) -> None:
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info("Idle timeout reached (%ds) — shutting down.", self._idle_timeout)
                server.close()
                await server.wait_closed()
                break

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logging.info("Connection opened from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        # Protocol: send metadata as the first frame so that
        # WebsocketClientPolicy._wait_for_server can proceed.
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                raw = await websocket.recv()
                self._last_active = time.time()
                msg = msgpack_numpy.unpackb(raw)
                resp = self._dispatch(msg)
                await websocket.send(packer.pack(resp))
            except websockets.ConnectionClosed:
                logging.info("Connection closed from %s", websocket.remote_address)
                break
            except Exception:
                logging.exception("Unhandled error in handler")
                await websocket.send(packer.pack({"status": "error", "traceback": traceback.format_exc()}))
                # Continue serving other messages from this connection instead of
                # killing it immediately — a single bad request shouldn't tear
                # down a live robot session.

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Top-level routing: ``type`` selects the handler."""
        mtype = msg.get("type", "infer")

        if mtype == "ping":
            return {"status": "ok", "type": "ping"}

        if mtype == "infer":
            return self._handle_infer(msg)

        return {
            "status": "error",
            "type": "unknown",
            "error": f"Unsupported message type {mtype!r}. Expected 'ping' or 'infer'.",
        }

    # ------------------------------------------------------------------
    # Inference handlers
    # ------------------------------------------------------------------

    def _handle_infer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Dispatch infer requests by *mode*."""
        mode = msg.get("mode", "latent")
        payload = msg.get("payload", msg)

        examples = payload.get("examples")
        if not examples:
            return {"status": "error", "type": "inference_result", "error": "'examples' is required in payload."}

        try:
            if mode == "teacher":
                data = self._infer_teacher(examples, payload)
            else:
                data = self._infer_latent(examples, payload)
            return {"status": "ok", "type": "inference_result", "data": data}
        except Exception as exc:
            logging.exception("Inference failed (mode=%s)", mode)
            return {"status": "error", "type": "inference_result", "error": str(exc)}

    @torch.inference_mode()
    def _infer_latent(self, examples: list[dict], payload: dict) -> dict:
        """Student / latent-slot inference."""
        decode_cot = bool(payload.get("decode_cot", False))
        max_cot_tokens = int(payload.get("max_cot_tokens", 128))
        return self._model.predict_action(
            examples=examples,
            decode_cot=decode_cot,
            max_cot_tokens=max_cot_tokens,
        )

    @torch.inference_mode()
    def _infer_teacher(self, examples: list[dict], payload: dict) -> dict:
        """Teacher-mode autoregressive CoT → action inference."""
        decode_cot = bool(payload.get("decode_cot", False))
        max_new_tokens = int(payload.get("max_new_tokens", 512))
        return self._model.predict_action_teacher_only(
            examples=examples,
            decode_cot_text=decode_cot,
            max_new_tokens=max_new_tokens,
        )


# ======================================================================
# CLI entry point
# ======================================================================


def main(args: argparse.Namespace) -> None:
    # -- load model via the framework registry -------------------------------
    logging.info("Loading checkpoint: %s", args.ckpt_path)
    model = baseframework.from_pretrained(args.ckpt_path)

    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Server: host=%s ip=%s port=%d", hostname, local_ip, args.port)
    logging.info("CoT fields: %s", model.field_names)
    logging.info("Eval mode (from config): %s", model.eval_mode)

    # -- start server ---------------------------------------------------------
    server = ImplicitCoTPolicyServer(
        model,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
    )
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QwenGR00TImplicitCoT policy server")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--port", type=int, default=10093, help="WebSocket listen port")
    parser.add_argument("--use_bf16", action="store_true", help="Cast model to bfloat16")
    parser.add_argument("--idle_timeout", type=int, default=-1, help="Idle timeout in seconds (-1 = never)")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    main(build_argparser().parse_args())
