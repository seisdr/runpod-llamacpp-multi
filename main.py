#!/usr/bin/env python3
"""RunPod llama.cpp worker (serverless + introspection UI)."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import runpod

import config as cfg_mod
from handler import llama_health, llama_list_models, handler

VOLUME = Path(os.environ.get("RUNPOD_VOLUME", "/workspace"))

cfg = cfg_mod.RuntimeConfig.load()
server = cfg_mod.ServerProcess(cfg)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def wait_for_server(timeout: int = int(os.environ.get("STARTUP_TIMEOUT", "180"))) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        h = llama_health()
        if h.get("ok"):
            print("llama-server ready.")
            return
        last_err = h.get("error", "...")
        time.sleep(1)
    raise RuntimeError(f"llama-server did not start: {last_err}")


def preload() -> None:
    models_env = os.environ.get("LLAMA_MODELS", "").strip()
    if not models_env:
        return
    from handler import pull_model
    for m in [x.strip() for x in models_env.split(",") if x.strip()]:
        try:
            r = pull_model(m)
            print(f"preload {m}: {r.get('status') or r.get('error')}")
        except Exception as e:  # noqa: BLE001
            print(f"preload {m}: {e}")


def run_ui_in_thread() -> None:
    import uvicorn
    from web import create_app
    app = create_app()

    def serve():
        try:
            uvicorn.run(app, host=os.environ.get("UI_HOST", "0.0.0.0"),
                        port=int(os.environ.get("UI_PORT", "8000")), log_level="info")
        except Exception as e:  # noqa: BLE001
            logging.getLogger("ui").exception("uvicorn failed")

    t = threading.Thread(target=serve, daemon=True, name="uvicorn")
    t.start()
    print(f"UI on http://{os.environ.get('UI_HOST','0.0.0.0')}:{os.environ.get('UI_PORT','8000')}")


def main() -> None:
    import web
    web.server_handle = server

    server.start()
    run_ui_in_thread()

    try:
        wait_for_server()
        preload()
        print("Ready.")
        runpod.serverless.start({"handler": handler})
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
