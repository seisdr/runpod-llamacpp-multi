#!/usr/bin/env python3
"""RunPod entrypoint: llama-server lifecycle + background FastAPI UI server."""

from __future__ import annotations

import logging
import os
import threading
import time

import runpod

import config as cfg_mod
from handler import handler, llama_health
from web import create_app

cfg = cfg_mod.RuntimeConfig.load()
server = cfg_mod.ServerProcess(cfg)


def wait_for_server(timeout: int = int(os.environ.get("STARTUP_TIMEOUT", "180"))) -> None:
    print(f"Waiting for llama-server at http://{cfg.llama_host}:{cfg.llama_port} ...")
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        h = llama_health()
        if h.get("ok"):
            print("llama-server is ready.")
            return
        last_err = h.get("error", "...")
        time.sleep(1)
    raise RuntimeError(f"llama-server did not start: {last_err}")


def preload() -> None:
    models_env = os.environ.get("LLAMA_MODELS", "").strip()
    if not models_env:
        print("No LLAMA_MODELS — models start via web UI / method=pull / first request.")
        return
    from handler import pull_model
    for m in [x.strip() for x in models_env.split(",") if x.strip()]:
        print(f"Preloading: {m}")
        try:
            r = pull_model(m)
            print(f"  preload {m}: {r.get('status') or r.get('error')}")
        except Exception as e:  # noqa: BLE001
            print(f"  preload {m} failed: {e}")


def run_ui_in_thread() -> None:
    """Run uvicorn in a daemon thread so runpod.serverless.start can own main."""
    import uvicorn

    app = create_app()
    ui_host = os.environ.get("UI_HOST", "0.0.0.0")
    ui_port = int(os.environ.get("UI_PORT", "8000"))

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    def serve() -> None:
        uvicorn.run(app, host=ui_host, port=ui_port, log_level="warning")

    t = threading.Thread(target=serve, daemon=True, name="uvicorn")
    t.start()
    print(f"Web UI on http://{ui_host}:{ui_port}")


def main() -> None:
    import web
    web.server_handle = server

    server.start()
    run_ui_in_thread()
    try:
        wait_for_server()
        preload()
        print("Ready - RunPod jobs + web UI active.")
        runpod.serverless.start({"handler": handler})
    finally:
        server.stop()


if __name__ == "__main__":
    main()
