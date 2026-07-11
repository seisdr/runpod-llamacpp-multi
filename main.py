#!/usr/bin/env python3
"""RunPod llama.cpp worker with persistent startup logging.

All boot output is mirrored to /workspace/startup.log so we can debug
container crashes that happen before any job is processed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import runpod

import config as cfg_mod
from handler import llama_health, handler

VOLUME = Path(os.environ.get("RUNPOD_VOLUME", "/workspace"))
STARTUP_LOG = VOLUME / "startup.log"

cfg = cfg_mod.RuntimeConfig.load()
server = cfg_mod.ServerProcess(cfg)


class StartupLog:
    """Tee Python logging + stdout/stderr into a file on the volume."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", buffering=1)
        self.orig_out = sys.stdout
        self.orig_err = sys.stderr

    def info(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.fw(line)
        try:
            self.orig_out.write(line + "\n")
            self.orig_out.flush()
        except Exception:
            pass

    def fw(self, msg: str) -> None:
        try:
            self.fh.write(msg + "\n")
            self.fh.flush()
        except Exception:
            pass

    def install(self) -> None:
        self.info("=== startup === " + time.strftime("%Y-%m-%d %H:%M:%S"))
        self.info(f"volume={VOLUME} llama_server_bin server running")

    def flush(self) -> None:
        try:
            self.fh.flush()
        except Exception:
            pass


log = StartupLog(STARTUP_LOG)


def wait_for_server(timeout: int = int(os.environ.get("STARTUP_TIMEOUT", "180"))) -> None:
    log.info(f"waiting for llama-server at http://{cfg.llama_host}:{cfg.llama_port} (timeout={timeout}s)")
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            h = llama_health()
            if h.get("ok"):
                log.info("llama-server health OK")
                return
            last_err = h.get("error", "...")
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(1)
    raise RuntimeError(f"llama-server did not start: {last_err}")


def preload() -> None:
    models_env = os.environ.get("LLAMA_MODELS", "").strip()
    if not models_env:
        log.info("LLAMA_MODELS empty — skipping preload")
        return
    from handler import pull_model
    for m in [x.strip() for x in models_env.split(",") if x.strip()]:
        try:
            r = pull_model(m)
            log.info(f"preload {m}: {r.get('status') or r.get('error')}")
        except Exception as e:  # noqa: BLE001
            log.info(f"preload {m}: {e}")


def run_ui_in_thread() -> None:
    import uvicorn
    from web import create_app
    app = create_app()

    def serve():
        try:
            uvicorn.run(
                app, host=os.environ.get("UI_HOST", "0.0.0.0"),
                port=int(os.environ.get("UI_PORT", "8000")), log_level="info",
            )
        except Exception as e:  # noqa: BLE001
            log.info(f"uvicorn crashed: {e}")

    t = threading.Thread(target=serve, daemon=True, name="uvicorn")
    t.start()
    log.info(f"UI on http://{os.environ.get('UI_HOST','0.0.0.0')}:{os.environ.get('UI_PORT','8000')}")


def main() -> None:
    import web
    web.server_handle = server

    log.install()
    log.info(f"cmdline: {' '.join(server.cmdline())}")

    try:
        server.start()
        log.info(f"llama-server launched pid={server.pid}")
    except Exception as e:  # noqa: BLE001
        log.info(f"failed to launch llama-server: {e}")
        log.flush()
        return

    run_ui_in_thread()

    try:
        log.info("waiting for llama-server to be ready")
        wait_for_server()
        log.info("llama-server ready")
        preload()
        log.info("starting serverless job poller")
        # runpod.serverless.start calls sys.exit on KeyboardInterrupt
        runpod.serverless.start({"handler": handler})
        log.info("runpod.serverless.start returned (unexpected)")
    except KeyboardInterrupt:
        log.info("interrupted")
    except SystemExit as e:
        log.info(f"SystemExit: {e}")
    except Exception as e:  # noqa: BLE001
        log.info(f"startup/job poll crashed: {e!r}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log.info(line)
    finally:
        log.info("stopping llama-server")
        server.stop()
        log.flush()


if __name__ == "__main__":
    main()
