"""Runtime configuration + llama-server process lifecycle.

Settings persist to the RunPod volume so a llama-server restart (to apply
new args) does not require a RunPod container restart. Models stay cached on
disk, so restart is fast.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_VOLUME = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", DEFAULT_VOLUME / "config"))
CONFIG_FILE = CONFIG_DIR / "runtime.json"

DEFAULTS: dict[str, Any] = {
    "n_gpu_layers": 999,
    "ctx_size": 8192,
    "parallel": 1,
    "flash_attn": "auto",
    "models_max": 1,
    "extra_args": "",
    "default_quant": "Q4_K_M",
    "hf_pull_mode": "auto",
    "llama_host": "127.0.0.1",
    "llama_port": 8080,
}


@dataclass
class RuntimeConfig:
    n_gpu_layers: int = 999
    ctx_size: int = 8192
    parallel: int = 1
    flash_attn: str = "auto"
    models_max: int = 1
    extra_args: str = ""
    default_quant: str = "Q4_K_M"
    hf_pull_mode: str = "auto"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080

    # ---- persistence ----

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(CONFIG_FILE)

    @classmethod
    def load(cls) -> "RuntimeConfig":
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                known = {k: v for k, v in data.items() if k in DEFAULTS}
                return cls(**known)
            except Exception as e:  # noqa: BLE001
                print(f"config load failed: {e}")
        c = cls()
        c.save()
        return c

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ServerProcess:
    """Owns the llama-server subprocess; restart applies new config."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.proc: subprocess.Popen | None = None
        self.lock = threading.RLock()
        self.started_at: float = 0.0
        self.restart_count: int = 0
        self._last_error: str = ""

    # ---- binary ----

    def _binary(self) -> str:
        b = os.environ.get("LLAMA_SERVER_BIN", "/app/llama-server")
        return b if os.path.isfile(b) else "llama-server"

    def _cache_dir(self) -> Path:
        return Path(os.environ.get("LLAMA_CACHE", DEFAULT_VOLUME / "llama-cache"))

    def _models_dir(self) -> Path:
        return Path(os.environ.get("MODELS_DIR", DEFAULT_VOLUME / "models"))

    # ---- command line ----

    def cmdline(self) -> list[str]:
        c = self.config
        cmd = [
            self._binary(),
            "--host", c.llama_host,
            "--port", str(c.llama_port),
            "--models-dir", str(self._models_dir()),
            "--models-max", str(c.models_max),
            "--n-gpu-layers", str(c.n_gpu_layers),
            "--ctx-size", str(c.ctx_size),
            "--parallel", str(c.parallel),
            "--flash-attn", c.flash_attn,
            "--jinja",
            "--no-webui",
        ]
        if c.extra_args:
            cmd.extend(c.extra_args.split())
        return cmd

    # ---- lifecycle ----

    def start(self) -> None:
        with self.lock:
            self._start_unsafe()

    def _start_unsafe(self) -> None:
        self._models_dir().mkdir(parents=True, exist_ok=True)
        self._cache_dir().mkdir(parents=True, exist_ok=True)
        os.environ["LLAMA_CACHE"] = str(self._cache_dir())
        cmd = self.cmdline()
        print("Starting llama-server:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)
        self.started_at = time.time()
        self.restart_count += 1

    def stop(self, timeout: int = 15) -> None:
        with self.lock:
            p = self.proc
            if p is None or p.poll() is not None:
                self.proc = None
                return
            try:
                p.terminate()
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)
            self.proc = None

    def restart(self, new_config: RuntimeConfig | None = None) -> None:
        with self.lock:
            self.stop()
            if new_config is not None:
                self.config = new_config
            self._start_unsafe()

    # ---- status ----

    @property
    def pid(self) -> int | None:
        with self.lock:
            return self.proc.pid if self.proc and self.proc.poll() is None else None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    @property
    def is_alive(self) -> bool:
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    @property
    def last_error(self) -> str:
        return self._last_error

    def status_dict(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive,
            "pid": self.pid,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "restart_count": self.restart_count,
            "cmdline": self.cmdline(),
            "last_error": self._last_error,
            "config": self.config.to_dict(),
        }


# ---- GPU info ----


def gpu_info() -> dict[str, Any]:
    """Try pynvml then nvidia-smi fallback."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            devices.append({
                "index": i,
                "name": name,
                "memory_total_mb": mem.total // (1024 * 1024),
                "memory_used_mb": mem.used // (1024 * 1024),
                "memory_free_mb": mem.free // (1024 * 1024),
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "temperature_c": temp,
            })
        pynvml.nvmlShutdown()
        return {"ok": True, "backend": "pynvml", "count": n, "devices": devices}
    except Exception:
        pass

    # nvidia-smi fallback
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        devices = []
        if out.returncode == 0:
            for i, line in enumerate(out.stdout.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    devices.append({
                        "index": i,
                        "name": parts[0],
                        "memory_total_mb": int(float(parts[1])),
                        "memory_used_mb": int(float(parts[2])),
                        "memory_free_mb": int(float(parts[3])),
                        "gpu_utilization": int(parts[4]),
                        "memory_utilization": int(parts[5]),
                        "temperature_c": int(parts[6]) if parts[6] != "N/A" else None,
                    })
        return {"ok": True, "backend": "nvidia-smi", "count": len(devices), "devices": devices}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ---- disk / cache info ----


def cache_info() -> dict[str, Any]:
    def dir_stats(p: Path) -> dict[str, Any]:
        if not p.exists():
            return {"path": str(p), "exists": False, "files": 0, "size_mb": 0}
        files = list(p.rglob("*")) if p.is_dir() else [p]
        ggufs = [f for f in files if f.is_file() and f.suffix.lower() == ".gguf"]
        total = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "path": str(p),
            "exists": True,
            "files": sum(1 for f in files if f.is_file()),
            "ggufs": len(ggufs),
            "size_mb": round(total / 1024 / 1024, 1),
            "gguf_sizes_mb": {
                str(f.relative_to(p)): round(f.stat().st_size / 1024 / 1024, 1)
                for f in ggufs
            },
        }

    return {
        "llama_cache": dir_stats(Path(os.environ.get("LLAMA_CACHE", DEFAULT_VOLUME / "llama-cache"))),
        "models_dir": dir_stats(Path(os.environ.get("MODELS_DIR", DEFAULT_VOLUME / "models"))),
        "config_dir": dir_stats(CONFIG_DIR),
    }
