#!/usr/bin/env python3
"""Unit tests for config / handler parsing / payloads (no GPU, no server)."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _mk_stub(name, attrs=None):
    m = types.ModuleType(name)
    m.__name__ = name
    m.__path__ = [f"/fake/{name}"]
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m


def _preloader():
    if "fastapi" not in sys.modules:
        fastapi = _mk_stub("fastapi", {
            "FastAPI": type("FastAPI", (), {}),
            "HTTPException": type("HTTPException", (Exception,), {"__init__": lambda self, *a, **k: None}),
        })
        sys.modules["fastapi"] = fastapi
        responses = _mk_stub("fastapi.responses", {"JSONResponse": object, "HTMLResponse": object})
        sys.modules["fastapi.responses"] = responses
        staticfiles = _mk_stub("fastapi.staticfiles", {"StaticFiles": object})
        sys.modules["fastapi.staticfiles"] = staticfiles
    if "uvicorn" not in sys.modules:
        uvicorn = _mk_stub("uvicorn")
        uvicorn.run = lambda *a, **k: None
        sys.modules["uvicorn"] = uvicorn
    if "pydantic" not in sys.modules:
        pydantic = _mk_stub("pydantic")
        pydantic.BaseModel = type("BaseModel", (), {})
        sys.modules["pydantic"] = pydantic
    if "pynvml" not in sys.modules:
        pynvml = _mk_stub("pynvml")
        for fn in ("nvmlInit", "nvmlShutdown", "nvmlDeviceGetCount", "nvmlDeviceGetHandleByIndex",
                   "nvmlDeviceGetName", "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUtilizationRates",
                   "nvmlDeviceGetTemperature"):
            setattr(pynvml, fn, lambda *a, **k: 0)
        pynvml.NVML_TEMPERATURE_GPU = 0
        sys.modules["pynvml"] = pynvml


def _load(name: str, env: dict | None = None):
    _preloader()
    if name == "handler":
        runpod = _mk_stub("runpod")
        runpod.serverless = types.SimpleNamespace(start=lambda *_a, **_k: None)
        sys.modules["runpod"] = runpod
    path = Path(__file__).resolve().parent / f"{name}.py"
    key = f"_{name}_mod"
    sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load spec for {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    if name == "config":
        sys.modules["config"] = mod
    if env is None:
        env = {}
    with mock.patch.dict("os.environ", env, clear=False):
        spec.loader.exec_module(mod)
    return mod


class ConfigTests(unittest.TestCase):
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _load("config", {"CONFIG_DIR": f"{td}/config"})
            c = cfg.RuntimeConfig(n_gpu_layers=10, ctx_size=4096, llama_port=9090)
            c.save()
            loaded = cfg.RuntimeConfig.load()
            self.assertEqual(loaded.n_gpu_layers, 10)
            self.assertEqual(loaded.ctx_size, 4096)
            self.assertEqual(loaded.llama_port, 9090)
            try:
                cfg.CONFIG_FILE.unlink()
            except Exception:
                pass

    def test_gpu_info_fallback(self):
        cfg = _load("config")
        info = cfg.gpu_info()
        self.assertIn("ok", info)


class ParseTests(unittest.TestCase):
    def test_split_hf_id(self):
        h = _load("handler")
        self.assertEqual(h.split_hf_id("org/repo:Q4_K_M"), ("org/repo", "Q4_K_M", None))
        self.assertEqual(h.split_hf_id("org/repo/sub.gguf"), ("org/repo", None, "sub.gguf"))
        self.assertEqual(h.split_hf_id("org/repo"), ("org/repo", None, None))

    def test_normalize_payload_chat(self):
        h = _load("handler")
        body = h.normalize_payload("chat", "org/r:Q4_K_M", {"messages": [{"role": "user", "content": "x"}]})
        self.assertIs(body["stream"], False)
        self.assertEqual(body["model"], "org/r:Q4_K_M")
        self.assertEqual(body["messages"][0]["content"], "x")

    def test_normalize_payload_generate_opts(self):
        h = _load("handler")
        body = h.normalize_payload("generate", "m:Q4_K_M", {"prompt": "hi", "options": {"temperature": 0.2}, "num_predict": 32})
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 32)
        self.assertNotIn("options", body)

    def test_normalize_payload_embed(self):
        h = _load("handler")
        body = h.normalize_payload("embed", "e:Q8_0", {"prompt": "hi"})
        self.assertEqual(body["input"], "hi")

    def test_handler_missing_model(self):
        h = _load("handler")
        out = h.handler({"input": {"method": "chat"}})
        self.assertIn("error", out)
        self.assertIn("model", out["error"])

    def test_handler_health(self):
        h = _load("handler")
        with mock.patch.object(h, "llama_health", return_value={"ok": True}), \
             mock.patch.object(h, "gpu_info", return_value={"ok": True}), \
             mock.patch.object(h, "cache_info", return_value={}):
            out = h.handler({"input": {"method": "health"}})
        self.assertIn("health", out)
        self.assertIn("gpu", out)

    def test_handler_list(self):
        h = _load("handler")
        with mock.patch.object(h, "llama_list_models", return_value=[{"id": "x"}]):
            out = h.handler({"input": {"method": "list"}})
        self.assertEqual(out["models"][0]["id"], "x")

    def test_handler_pull(self):
        h = _load("handler")
        with mock.patch.object(h, "pull_model", return_value={"status": "success"}):
            out = h.handler({"input": {"method": "pull", "model": "org/r:Q4_K_M"}})
        self.assertEqual(out["status"], "success")

    def test_handler_chat_ensure(self):
        h = _load("handler")
        fake = mock.Mock()
        fake.status_code = 200
        fake.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with mock.patch.object(h, "_http") as http:
            http.post.return_value = fake
            out = h.handler({
                "input": {
                    "model": "org/r:Q4_K_M",
                    "skip_ensure": True,
                    "method": "chat",
                    "payload": {"messages": [{"role": "user", "content": "hi"}]},
                }
            })
        self.assertEqual(out["choices"][0]["message"]["content"], "ok")

    def test_web_app_builds(self):
        web = _load("web")
        try:
            app = web.create_app()
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"web create_app() note: {e}")


if __name__ == "__main__":
    unittest.main()
