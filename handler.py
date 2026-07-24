#!/usr/bin/env python3
"""Inference handler + introspection API for the web UI.

Separate from the web server so RunPod's runpod.serverless.start can own the
job polling loop while a background FastAPI serves the UI and introspection.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from config import RuntimeConfig, cache_info, gpu_info

LLAMA_BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
DEFAULT_QUANT = os.environ.get("DEFAULT_QUANT", "Q4_K_M")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/runpod-volume/models"))
LLAMA_CACHE = Path(os.environ.get("LLAMA_CACHE", "/runpod-volume/llama-cache"))

# Ollama-style method names → llama-server routes
METHOD_ROUTES = {
    "chat": "/v1/chat/completions",
    "generate": "/v1/completions",
    "completion": "/v1/completions",
    "completions": "/v1/completions",
    "embed": "/v1/embeddings",
    "embedding": "/v1/embeddings",
    "embeddings": "/v1/embeddings",
    "v1/chat/completions": "/v1/chat/completions",
    "v1/completions": "/v1/completions",
    "v1/embeddings": "/v1/embeddings",
    "chat/completions": "/v1/chat/completions",
    "completion_native": "/completion",
    "embedding_native": "/embedding",
}

META_METHODS = {"pull", "list", "tags", "models", "rm", "delete", "show", "health", "cache", "config"}

_http = requests.Session()
_http.headers["Content-Type"] = "application/json"
if HF_TOKEN:
    _http.headers["Authorization"] = f"Bearer {HF_TOKEN}"


# ---------------------------------------------------------------------------
# Model id parsing
# ---------------------------------------------------------------------------

def split_hf_id(model: str) -> tuple[str, str | None, str | None]:
    model = model.strip()
    if model.endswith(".gguf") and "/" in model:
        if model.count("/") >= 2 and ":" not in model.split("/", 2)[2]:
            repo, fname = model.rsplit("/", 1)
            return repo, None, fname
    if ":" in model:
        repo, tag = model.split(":", 1)
        tag = tag.strip()
        if tag.lower().endswith(".gguf"):
            return repo, None, tag
        return repo, tag, None
    return model, None, None


def resolve_quant(model: str) -> str:
    """Ensure model id has a quant tag for llama-server POST /models."""
    model = model.strip()
    if model.startswith(("http://", "https://", "/", "./", "../")):
        return model
    if model.endswith(".gguf") and "/" not in model:
        return model
    if re.match(r"^[^/]+/[^/]+:.+$", model) or (model.count("/") >= 2 and model.endswith(".gg")):
        return model
    if re.match(r"^[^/]+/[^/:]+$", model):
        return f"{model}:{DEFAULT_QUANT}"
    return model


# ---------------------------------------------------------------------------
# Llama server introspection
# ---------------------------------------------------------------------------

def llama_health() -> dict[str, Any]:
    try:
        r = _http.get(f"{LLAMA_BASE}/health", timeout=5)
        return {"ok": r.status_code == 200, "status": r.status_code}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def llama_list_models(reload: bool = False) -> list[dict[str, Any]]:
    params = {"reload": "1"} if reload else None
    r = _http.get(f"{LLAMA_BASE}/models", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = data.get("data") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    return [_normalize_entry(m) for m in items]


def _normalize_entry(m: dict[str, Any]) -> dict[str, Any]:
    status = m.get("status") or {}
    if isinstance(status, dict):
        value = status.get("value")
        failed = status.get("failed", False)
        progress = status.get("progress")
    else:
        value = status
        failed = False
        progress = None
    info = m.get("info") or {}
    return {
        "id": m.get("id") or m.get("model"),
        "path": m.get("path"),
        "status": value,
        "failed": failed,
        "progress": progress,
        "meta": m.get("meta"),
        "info": info,
        "mmproj": info.get("has_mmproj") or info.get("mmproj"),
        "params": m.get("params"),
    }


# ---------------------------------------------------------------------------
# HuggingFace pull
# ---------------------------------------------------------------------------

def hf_list_ggufs(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi  # type: ignore

    api = HfApi(token=HF_TOKEN or None)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    return [f for f in files if f.lower().endswith(".gguf")]


def pick_gguf_file(files: list[str], quant: str | None, explicit: str | None) -> str:
    if explicit:
        if explicit in files:
            return explicit
        for f in files:
            if Path(f).name == explicit:
                return f
        raise FileNotFoundError(f"{explicit} not in repo (have {len(files)} ggufs)")

    if not files:
        raise FileNotFoundError("no .gguf files in Hugging Face repo")

    weights = [f for f in files if "mmproj" not in Path(f).name.lower()] or files
    if quant:
        q = quant.lower().replace("-", "_")
        scored = []
        for f in weights:
            stem = Path(f).stem.lower().replace("-", "_")
            if f"_{q}" in stem:
                scored.append((0, f))
            elif q in stem:
                scored.append((1, f))
        if scored:
            scored.sort()
            return scored[0][1]
        raise FileNotFoundError(
            f"no GGUF matching quant {quant!r}; examples: {[Path(f).name for f in weights[:8]]}"
        )

    dq = DEFAULT_QUANT.lower().replace("-", "_")
    for f in weights:
        if f"_{dq}" in Path(f).stem.lower().replace("-", "_"):
            return f
    return sorted(weights, key=lambda x: len(Path(x).name))[0]


def pull_via_hub(model: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download  # type: ignore

    repo_id, quant, explicit = split_hf_id(model)
    if not re.match(r"^[^/]+/[^/]+$", repo_id):
        return {"error": f"not a Hugging Face repo id: {model}"}

    try:
        files = hf_list_ggufs(repo_id)
    except Exception as e:  # noqa: BLE001
        return {"error": f"failed to list HF repo {repo_id}: {e}"}

    try:
        filename = pick_gguf_file(files, quant, explicit)
    except FileNotFoundError as e:
        return {"error": str(e), "repo": repo_id, "gguf_count": len(files)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            token=HF_TOKEN or None,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"hf_hub_download failed: {e}", "repo": repo_id, "file": filename}

    mmproj = [f for f in files if "mmproj" in Path(f).name.lower() and f.lower().endswith(".gguf")]
    mmproj_path = None
    if mmproj:
        mmproj.sort(key=lambda f: (0 if re.search(r"f16|f32", f, re.I) else 1, f))
        try:
            mmproj_path = hf_hub_download(repo_id=repo_id, filename=mmproj[0], local_dir=str(MODELS_DIR), token=HF_TOKEN or None)
        except Exception:
            pass

    try:
        llama_list_models(reload=True)
    except Exception:
        pass

    return {
        "status": "success",
        "source": "huggingface_hub",
        "model": model,
        "repo": repo_id,
        "file": filename,
        "path": path,
        "local_model_id": Path(filename).name,
        "mmproj": mmproj_path,
    }


def pull_via_router(model: str) -> dict[str, Any]:
    model = resolve_quant(model)
    r = _http.post(f"{LLAMA_BASE}/models", json={"model": model}, timeout=120)
    if r.status_code >= 400:
        try:
            d = r.json()
        except Exception:
            d = r.text
        msg = str(d)
        if "already exists" in msg.lower():
            return {"status": "success", "source": "router", "model": model, "detail": "already_exists"}
        return {"error": f"router pull failed to start (HTTP {r.status_code})", "model": model, "details": d}
    return {"status": "success", "source": "router", "model": model, "detail": "started"}


def pull_model(model: str) -> dict[str, Any]:
    model = resolve_quant(model)
    local = MODELS_DIR / model if not model.endswith(".gguf") else MODELS_DIR / model
    if model.endswith(".gguf") and (MODELS_DIR / model).is_file():
        return {"status": "success", "source": "local", "model": model, "path": str(MODELS_DIR / model)}

    mode = os.environ.get("HF_PULL_MODE", "auto")
    router_err = None
    if mode in {"auto", "router"}:
        r = pull_via_router(model)
        if not r.get("error"):
            return r
        router_err = r

    if mode in {"auto", "hub"}:
        r = pull_via_hub(model)
        if not r.get("error"):
            return r
        return {"error": "HF pull failed", "model": model, "router": router_err, "hub": r, "hint": "Check repo id, HF_TOKEN, and that the repo has .gguf files."}

    return router_err or {"error": "no pull method matched"}


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def normalize_payload(method: str, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["model"] = model
    body["stream"] = False

    if method in {"chat", "v1/chat/completions", "chat/completions"}:
        body.setdefault("messages", body.get("messages") or [])
    elif method in {"generate", "completion", "completions", "v1/completions"}:
        if "prompt" not in body and "input" in body:
            body["prompt"] = body.pop("input")
        opts = body.pop("options", None)
        if isinstance(opts, dict):
            for k, v in opts.items():
                body.setdefault(k, v)
        if "num_predict" in body and "max_tokens" not in body:
            body["max_tokens"] = body.pop("num_predict")
    elif method in {"embed", "embedding", "embeddings", "v1/embeddings"}:
        if "input" not in body and "prompt" in body:
            body["input"] = body.pop("prompt")
        if "input" not in body and "texts" in body:
            body["input"] = body.pop("texts")
    return body


# ---------------------------------------------------------------------------
# Top-level handler (RunPod)
# ---------------------------------------------------------------------------

def handler(job: dict[str, Any]) -> dict[str, Any]:
    inp = job.get("input") or {}
    method = (inp.get("method") or "chat").strip().lstrip("/").lower()

    # Meta / introspection
    if method == "health":
        return {"health": llama_health(), "gpu": gpu_info(), "cache": cache_info()}
    if method == "cache":
        return cache_info()
    if method == "config":
        cfg = RuntimeConfig.load()
        if inp.get("payload"):
            for k, v in inp["payload"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            cfg.save()
            return {"status": "updated", "config": cfg.to_dict()}
        return {"config": cfg.to_dict()}
    if method in {"list", "tags", "models"}:
        return {"models": llama_list_models(reload=True), "aliases_sample": []}
    if method == "pull":
        model = inp.get("model") or (inp.get("payload") or {}).get("model")
        return pull_model(model) if model else {"error": "pull requires model"}
    if method in {"rm", "delete"}:
        model = inp.get("model") or (inp.get("payload") or {}).get("model")
        r = _http.delete(f"{LLAMA_BASE}/models", params={"model": model}, timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.status_code, "details": r.text}
    if method == "show":
        model = inp.get("model")
        for m in llama_list_models():
            if m["id"] == model:
                return m
        return {"error": "model not found", "model": model}

    # Inference
    model = inp.get("model")
    if not model:
        return {"error": "Missing required field: model", "hint": "HF GGUF id, e.g. 'ggml-org/gemma-3-4b-it-GGUF:Q4_K_M'"}

    route = METHOD_ROUTES.get(method)
    if not route:
        return {"error": f"Unknown method: {method}", "supported": sorted(set(METHOD_ROUTES) | META_METHODS)}

    if not isinstance(payload, dict):
        return {"error": "payload must be a JSON object"}
    # Merge top-level input fields (model, prompt, messages, max_tokens, etc.)
    # into payload when payload is empty — callers send flat {"input": {...}}
    if not payload:
        skip = {"method", "model", "payload"}
        payload = {k: v for k, v in inp.items() if k not in skip}

    body = normalize_payload(method, model, payload)

    try:
        r = _http.post(f"{LLAMA_BASE}{route}", json=body, timeout=int(os.environ.get("REQUEST_TIMEOUT", "6000")))
        try:
            return r.json()
        except Exception:
            return {"raw": r.text, "status": r.status_code}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "route": route, "model": model}
