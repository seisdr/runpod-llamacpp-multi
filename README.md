# runpod-llamacpp-multi

[runpod-ollama-multi](https://github.com/SamratDuttaOfficial/runpod-ollama-multi) style worker, but **llama.cpp** (not Ollama) with **Hugging Face GGUF pulls** and a **web UI**.

## What you get

- `llama-server` (CUDA) in **router mode** — multi-model, pick per request
- **Pull models from Hugging Face** (GGUF repos)
- Auto-pull on first request, explicit `method=pull`, or pre-pull on boot
- **Web UI** (`/`) with model manager + live introspection:
  - Server status: PID, uptime, restart count, health, cmdline
  - GPU: utilization, VRAM bars, temps
  - Cache & disk: llama-cache + models_dir sizes and file list
  - Models table with status pills, pull form, presets, and test inference
  - Runtime config editor: change `n_gpu_layers`, `ctx_size`, `parallel`, `flash_attn`, `models_max`, `default_quant`, `hf_pull_mode`, host/port, extra args — **save without rebuilding the container**; llama-server restarts while models stay cached
- HF pull via **llama-server router** (`/models` endpoint) with **fallback to `huggingface_hub`**

## Pre-configured models

The web UI includes presets and the config defaults to these:

- `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`
- `llmfan46/Ornith-1.0-35B-uncensored-heretic-GGUF`
- `HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP`

The web UI will auto-select a `Q4_K_M` (or whatever `DEFAULT_QUANT` is) GGUF from any repo you paste in.

## Deploy on RunPod

1. Push this repo / connect GitHub to a Serverless **custom** endpoint
2. Mount a **network volume** at `/runpod-volume` (keeps HF downloads + config)
3. Container disk ≥ 40 GB
4. Set env vars (below)
5. Build

### Environment variables

| Variable | Description | Default / example |
|---|---|---|
| `LLAMA_MODELS` | Pre-pull these HF models on worker start | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` |
| `HF_TOKEN` | Hugging Face token (gated repos) | `hf_...` |
| `DEFAULT_QUANT` | Quant when you pass `org/repo` alone | `Q4_K_M` |
| `HF_PULL_MODE` | `auto` (router then hub), `router`, `hub` | `auto` |
| `MODEL_ALIASES` | Short name → HF id map | `qwen3:8b=unsloth/Qwen3-8B-GGUF:Q4_K_M` |
| `MODELS_MAX` | Max models in VRAM | `1` |
| `N_GPU_LAYERS` | GPU layers offloaded | `999` |
| `CTX_SIZE` | Context size | `8192` |
| `LLAMA_CACHE` | Router HF cache (on volume) | `/runpod-volume/llama-cache` |
| `MODELS_DIR` | Local GGUF dir / hub target | `/runpod-volume/models` |
| `CONFIG_DIR` | Persistent runtime config (on volume) | `/runpod-volume/config` |
| `UI_PORT` | Web UI port | `8000` |
| `LLAMA_EXTRA_ARGS` | Extra `llama-server` flags | `--cache-type-k q8_0` |

All of `N_GPU_LAYERS`, `CTX_SIZE`, `PARALLEL`, `FLASH_ATTN`, `MODELS_MAX`, `DEFAULT_QUANT`, `HF_PULL_MODE`, `LLAMA_EXTRA_ARGS` are also editable live in the web UI.

## Model id forms

| Form | Example |
|---|---|
| Repo + quant (preferred) | `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M` |
| Repo only | `ggml-org/gemma-3-4b-it-GGUF` → uses `DEFAULT_QUANT` |
| Exact file | `bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf` |
| Alias (optional) | `qwen3:8b` if mapped |

## Pulling from Hugging Face

### Auto-pull on first inference

```json
{
  "input": {
    "model": "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    "method": "chat",
    "payload": {
      "messages": [{ "role": "user", "content": "Hello" }]
    }
  }
}
```

If not cached → HF download, then serve.

### Explicit pull

```json
{ "input": { "method": "pull", "model": "unsloth/Qwen3-8B-GGUF:Q4_K_M" } }
```

### Web UI pull

Paste any HF repo into the Models tab. Presets for your models are built in.

### How pull works

1. **Router** (`POST /models` on llama-server) — native llama.cpp HF client into `LLAMA_CACHE` (uses `HF_TOKEN`)
2. If that fails and `HF_PULL_MODE=auto` → **`huggingface_hub`** selects the right `.gguf` and downloads it into `MODELS_DIR`

Either way the router auto-discovers the new file.

## Inference methods

| `method` | Backend route |
|---|---|
| `chat` (default) | `/v1/chat/completions` |
| `generate` / `completion` | `/v1/completions` |
| `embed` / `embeddings` | `/v1/embeddings` |
| `pull` | HF download |
| `list` / `tags` | cached models |
| `show` | single model status |
| `rm` | delete from cache |
| `health` | server + gpu + cache JSON |
| `cache` | disk usage |
| `config` | read/write runtime config |

Streaming is disabled (`stream: false`) for serverless.

### Generate example

```json
{
  "input": {
    "model": "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    "method": "generate",
    "payload": {
      "prompt": "Why is the sky blue?",
      "max_tokens": 256
    }
  }
}
```

## Runtime config (no rebuild)

Changing `n_gpu_layers`, `ctx_size`, `parallel`, `flash_attn`, `models_max`, `default_quant`, `hf_pull_mode`, or `extra_args` in the web UI writes to `/runpod-volume/config/runtime.json` and restarts llama-server in place — models stay on disk.

Changing `llama_host` / `llama_port` triggers a restart as well.

## Introspection

`GET /api/status` (web) returns full JSON. Via RunPod:

```json
{ "input": { "method": "health" } }
```

Returns `{server, health, gpu, cache}`.

## Notes

- **Network volume at `/runpod-volume` is required** if you do not want re-downloads every cold start and to keep config
- First pull takes minutes for large GGUFs; presets in the UI speed this up
- Gated models need `HF_TOKEN` on the endpoint
- Use HF GGUF ids — not `ollama.com/library` names — or map them with `MODEL_ALIASES`
- Speed: thinner stack than Ollama; real gains depend on model/quant/GPU settings

## Architecture

```
RunPod job ─► handler.py (pull + inference)
main.py  ─► ServerProcess (llama-server lifecycle)
         └─ background FastAPI (web.py)
              ├─ /api/status  server + gpu + cache
              ├─ /api/models  list / pull / delete
              ├─ /api/config  read / save + restart
              └─ /            vanilla-JS UI
```

Base image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
