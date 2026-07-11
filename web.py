#!/usr/bin/env python3
"""FastAPI web UI with model manager + runtime config editor.

Served alongside llama-server. Endpoints are REST; the frontend is vanilla JS.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config as cfg_mod
from handler import (
    llama_health,
    llama_list_models,
    pull_model,
    pull_via_hub,
    pull_via_router,
    resolve_quant,
)

VOLUME = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))

# Shared mutable server handle goes in module global (set by main at bootstrap)
server_handle: cfg_mod.ServerProcess | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="llama.cpp RunPod", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _INDEX_HTML

    # ---- status ----

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return {
            "server": server_handle.status_dict() if server_handle else {"alive": False},
            "health": llama_health(),
            "gpu": cfg_mod.gpu_info(),
            "cache": cfg_mod.cache_info(),
        }

    # ---- models ----

    @app.get("/api/models")
    async def api_models(reload: bool = True) -> dict[str, Any]:
        return {"models": llama_list_models(reload=reload)}

    class PullReq(BaseModel):
        model: str
        mode: str | None = None  # override HF_PULL_MODE for this pull

    @app.post("/api/pull")
    async def api_pull(req: PullReq) -> dict[str, Any]:
        if not req.model:
            raise HTTPException(400, "model required")
        return pull_model(req.model)

    @app.delete("/api/models/{model_id:path}")
    async def api_delete_model(model_id: str) -> dict[str, Any]:
        import requests as rq
        base = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
        r = rq.delete(f"{base}/models", params={"model": model_id}, timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.status_code, "details": r.text}

    # ---- config ----

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        cfg = cfg_mod.RuntimeConfig.load()
        return {"config": cfg.to_dict()}

    class ConfigUpdate(BaseModel):
        n_gpu_layers: int | None = None
        ctx_size: int | None = None
        parallel: int | None = None
        flash_attn: str | None = None
        models_max: int | None = None
        extra_args: str | None = None
        default_quant: str | None = None
        hf_pull_mode: str | None = None
        llama_host: str | None = None
        llama_port: int | None = None

    @app.post("/api/config")
    async def set_config(req: ConfigUpdate) -> dict[str, Any]:
        cfg = cfg_mod.RuntimeConfig.load()
        changed = False
        restart_keys = {"llama_host", "llama_port"}  # these need restart
        for k, v in req.dict(exclude_none=True).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                changed = True
        if not changed:
            return {"status": "no_change", "config": cfg.to_dict()}
        cfg.save()
        needs_restart = any(getattr(cfg, k) != cfg_mod.RuntimeConfig().to_dict().get(k) and k in restart_keys for k in req.dict(exclude_none=True))
        if needs_restart and server_handle:
            server_handle.restart()
        return {"status": "updated", "config": cfg.to_dict(), "needs_restart": needs_restart}

    @app.post("/api/restart")
    async def restart_server() -> dict[str, Any]:
        if server_handle:
            server_handle.restart()
            return {"status": "restarting"}
        return {"error": "no server handle"}

    # ---- test inference (small call) ----

    class TryReq(BaseModel):
        model: str
        prompt: str = "Hello"

    @app.post("/api/try")
    async def api_try(req: TryReq) -> dict[str, Any]:
        try:
            from handler import normalize_payload
            base = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
            import requests as rq
            body = normalize_payload("generate", req.model, {
                "prompt": req.prompt,
                "max_tokens": 32,
            })
            r = rq.post(f"{base}/v1/completions", json=body, timeout=120)
            return r.json() if r.status_code == 200 else {"error": r.status_code, "details": r.text}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    return app


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>llama.cpp RunPod</title>
<style>
:root {
  --bg: #0f1117; --panel: #171a23; --fg: #e5e7eb; --muted: #9ca3af;
  --accent: #60a5fa; --ok: #34d399; --warn: #fbbf24; --err: #f87171; --border: #282c34;
}
* { box-sizing: border-box; }
body { margin:0; font:14px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
header { padding:14px 18px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:12px; }
header h1 { margin:0; font-size:16px; font-weight:600; }
.tabbar { padding:8px 18px; display:flex; gap:6px; border-bottom:1px solid var(--border); flex-wrap:wrap; }
.tabbar button { background:var(--panel); color:var(--muted); border:1px solid var(--border); padding:6px 12px; border-radius:6px; cursor:pointer; font-size:13px; }
.tabbar button.active { color:var(--fg); border-color:var(--accent); }
.container { padding:18px; max-width:1100px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }
.card h2 { margin:0 0 10px; font-size:14px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.row { display:flex; gap:14px; flex-wrap:wrap; }
.kv { display:grid; grid-template-columns: 160px 1fr; gap:6px 12px; font-size:13px; }
.kv div:nth-child(odd) { color:var(--muted); }
.bar { height:6px; background:#222; border-radius:3px; overflow:hidden; margin-top:6px; }
.bar > span { display:block; height:100%; background:var(--accent); }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }
.pill.ok { background:rgba(52,211,153,.15); color:var(--ok); }
.pill.warn { background:rgba(251,191,36,.15); color:var(--warn); }
.pill.err { background:rgba(248,113,113,.15); color:var(--err); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:500; }
input, select, textarea { background:#0c0e14; border:1px solid var(--border); color:var(--fg); padding:6px 8px; border-radius:6px; font:13px system-ui,sans-serif; }
label { display:block; font-size:12px; color:var(--muted); margin-bottom:3px; }
.form-row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; margin-bottom:10px; }
.form-row > div { flex:1; min-width:140px; }
.btn { background:var(--accent); color:#111; border:0; padding:7px 14px; border-radius:6px; cursor:pointer; font-weight:600; font-size:13px; }
.btn.secondary { background:var(--panel); color:var(--fg); border:1px solid var(--border); font-weight:500; }
.btn.danger { background:var(--err); color:#fff; }
.btn:disabled { opacity:.5; cursor:default; }
.toast { position:fixed; right:16px; bottom:16px; background:var(--panel); border:1px solid var(--border); padding:10px 14px; border-radius:8px; font-size:13px; display:none; max-width:320px; }
.toast.show { display:block; }
.toast.ok { border-color:var(--ok); }
.toast.err { border-color:var(--err); }
pre.log { background:#0c0e14; padding:10px; border-radius:6px; max-height:280px; overflow:auto; font-size:12px; white-space:pre-wrap; }
</style>
</head>
<body>
<header>
  <h1>llama.cpp RunPod</h1>
  <span id="srv-pill" class="pill">…</span>
</header>
<div class="tabbar">
  <button class="active" data-tab="status">Status</button>
  <button data-tab="models">Models</button>
  <button data-tab="config">Config</button>
  <button data-tab="logs">Logs</button>
</div>
<div class="container">

<!-- STATUS -->
<section id="tab-status" class="tab">
  <div class="row">
    <div class="card" style="flex:1;min-width:280px">
      <h2>Server</h2>
      <div class="kv" id="srv-kv"></div>
    </div>
    <div class="card" style="flex:1;min-width:280px">
      <h2>GPU</h2>
      <div id="gpu-info"></div>
    </div>
  </div>
  <div class="card">
    <h2>Cache & Disk</h2>
    <div class="kv" id="cache-kv"></div>
  </div>
</section>

<!-- MODELS -->
<section id="tab-models" class="tab" style="display:none">
  <div class="card">
    <h2>Pull a model (HuggingFace)</h2>
    <div class="form-row">
      <div style="flex:2;min-width:240px">
        <label>Model id (e.g. org/repo:Q4_K_M or org/repo/file.gguf)</label>
        <input id="pull-model" style="width:100%" placeholder="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"/>
      </div>
      <div>
        <label>&nbsp;</label>
        <button class="btn" id="pull-btn">Pull</button>
      </div>
    </div>
    <div class="form-row">
      <div style="flex:2;min-width:240px">
        <label>Quick: preset models</label>
        <select id="pull-preset">
          <option value="">— pick —</option>
          <option value="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive">Qwen3.6-35B-A3B (HauhauCS Aggressive)</option>
          <option value="llmfan46/Ornith-1.0-35B-uncensored-heretic-GGUF">Ornith-1.0-35B (llmfan46)</option>
          <option value="HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP">Gemma4-31B-QAT (HauhauCS Balanced)</option>
        </select>
      </div>
      <div>
        <label>&nbsp;</label>
        <button class="btn secondary" id="pull-preset-btn">Use preset</button>
      </div>
    </div>
    <pre class="log" id="pull-log" style="margin-top:10px"></pre>
  </div>

  <div class="card">
    <h2>Cached / loaded models</h2>
    <table>
      <thead><tr><th>Id</th><th>Status</th><th>Path</th><th>Info</th><th></th></tr></thead>
      <tbody id="models-tbody"></tbody>
    </table>
    <div style="margin-top:10px"><button class="btn secondary" id="refresh-btn">Refresh</button></div>
  </div>

  <div class="card">
    <h2>Try inference</h2>
    <div class="form-row">
      <div>
        <label>Model</label>
        <select id="try-model"></select>
      </div>
      <div style="flex:2">
        <label>Prompt</label>
        <input id="try-prompt" style="width:100%" value="Hello"/>
      </div>
      <div><label>&nbsp;</label><button class="btn" id="try-btn">Try</button></div>
    </div>
    <pre class="log" id="try-log" style="margin-top:10px"></pre>
  </div>
</section>

<!-- CONFIG -->
<section id="tab-config" class="tab" style="display:none">
  <div class="card">
    <h2>llama-server settings</h2>
    <p style="color:var(--muted);font-size:12px;margin-top:0">Saved to disk; restarting llama-server (not the container) applies them. Models stay cached.</p>
    <div class="row">
      <div style="flex:1;min-width:200px">
        <label>n_gpu_layers</label>
        <input type="number" id="c_n_gpu_layers" style="width:100%"/>
      </div>
      <div style="flex:1;min-width:200px">
        <label>ctx_size</label>
        <input type="number" id="c_ctx_size" style="width:100%"/>
      </div>
      <div style="flex:1;min-width:200px">
        <label>parallel</label>
        <input type="number" id="c_parallel" style="width:100%"/>
      </div>
      <div style="flex:1;min-width:200px">
        <label>flash_attn</label>
        <select id="c_flash_attn"><option>auto</option><option>on</option><option>off</option></select>
      </div>
      <div style="flex:1;min-width:200px">
        <label>models_max</label>
        <input type="number" id="c_models_max" style="width:100%"/>
      </div>
      <div style="flex:1;min-width:200px">
        <label>default_quant</label>
        <input id="c_default_quant" style="width:100%"/>
      </div>
      <div style="flex:1;min-width:200px">
        <label>hf_pull_mode</label>
        <select id="c_hf_pull_mode"><option>auto</option><option>router</option><option>hub</option></select>
      </div>
      <div style="flex:1;min-width:200px">
        <label>llama_host</label>
        <input id="c_llama_host" style="width:100%"/>
      </div>
      <div style="flex:1;min-width:200px">
        <label>llama_port</label>
        <input type="number" id="c_llama_port" style="width:100%"/>
      </div>
    </div>
    <div class="form-row">
      <div style="flex:1">
        <label>extra_args</label>
        <input id="c_extra_args" style="width:100%" placeholder="--cache-type-k q8_0 ..."/>
      </div>
    </div>
    <div class="form-row">
      <button class="btn" id="save-config">Save & Apply</button>
      <button class="btn secondary" id="reload-config">Reload</button>
      <button class="btn danger" id="restart-server">Restart llama-server</button>
    </div>
  </div>
</section>

<!-- LOGS -->
<section id="tab-logs" class="tab" style="display:none">
  <div class="card">
    <h2>Status + raw JSON</h2>
    <button class="btn secondary" id="refresh-logs">Refresh</button>
    <pre class="log" id="logs-out" style="margin-top:10px"></pre>
  </div>
</section>

</div>
<div class="toast" id="toast"></div>

<script>
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

function toast(msg, kind='ok') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast show ' + kind;
  setTimeout(() => el.className = 'toast', 2500);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

// tabs
$$('.tabbar button').forEach(b => b.onclick = () => {
  $$('.tabbar button').forEach(x => x.classList.remove('active'));
  $$('.tab').forEach(x => x.style.display = 'none');
  b.classList.add('active');
  $('#tab-' + b.dataset.tab).style.display = '';
});

// status
async function refreshStatus() {
  const s = await api('/api/status');
  const srv = s.server || {};
  const health = s.health || {};
  const gpu = s.gpu || {};
  const cache = s.cache || {};

  $('#srv-pill').textContent = srv.alive ? '● alive' : '○ down';
  $('#srv-pill').className = 'pill ' + (srv.alive ? 'ok' : 'err');

  const skv = $('#srv-kv');
  skv.innerHTML = `
    <div>PID</div><div>${srv.pid ?? '—'}</div>
    <div>Uptime</div><div>${Math.floor(srv.uptime_seconds||0)}s</div>
    <div>Restarts</div><div>${srv.restart_count||0}</div>
    <div>Health endpoint</div><div class="pill ${health.ok?'ok':'err'}">${health.ok?'ok':'fail'}</div>
    <div>Host:port</div><div>${srv.config?.llama_host}:${srv.config?.llama_port}</div>
    <div>cmd</div><div style="font-size:11px;word-break:break-all">${(srv.cmdline||[]).join(' ')}</div>
  `;

  const gi = $('#gpu-info');
  if (!gpu.ok) { gi.innerHTML = `<span class="pill err">No GPU / ${gpu.error||''}</span>`; }
  else {
    gi.innerHTML = (gpu.devices||[]).map(d => `
      <div style="margin-bottom:10px">
        <div><b>${d.name}</b> <span class="pill">${d.index}</span></div>
        <div>VRAM ${d.memory_used_mb}/${d.memory_total_mb} MB
          <div class="bar"><span style="width:${d.memory_total_mb?Math.round(d.memory_used_mb/d.memory_total_mb*100):0}%"></span></div>
        </div>
        <div>GPU ${d.gpu_utilization}% · mem ${d.memory_utilization}% · ${d.temperature_c||'?'}°C</div>
      </div>
    `).join('');
  }

  const ck = $('#cache-kv');
  const lc = cache.llama_cache || {}, md = cache.models_dir || {};
  ck.innerHTML = `
    <div>llama_cache</div><div>${lc.exists?`${lc.ggufs} ggufs, ${lc.size_mb} MB (${lc.files} files)`:'—'}</div>
    <div>models_dir</div><div>${md.exists?`${md.ggufs} ggufs, ${md.size_mb} MB (${md.files} files)`:'—'}</div>
  `;
  if (lc.gguf_sizes_mb) {
    const rows = Object.entries(lc.gguf_sizes_mb).map(([k,v]) => `<div style="font-size:11px;color:var(--muted)">${k}</div><div style="font-size:11px">${v} MB</div>`).join('');
    ck.innerHTML += `<div style="grid-column:1/3;margin-top:6px"><details><summary style="cursor:pointer;color:var(--muted)">cache files</summary><div class="kv" style="margin-top:6px">${rows}</div></details></div>`;
  }
}

// models
async function refreshModels() {
  const m = await api('/api/models?reload=1');
  const models = m.models || [];
  const tb = $('#models-tbody');
  if (!models.length) { tb.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">No models yet — pull one below.</td></tr>'; }
  else {
    tb.innerHTML = models.map(x => `
      <tr>
        <td style="word-break:break-all">${x.id}</td>
        <td><span class="pill ${x.status==='loaded'?'ok':x.status==='failed'?'err':'warn'}">${x.status||'?'}</span></td>
        <td style="font-size:11px;color:var(--muted);word-break:break-all">${x.path||''}</td>
        <td style="font-size:11px;color:var(--muted)">${x.meta?JSON.stringify(x.meta):''}</td>
        <td><button class="btn danger" style="padding:3px 8px;font-size:11px" data-rm="${x.id}">rm</button></td>
      </tr>
    `).join('');
    tb.querySelectorAll('[data-rm]').forEach(b => b.onclick = async () => {
      if (!confirm('Remove ' + b.dataset.rm + '?')) return;
      await api('/api/models/' + encodeURIComponent(b.dataset.rm), {method:'DELETE'});
      refreshModels();
    });
  }
  // try-model selector
  const sel = $('#try-model');
  sel.innerHTML = models.map(m => `<option>${m.id}</option>`).join('');
}

$('#pull-btn').onclick = async () => {
  const m = $('#pull-model').value.trim();
  if (!m) return toast('Enter a model id', 'err');
  $('#pull-log').textContent = 'Starting pull of ' + m + ' ...';
  const r = await api('/api/pull', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({model:m})});
  $('#pull-log').textContent = JSON.stringify(r, null, 2);
  if (r.status === 'success') toast('Pull started / done');
  else toast('Pull failed', 'err');
  refreshModels();
};
$('#pull-preset-btn').onclick = () => {
  const v = $('#pull-preset').value;
  if (v) $('#pull-model').value = v;
};
$('#refresh-btn').onclick = refreshModels;

$('#try-btn').onclick = async () => {
  const m = $('#try-model').value, p = $('#try-prompt').value;
  $('#try-log').textContent = 'Running...';
  const r = await api('/api/try', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({model:m, prompt:p})});
  $('#try-log').textContent = JSON.stringify(r, null, 2);
};

// config
function loadConfig(c) {
  $('#c_n_gpu_layers').value = c.n_gpu_layers;
  $('#c_ctx_size').value = c.ctx_size;
  $('#c_parallel').value = c.parallel;
  $('#c_flash_attn').value = c.flash_attn;
  $('#c_models_max').value = c.models_max;
  $('#c_default_quant').value = c.default_quant;
  $('#c_hf_pull_mode').value = c.hf_pull_mode;
  $('#c_llama_host').value = c.llama_host;
  $('#c_llama_port').value = c.llama_port;
  $('#c_extra_args').value = c.extra_args;
}
async function fetchConfig() {
  const r = await api('/api/config');
  loadConfig(r.config);
}
$('#reload-config').onclick = fetchConfig;
$('#save-config').onclick = async () => {
  const cfg = {
    n_gpu_layers: parseInt($('#c_n_gpu_layers').value),
    ctx_size: parseInt($('#c_ctx_size').value),
    parallel: parseInt($('#c_parallel').value),
    flash_attn: $('#c_flash_attn').value,
    models_max: parseInt($('#c_models_max').value),
    default_quant: $('#c_default_quant').value,
    hf_pull_mode: $('#c_hf_pull_mode').value,
    llama_host: $('#c_llama_host').value,
    llama_port: parseInt($('#c_llama_port').value),
    extra_args: $('#c_extra_args').value,
  };
  const r = await api('/api/config', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(cfg)});
  toast(r.status === 'updated' ? 'Saved' : r.status);
  if (r.needs_restart) toast('Host/port changed — server will restart', 'warn');
  refreshStatus();
};
$('#restart-server').onclick = async () => {
  await api('/api/restart', {method:'POST'});
  toast('Restarting llama-server...');
  setTimeout(refreshStatus, 2000);
};

// logs
$('#refresh-logs').onclick = async () => {
  const s = await api('/api/status');
  $('#logs-out').textContent = JSON.stringify(s, null, 2);
};

refreshStatus(); refreshModels(); fetchConfig();
setInterval(refreshStatus, 4000);
</script>
</body>
</html>
"""
