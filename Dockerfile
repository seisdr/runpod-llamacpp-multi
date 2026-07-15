# RunPod serverless worker: multi-model llama.cpp (CUDA) + Hugging Face GGUF pulls
# + FastAPI web UI with model manager + runtime config editor.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    runpod \
    requests \
    "huggingface_hub>=0.26.0" \
    fastapi \
    uvicorn \
    pydantic \
    pynvml

COPY handler.py config.py web.py main.py /app/

# Mount RunPod network volume at /runpod-volume (serverless default) so HF pulls + config survive cold starts
ENV LLAMA_CACHE=/runpod-volume/llama-cache \
    MODELS_DIR=/runpod-volume/models \
    CONFIG_DIR=/runpod-volume/config \
    RUNPOD_VOLUME=/runpod-volume \
    LLAMA_MODELS="" \
    DEFAULT_QUANT=Q4_K_M \
    HF_PULL_MODE=auto \
    MODELS_MAX=1 \
    N_GPU_LAYERS=999 \
    CTX_SIZE=8192 \
    LLAMA_PARALLEL=1 \
    FLASH_ATTN=auto \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    UI_HOST=0.0.0.0 \
    UI_PORT=8000

EXPOSE 8080 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8080/health > /dev/null || exit 1

# Own process lifecycle (image default ENTRYPOINT is llama-server)
WORKDIR /app
ENTRYPOINT []
CMD ["python3", "main.py"]
