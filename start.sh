#!/bin/sh
set -e
echo "=== llmacpp-multi boot ==="
apt-get update -qq && apt-get install -y -qq python3 python3-pip ca-certificates curl 2>&1 | tail -5
pip3 install --break-system-packages runpod requests fastapi uvicorn pydantic "huggingface_hub>=0.26.0" 2>&1 | tail -5
echo "=== downloading code ==="
curl -sSL https://raw.githubusercontent.com/seisdr/runpod-llamacpp-multi/master/handler.py -o /app/handler.py
curl -sSL https://raw.githubusercontent.com/seisdr/runpod-llamacpp-multi/master/config.py -o /app/config.py
curl -sSL https://raw.githubusercontent.com/seisdr/runpod-llamacpp-multi/master/web.py -o /app/web.py
curl -sSL https://raw.githubusercontent.com/seisdr/runpod-llamacpp-multi/master/main.py -o /app/main.py
echo "=== starting ==="
exec python3 /app/main.py
