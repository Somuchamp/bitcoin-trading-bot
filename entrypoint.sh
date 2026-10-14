#!/bin/bash
set -e

mkdir -p /app/data

echo "Starting FastAPI backend on port 8000..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

echo "Starting Binance Prediction & Paper Trading Worker in background..."
python -m src.prediction.worker &

echo "Starting Next.js Frontend on port 3000..."
cd /app/dashboard && npm start -- -p 3000 &

echo "Starting Nginx reverse proxy on public port 7860..."
nginx -g "daemon off;"
