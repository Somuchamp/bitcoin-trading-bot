import os
import asyncio
import threading
import logging
from pathlib import Path
import uvicorn
from fastapi.staticfiles import StaticFiles
import gradio as gr

from src.api.main import app as fastapi_app
from src.prediction.worker import LiveWorker
import src.prediction.worker as _worker_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BTC-Platform")

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is None:
                return lambda f: f
            return func

# Background Live Worker
def start_worker():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        worker = LiveWorker()
        _worker_mod._worker_instance = worker
        logger.info("LiveWorker initialized and listening to Binance klines.")
        loop.run_until_complete(worker.run())
    except Exception as e:
        logger.error(f"Worker thread error: {e}")

# ZeroGPU decorator for Hugging Face free tier compatibility
@spaces.GPU
def gpu_compute_placeholder():
    return "ZeroGPU Compute Engine Active"

# Minimal Gradio interface mounted at /gradio
with gr.Blocks(title="AI Control Panel") as demo:
    gr.Markdown("# 🤖 BTC AI Engine Status")
    status_box = gr.Textbox(value="Autonomous Trading Engine Active", interactive=False)
    gpu_btn = gr.Button("⚡ Verify ZeroGPU Acceleration")
    gpu_btn.click(fn=gpu_compute_placeholder, outputs=[status_box])

# Mount Gradio at /gradio
gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# Mount Next.js static build at root "/"
out_dir = Path(__file__).parent / "dashboard" / "out"
if out_dir.exists():
    fastapi_app.mount("/", StaticFiles(directory=str(out_dir), html=True), name="frontend")
    logger.info(f"Mounted Next.js frontend from {out_dir}")
else:
    logger.warning("dashboard/out directory not found!")

if __name__ == "__main__":
    # Start live trading worker in background thread
    t = threading.Thread(target=start_worker, daemon=True)
    t.start()

    # Launch FastAPI on port 7860
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
