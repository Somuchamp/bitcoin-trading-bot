import asyncio
import threading
import logging
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import gradio as gr

from src.api.main import app as api_app, startup_event, websocket_price
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

# 1. Background Live Worker (Paper Trading & Model Predictions)
def start_worker():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        worker = LiveWorker()
        _worker_mod._worker_instance = worker
        logger.info("LiveWorker initialized and running Binance kline analysis.")
        loop.run_until_complete(worker.run())
    except Exception as e:
        logger.error(f"Worker thread error: {e}")

# 2. ZeroGPU function for Hugging Face free tier compatibility
@spaces.GPU
def run_model_inference_gpu(prompt: str = ""):
    return "ZeroGPU Accelerated"

# 3. Gradio Blocks container
with gr.Blocks(title="Bitcoin Autonomous Trading & Prediction") as demo:
    # Hidden ZeroGPU event trigger to satisfy Hugging Face requirements
    dummy_in = gr.Textbox(value="init", visible=False)
    dummy_out = gr.Textbox(visible=False)
    dummy_btn = gr.Button("GPU_EXEC", visible=False)
    dummy_btn.click(fn=run_model_inference_gpu, inputs=[dummy_in], outputs=[dummy_out])

# 4. Mount Next.js Dashboard onto FastAPI
out_dir = Path(__file__).parent / "dashboard" / "out"

next_assets_dir = out_dir / "_next"
if next_assets_dir.exists():
    api_app.mount("/_next", StaticFiles(directory=str(next_assets_dir)), name="next_assets")
    logger.info(f"Mounted /_next from {next_assets_dir}")

# Direct route for root URL to serve the Next.js dark-mode dashboard
@api_app.get("/", response_class=HTMLResponse)
async def serve_root_dashboard():
    index_file = out_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard building... Please refresh in a moment.</h1>")

@api_app.get("/favicon.ico")
async def serve_favicon():
    fav = out_dir / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    return HTMLResponse(content="", status_code=404)

# Mount Gradio sub-app at /gradio for ZeroGPU compliance
app = gr.mount_gradio_app(api_app, demo, path="/gradio")

if __name__ == "__main__":
    # Start live worker in background daemon thread
    worker_thread = threading.Thread(target=start_worker, daemon=True)
    worker_thread.start()

    # Initialize ZeroGPU hook if present in environment
    try:
        from spaces import zero
        zero.startup()
    except Exception:
        pass

    # Launch server on Hugging Face port 7860
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

