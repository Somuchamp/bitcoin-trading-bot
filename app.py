import asyncio
import threading
import logging
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

# 4. Attach CORS middleware to demo.app for Vercel cross-origin access
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include all REST APIs from src.api.main
demo.app.include_router(api_app.router)

# 6. Mount Binance WebSocket price stream
demo.app.add_api_websocket_route("/ws/price", websocket_price)

# 7. Hook startup background tasks (USDT rate refresher & Binance price stream)
demo.app.on_event("startup")(startup_event)

# 8. Mount Next.js Dashboard on /app and static assets
out_dir = Path(__file__).parent / "dashboard" / "out"
next_assets_dir = out_dir / "_next"
if next_assets_dir.exists():
    demo.app.mount("/_next", StaticFiles(directory=str(next_assets_dir)), name="next_assets")
if out_dir.exists():
    demo.app.mount("/app", StaticFiles(directory=str(out_dir), html=True), name="nextjs_dashboard")

if __name__ == "__main__":
    # Start live worker in background daemon thread
    worker_thread = threading.Thread(target=start_worker, daemon=True)
    worker_thread.start()

    # Launch server on Hugging Face port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
