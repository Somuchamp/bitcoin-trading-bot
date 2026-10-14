import asyncio
import threading
import logging
from pathlib import Path
import gradio as gr
from fastapi.staticfiles import StaticFiles

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

# 3. Build Full-Viewport Dashboard using Gradio + Next.js
custom_css = """
body, html {
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background-color: #0b0f19 !important;
}
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    width: 100vw !important;
    border: none !important;
}
footer {
    display: none !important;
}
.contain {
    padding: 0 !important;
    margin: 0 !important;
}
"""

with gr.Blocks(title="Bitcoin AI Trading Platform") as demo:
    # Fullscreen responsive iframe embedding the custom Next.js dark-mode dashboard
    gr.HTML("""
    <iframe
        id="nextjs-dashboard-frame"
        src="/app"
        style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; border: none; margin: 0; padding: 0; z-index: 99999;"
        allow="clipboard-write; fullscreen"
    ></iframe>
    """)
    
    # Hidden ZeroGPU event trigger to satisfy Hugging Face requirements
    with gr.Row(visible=False):
        dummy_in = gr.Textbox(value="init", visible=False)
        dummy_out = gr.Textbox(visible=False)
        dummy_btn = gr.Button("GPU_EXEC", visible=False)
        dummy_btn.click(fn=run_model_inference_gpu, inputs=[dummy_in], outputs=[dummy_out])

# 4. Mount API Routes and Static Next.js Build onto Gradio's FastAPI engine
out_dir = Path(__file__).parent / "dashboard" / "out"
if out_dir.exists():
    demo.app.mount("/app", StaticFiles(directory=str(out_dir), html=True), name="nextjs_dashboard")
    logger.info(f"Mounted Next.js static build from {out_dir} at /app")
else:
    logger.warning("dashboard/out directory not found!")

# Include REST API routes from src.api.main
demo.app.include_router(api_app.router)

# Mount Binance WebSocket price stream
demo.app.add_api_websocket_route("/ws/price", websocket_price)

# Hook startup background tasks (USDT rate refresher & Binance price stream)
demo.app.on_event("startup")(startup_event)

if __name__ == "__main__":
    # Start live worker in background daemon thread
    worker_thread = threading.Thread(target=start_worker, daemon=True)
    worker_thread.start()

    # Launch Gradio server on Hugging Face port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
