from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.detector import ObjectDetector
from app.stream_manager import StreamManager
import json

app = FastAPI(title="Object Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

detector = ObjectDetector(model_path="yolov8n.pt", conf_threshold=0.5)
stream = StreamManager(detector)

@app.websocket("/ws/detect")
async def websocket_detect(ws: WebSocket):
    await ws.accept()
    stream.start(source=0)
    try:
        async for frame_data in stream.generate_frames():
            await ws.send_json(frame_data)
    except WebSocketDisconnect:
        pass
    finally:
        stream.stop()

@app.get("/api/stats")
def get_stats():
    return detector.get_statistics()

@app.post("/api/config")
def update_config(conf_threshold: float = 0.5, model: str = "yolov8n.pt"):
    detector.conf_threshold = conf_threshold
    return {"status": "updated"}
