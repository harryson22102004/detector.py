import cv2
import asyncio
import base64
from typing import Optional
from app.detector import ObjectDetector

class StreamManager:
    """Manages video capture and detection streaming."""

    def __init__(self, detector: ObjectDetector):
        self.detector = detector
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False

    def start(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()

    async def generate_frames(self):
        while self.running and self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break
            result = self.detector.detect(frame)
            _, buffer = cv2.imencode('.jpg', result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            yield {
                "image": img_b64,
                "detections": [
                    {"class": d.class_name, "confidence": d.confidence,
                     "bbox": d.bbox, "track_id": d.track_id}
                    for d in result.detections
                ],
                "count": len(result.detections),
                "inference_ms": result.inference_ms,
                "frame": result.frame_number,
            }
            await asyncio.sleep(0.033)  # ~30 FPS
