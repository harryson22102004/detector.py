from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]
    track_id: Optional[int] = None

@dataclass
class FrameResult:
    detections: list[Detection]
    annotated_frame: np.ndarray
    inference_ms: float
    frame_number: int

class ObjectDetector:
    """YOLOv8-based object detector with tracking."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_counts: dict[str, int] = defaultdict(int)
        self.frame_count = 0

    def detect(self, frame: np.ndarray, track: bool = True) -> FrameResult:
        start = time.perf_counter()
        self.frame_count += 1

        if track:
            results = self.model.track(frame, conf=self.conf_threshold, persist=True, verbose=False)
        else:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for box in results[0].boxes:
            det = Detection(
                class_name=self.model.names[int(box.cls)],
                confidence=round(float(box.conf), 3),
                bbox=box.xyxy[0].tolist(),
                track_id=int(box.id) if box.id is not None else None,
            )
            detections.append(det)
            self.class_counts[det.class_name] += 1

        annotated = results[0].plot()
        inference_ms = (time.perf_counter() - start) * 1000

        return FrameResult(
            detections=detections,
            annotated_frame=annotated,
            inference_ms=round(inference_ms, 1),
            frame_number=self.frame_count,
        )

    def get_statistics(self) -> dict:
        return {
            "total_frames": self.frame_count,
            "class_counts": dict(self.class_counts),
            "unique_classes": len(self.class_counts),
        }
