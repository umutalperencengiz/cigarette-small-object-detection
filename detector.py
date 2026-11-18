"""Model wrapper for the YOLO11n checkpoint from the study.

The checkpoint in ``models/`` is a YOLO11-nano model fine-tuned to localise
cigarettes (single class: ``cigarette``) as part of the comparative
Faster R-CNN / RetinaNet / YOLO11 benchmark.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "cigarette_yolo11n.pt"

# BGR colours
GREEN = (120, 220, 60)
RED = (60, 60, 220)


class SmokingDetector:
    """Thread-safe wrapper around the YOLO checkpoint."""

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        env_path = os.environ.get("MODEL_PATH")
        self.model_path = Path(model_path or env_path or DEFAULT_MODEL)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}. "
                "Place the trained YOLO weights there or set MODEL_PATH."
            )
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names
        self.device = os.environ.get("MODEL_DEVICE") or None
        self._lock = threading.Lock()

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[dict]:
        """Run inference on a single BGR frame."""
        with self._lock:
            result = self.model.predict(
                frame_bgr,
                conf=max(0.01, min(conf, 0.99)),
                iou=max(0.01, min(iou, 0.99)),
                verbose=False,
                device=self.device,
            )[0]

        detections: list[dict] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                {
                    "label": self.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    @staticmethod
    def annotate(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw boxes and labels onto a copy of the frame."""
        img = frame_bgr.copy()
        h, w = img.shape[:2]
        thickness = max(2, round(min(w, h) / 320))
        font_scale = max(0.45, min(w, h) / 900)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            colour = GREEN if det["confidence"] >= 0.5 else RED
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

            label = f'{det["label"]} {det["confidence"]:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            top = max(0, y1 - th - baseline - 6)
            label_x = min(x1, max(0, w - tw - 8))
            cv2.rectangle(img, (label_x, top), (label_x + tw + 8, y1), colour, -1)
            cv2.putText(
                img,
                label,
                (label_x + 4, y1 - baseline - 2),
                font,
                font_scale,
                (10, 15, 20),
                1,
                cv2.LINE_AA,
            )
        return img
