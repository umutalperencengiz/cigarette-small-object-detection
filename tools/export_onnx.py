"""Re-export the shipped YOLO11n checkpoint to ONNX for the browser demo.

    pip install ultralytics
    python tools/export_onnx.py

The page expects `models/cigarette_yolo11n.onnx` with a static 640x640 input
and the usual Ultralytics detection head output `[1, 4 + classes, anchors]`.
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "cigarette_yolo11n.pt"


def main() -> None:
    model = YOLO(str(CHECKPOINT))
    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        dynamic=False,
        simplify=False,
    )
    print(f"exported: {exported}")


if __name__ == "__main__":
    main()
