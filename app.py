"""Flask server for the cigarette small-object detection study.

Serves the paper page and a live demo of the YOLO11n checkpoint that was
trained for the comparison.

Endpoints
---------
GET  /                  the single-page UI
POST /api/detect        single frame (image upload or webcam JPEG)
POST /api/video         start a background video job
GET  /api/job/<id>      job progress and result
GET  /api/health        model / runtime info
"""

from __future__ import annotations

import base64
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from detector import SmokingDetector

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_SECONDS = 180

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

detector = SmokingDetector()

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# job helpers
# --------------------------------------------------------------------------- #
def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "result": None,
            "error": None,
            "started": time.time(),
        }
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _prune_results(keep: int = 12) -> None:
    files = sorted(RESULTS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# video processing
# --------------------------------------------------------------------------- #
def _build_events(timeline: list[dict]) -> list[dict]:
    """Collapse per-sample hits into contiguous event windows."""
    events: list[dict] = []
    current: dict | None = None
    for sample in timeline:
        if sample["count"] > 0:
            if current is None:
                current = {
                    "start": sample["t"],
                    "end": sample["t"],
                    "samples": 0,
                    "hits": 0,
                    "max_confidence": sample["max_confidence"],
                }
            current["end"] = sample["t"]
            current["samples"] += 1
            current["hits"] += sample["count"]
            current["max_confidence"] = max(
                current["max_confidence"], sample["max_confidence"]
            )
        elif current is not None:
            events.append(current)
            current = None
    if current is not None:
        events.append(current)
    return events


def _process_video(
    job_id: str,
    source: Path,
    conf: float,
    iou: float,
    target_fps: float,
    max_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        _update_job(job_id, status="error", error="Could not open the uploaded video.")
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, int(round(fps / max(0.5, target_fps))))
        limit = total // stride if total else 0
        if max_frames:
            limit = min(limit, max_frames) if limit else max_frames

        ok, first = cap.read()
        if not ok:
            _update_job(job_id, status="error", error="The video contains no readable frames.")
            return

        height, width = first.shape[:2]
        scale = min(1.0, 960 / max(width, 1))
        out_w = int(width * scale) // 2 * 2
        out_h = int(height * scale) // 2 * 2
        writer_fps = max(1.0, min(target_fps, fps))

        out_path = RESULTS_DIR / f"{job_id}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (out_w, out_h)
        )

        timeline: list[dict] = []
        processed = 0
        total_hits = 0
        max_conf = 0.0
        frames_with_hits = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % stride:
                continue
            if max_frames and processed >= max_frames:
                break
            if frame_index > fps * MAX_VIDEO_SECONDS:
                break

            detections = detector.detect(frame, conf=conf, iou=iou)
            annotated = detector.annotate(frame, detections)
            if (out_w, out_h) != (annotated.shape[1], annotated.shape[0]):
                annotated = cv2.resize(annotated, (out_w, out_h))
            writer.write(annotated)

            hits = len(detections)
            top = max((d["confidence"] for d in detections), default=0.0)
            timeline.append(
                {
                    "t": round(frame_index / fps, 2),
                    "count": hits,
                    "max_confidence": round(top, 4),
                }
            )
            total_hits += hits
            max_conf = max(max_conf, top)
            if hits:
                frames_with_hits += 1
            processed += 1

            if processed % 3 == 0 or (limit and processed >= limit):
                pct = round(100 * processed / limit) if limit else 0
                _update_job(
                    job_id,
                    status="running",
                    progress=min(pct, 99),
                    message=f"Analyzed {processed} of ~{limit or '?'} frames",
                )

        writer.release()

        result = {
            "video_url": f"/static/results/{job_id}.mp4",
            "timeline": timeline,
            "events": _build_events(timeline),
            "summary": {
                "frames_analyzed": processed,
                "frames_with_detection": frames_with_hits,
                "total_detections": total_hits,
                "max_confidence": round(max_conf, 4),
                "duration_seconds": round(len(timeline) / writer_fps, 2) if timeline else 0,
            },
        }
        _update_job(job_id, status="done", progress=100, message="Complete", result=result)
    except Exception as exc:  # noqa: BLE001 - surface the message to the UI
        _update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        cap.release()
        try:
            source.unlink()
        except OSError:
            pass
        _prune_results()


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": detector.model_path.name,
            "classes": list(detector.names.values()),
            "device": detector.device or "auto",
        }
    )


@app.post("/api/detect")
def detect_frame():
    started = time.perf_counter()
    try:
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
    except ValueError:
        return jsonify({"error": "conf and iou must be numbers"}), 400
    want_annotated = request.form.get("annotate", "1") != "0"

    upload = request.files.get("image")
    raw = upload.read() if upload else request.get_data()
    if not raw:
        return jsonify({"error": "No image data received."}), 400

    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Unsupported or corrupt image."}), 400

    detections = detector.detect(frame, conf=conf, iou=iou)
    payload = {
        "count": len(detections),
        "detections": detections,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "inference_ms": round((time.perf_counter() - started) * 1000, 1),
    }

    if want_annotated:
        annotated = detector.annotate(frame, detections)
        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            payload["annotated"] = (
                "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
            )
    return jsonify(payload)


@app.post("/api/video")
def start_video_job():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "No video file received."}), 400

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEOS:
        return jsonify({"error": f"Unsupported video type '{suffix}'."}), 400

    try:
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
        target_fps = float(request.form.get("fps", 3))
        max_frames = int(request.form.get("max_frames", 240))
    except ValueError:
        return jsonify({"error": "Invalid processing options."}), 400

    job_id = _new_job()
    source = RESULTS_DIR / f"{job_id}_source{suffix}"
    upload.save(source)

    threading.Thread(
        target=_process_video,
        args=(job_id, source, conf, iou, target_fps, max_frames),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        return jsonify({"error": "Unknown job id."}), 404
    return jsonify(snapshot)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File is too large (limit: 200 MB)."}), 413


if __name__ == "__main__":
    _prune_results()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
