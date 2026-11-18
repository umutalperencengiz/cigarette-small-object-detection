# Cigarette Small Object Detection

A comparative study of **Faster R-CNN**, **RetinaNet** and **YOLO11** on the same
cigarette-detection dataset — plus a browser demo that runs the trained YOLO11 model
locally.

> **Paper:** [A Comparative Analysis of Convolutional Neural Networks for Cigarette
> Small Object Detection](static/paper/SmallObjectDetectionPaper.pdf) · Umut Alperen Cengiz ·
> October 2025

![Page preview](docs/preview.png)

## Results

Validation split: 1,370 images, single class `cigarette`. Values reproduced from Table 1
of the manuscript. The bundled checkpoint is the `train15` run whose predictions appear in
the paper's figures.

| Model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> | Training |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Faster R-CNN** (R50-FPN) | **43.64** | 87.25 | **38.95** | **27.41** | **45.07** | **51.75** | 10–12 h |
| RetinaNet (R50-FPN) | 41.76 | 84.67 | 35.70 | 24.30 | 43.36 | 51.23 | 8–10 h |
| YOLO11n | 40.40 | **88.50** | 35.20 | — | — | — | **3–4 h** |

**Takeaway.** Two-stage detection still wins on accuracy, and the margin grows as the IoU
threshold tightens. YOLO11 gives up ~3 AP overall but trains in roughly a third of the time
and has the best AP<sub>50</sub> — the trade-off the paper argues for.

## Demo

The Flask app serves the study page and a live demo of the shipped YOLO11n checkpoint:

- **Image** — drag-and-drop, annotated frame, per-box confidence.
- **Video** — background job, sampled frames, annotated MP4, hit timeline and merged events.
- **Camera** — live overlay at a few frames per second with a rolling confidence sparkline.

Everything runs on the local machine; no frames are uploaded anywhere.

Only YOLO11 is bundled. The Faster R-CNN (322 MB) and RetinaNet (296 MB) checkpoints were
trained for CUDA and are not included; their published predictions are shown in the page
gallery instead.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Camera access requires `localhost` or HTTPS.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/cigarette_yolo11n.pt` | Swap in another YOLO checkpoint |
| `MODEL_DEVICE` | auto | `cpu`, `0`, `cuda:0`, … |

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Model name, classes, device |
| `POST` | `/api/detect` | `multipart/form-data`: `image`, `conf`, `iou`, `annotate` |
| `POST` | `/api/video` | `multipart/form-data`: `video`, `conf`, `iou`, `fps`, `max_frames` |
| `GET` | `/api/job/<id>` | Job status, progress, timeline and events |

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -F "image=@frame.jpg" -F "conf=0.25" -F "iou=0.45" -F "annotate=1"
```

## Dataset

Three public Roboflow sets, merged into one class and cleaned of false annotations:

| Source | Link |
| --- | --- |
| Smoking Detection — Lppro | <https://universe.roboflow.com/lppro/smoking-detection-yg80w> |
| Cigarette — PhoneDT | <https://universe.roboflow.com/phonedt/cigarette-bkjaq> |
| Smoke Detection — ENSAM | <https://universe.roboflow.com/ensam-swz2h/smoke-detection-88ggt> |

Split after merging: **11,056** train / **1,370** validation / **1,868** test images.
The test split was used for qualitative checks only.

## Training setup

| | Faster R-CNN & RetinaNet | YOLO11n |
| --- | --- | --- |
| Framework | Detectron2 | Ultralytics |
| Config | `faster_rcnn_R_50_FPN_1x` / `retinanet_R_50_FPN_1x` | `yolo11n.pt` |
| Schedule | 35,000 iterations (~25 epochs) | 80 epochs, early stop at 72 |
| Batch | 4 × 2 grad-accum | 16 |
| Optimizer | Adam, 1e-3, AMP | Adam, 1e-4 |
| Input | — | 640 × 640 |
| Hardware | RTX 3060 6 GB | RTX 3060 6 GB |

Reproducing the Detectron2 rows requires Detectron2 and the two ~300 MB checkpoints,
which are not in this repository. The training scripts are kept in the paper repository.

## Project layout

```
cigarette-small-object-detection/
├── app.py                     Flask routes, video job runner
├── detector.py                YOLO wrapper (thread-safe) + annotator
├── models/
│   └── cigarette_yolo11n.pt   YOLO11n checkpoint (5.2 MB)
├── templates/index.html       study page + demo
├── static/
│   ├── app.js                 tabs, uploads, polling, camera loop, charts
│   ├── style.css
│   ├── img/                   figures reproduced from the paper
│   ├── paper/                 manuscript PDF
│   └── results/               generated annotated clips
└── requirements.txt
```

## Related work

This is a standalone object-detection study. It is **not** part of the IR video anomaly
detection capstone; the two projects happen to share a workspace, not a pipeline.

## Ethics

Research and education only. The model detects a single narrow class, performs no identity
recognition and no tracking. Do not deploy it on people without consent.
