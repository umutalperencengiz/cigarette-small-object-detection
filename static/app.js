/* Cigarette small-object detection — page controller + in-browser YOLO11 demo */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const MODEL_URL = "models/cigarette_yolo11n.onnx";
const INPUT = 640;
const CLASS_NAME = "cigarette";

const state = {
  conf: 0.25,
  iou: 0.45,
  cameraOn: false,
  cameraLoop: null,
  history: [],
  fpsSamples: [],
};

/* ------------------------------------------------------------------ utils */
const fmt = (n, digits = 2) => Number(n).toFixed(digits);

function setStatus(text, kind = "") {
  const pill = $("#statusPill");
  $("#statusText").textContent = text;
  pill.classList.remove("ok", "bad");
  if (kind) pill.classList.add(kind);
}

const statBox = (value, label, accent = false) =>
  `<div class="stat${accent ? " accent" : ""}"><b>${value}</b><span>${label}</span></div>`;

function once(target, event, timeout = 20000) {
  return new Promise((resolve, reject) => {
    let timer = null;
    const done = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("The browser could not decode this file."));
    };
    const cleanup = () => {
      target.removeEventListener(event, done);
      target.removeEventListener("error", fail);
      if (timer) clearTimeout(timer);
    };
    target.addEventListener(event, done);
    target.addEventListener("error", fail);
    if (timeout) {
      timer = setTimeout(() => {
        cleanup();
        reject(new Error("Timed out while reading the file."));
      }, timeout);
    }
  });
}

/* --------------------------------------------------------- inference engine */
const engine = {
  session: null,
  promise: null,
  input: document.createElement("canvas"),
  ctx: null,
};

engine.input.width = INPUT;
engine.input.height = INPUT;
engine.ctx = engine.input.getContext("2d", { willReadFrequently: true });

function loadEngine() {
  if (engine.promise) return engine.promise;

  setStatus("loading model", "");
  $("#chipModel").textContent = "model: yolo11n.onnx";

  engine.promise = (async () => {
    if (typeof ort === "undefined") {
      throw new Error("ONNX Runtime could not be loaded — check your connection and reload the page.");
    }

    // GitHub Pages cannot send COOP/COEP headers, so request the single-threaded WASM path.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    engine.session = session;
    setStatus("model ready", "ok");
    $("#chipModel").textContent = "model: yolo11n.onnx · wasm";
    return session;
  })().catch((err) => {
    engine.promise = null;
    setStatus("model unavailable", "bad");
    console.error(err);
    throw err;
  });

  return engine.promise;
}

loadEngine().catch(() => {});

/** Letterbox the source into 640×640 and normalise to CHW float32. */
function preprocess(source, sw, sh) {
  const canvas = engine.input;
  const ctx = engine.ctx;

  const scale = Math.min(INPUT / sw, INPUT / sh);
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));
  const dx = Math.floor((INPUT - dw) / 2);
  const dy = Math.floor((INPUT - dh) / 2);

  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, INPUT, INPUT);
  ctx.drawImage(source, 0, 0, sw, sh, dx, dy, dw, dh);

  const { data } = ctx.getImageData(0, 0, INPUT, INPUT);
  const plane = INPUT * INPUT;
  const tensor = new Float32Array(plane * 3);

  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    tensor[i] = data[p] / 255;
    tensor[plane + i] = data[p + 1] / 255;
    tensor[plane * 2 + i] = data[p + 2] / 255;
  }

  return { tensor, meta: { scale, dx, dy, sw, sh } };
}

function boxIou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (!inter) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

/** Decode the [1, 4+classes, anchors] output, threshold and run NMS. */
function decode(raw, meta, confThreshold, iouThreshold) {
  const anchors = raw.dims[2];
  const channels = raw.dims[1];
  const out = raw.data;

  const boxes = [];
  const scores = [];

  for (let a = 0; a < anchors; a += 1) {
    let best = 0;
    for (let c = 4; c < channels; c += 1) {
      const score = out[c * anchors + a];
      if (score > best) best = score;
    }
    if (best < confThreshold) continue;

    const cx = out[a];
    const cy = out[anchors + a];
    const w = out[anchors * 2 + a];
    const h = out[anchors * 3 + a];

    boxes.push([
      (cx - w / 2 - meta.dx) / meta.scale,
      (cy - h / 2 - meta.dy) / meta.scale,
      (cx + w / 2 - meta.dx) / meta.scale,
      (cy + h / 2 - meta.dy) / meta.scale,
    ]);
    scores.push(best);
  }

  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const kept = [];

  while (order.length) {
    const index = order.shift();
    kept.push(index);
    for (let k = order.length - 1; k >= 0; k -= 1) {
      if (boxIou(boxes[index], boxes[order[k]]) > iouThreshold) order.splice(k, 1);
    }
  }

  return kept.map((index) => {
    const [x1, y1, x2, y2] = boxes[index];
    return {
      label: CLASS_NAME,
      confidence: scores[index],
      box: [
        Math.max(0, x1),
        Math.max(0, y1),
        Math.min(meta.sw, x2),
        Math.min(meta.sh, y2),
      ],
    };
  });
}

async function detect(source, sw, sh) {
  const session = await loadEngine();
  const started = performance.now();

  const { tensor, meta } = preprocess(source, sw, sh);
  const feeds = {
    [session.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, INPUT, INPUT]),
  };

  const output = await session.run(feeds);
  const raw = output[session.outputNames[0]];
  const detections = decode(raw, meta, state.conf, state.iou);

  return {
    detections,
    inference_ms: Math.round(performance.now() - started),
    width: sw,
    height: sh,
  };
}

/* ---------------------------------------------------------------- drawing */
function drawLabels(ctx, detections, srcW, srcH, targetW, targetH, lineWidth, fontPx) {
  const sx = targetW / srcW;
  const sy = targetH / srcH;

  ctx.lineWidth = lineWidth;
  ctx.font = `${fontPx}px ui-monospace, Consolas, monospace`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const X = x1 * sx;
    const Y = y1 * sy;
    const W = (x2 - x1) * sx;
    const H = (y2 - y1) * sy;
    const colour = det.confidence >= 0.5 ? "#35d0a5" : "#ffb454";

    ctx.strokeStyle = colour;
    ctx.strokeRect(X, Y, W, H);

    const label = `${det.label} ${det.confidence.toFixed(2)}`;
    const textWidth = ctx.measureText(label).width;
    const labelY = Math.max(0, Y - fontPx - 6);

    ctx.fillStyle = colour;
    ctx.fillRect(X, labelY, textWidth + 10, fontPx + 6);
    ctx.fillStyle = "#04120e";
    ctx.fillText(label, X + 5, labelY + 3);
  }
}

/** Draw source + boxes into a data URL so the <img> needs no extra layers. */
function annotate(source, sw, sh, detections) {
  const maxSide = 1280;
  const scale = Math.min(1, maxSide / Math.max(sw, sh));

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(sw * scale);
  canvas.height = Math.round(sh * scale);

  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, sw, sh, 0, 0, canvas.width, canvas.height);
  drawLabels(
    ctx,
    detections,
    sw,
    sh,
    canvas.width,
    canvas.height,
    Math.max(2, canvas.width / 320),
    Math.max(12, canvas.width / 46)
  );

  return canvas.toDataURL("image/jpeg", 0.9);
}

/* ----------------------------------------------------------------- charts */
const MODELS = [
  { name: "Faster R-CNN", short: "FR-CNN", color: "#3b9bff", ap: 43.635, ap50: 87.245, ap75: 38.951, aps: 27.405, hours: 11 },
  { name: "RetinaNet", short: "RetinaNet", color: "#ffb454", ap: 41.762, ap50: 84.665, ap75: 35.705, aps: 24.304, hours: 9 },
  { name: "YOLO11", short: "YOLO11", color: "#35d0a5", ap: 40.4, ap50: 88.5, ap75: 35.2, aps: null, hours: 3.5 },
];

const METRICS = [
  { key: "ap", label: "AP" },
  { key: "ap50", label: "AP50" },
  { key: "ap75", label: "AP75" },
  { key: "aps", label: "APs" },
];

const AXIS = "#8b9bb0";
const GRID = "#1d2734";
const MONO = '11px "JetBrains Mono", Consolas, monospace';

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width, 10);
  const h = Math.max(rect.height, 10);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawApChart() {
  const canvas = $("#apChart");
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);

  const pad = { top: 12, right: 8, bottom: 28, left: 34 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const toY = (v) => pad.top + plotH - (v / 100) * plotH;

  ctx.font = MONO;
  ctx.textBaseline = "middle";

  // grid + y ticks
  for (let v = 0; v <= 100; v += 25) {
    const y = toY(v);
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y + 0.5);
    ctx.lineTo(pad.left + plotW, y + 0.5);
    ctx.stroke();

    ctx.fillStyle = AXIS;
    ctx.textAlign = "right";
    ctx.fillText(String(v), pad.left - 7, y);
  }

  const groupW = plotW / METRICS.length;
  const inner = groupW * 0.72;
  const barW = inner / 3;

  ctx.textAlign = "center";
  METRICS.forEach((metric, gi) => {
    const gx = pad.left + gi * groupW + (groupW - inner) / 2;

    MODELS.forEach((model, mi) => {
      const value = model[metric.key];
      const x = gx + mi * barW;
      if (value == null) {
        ctx.fillStyle = "#1a2431";
        ctx.fillRect(x + 1, pad.top + plotH - 3, barW - 2, 3);
        return;
      }
      const y = toY(value);
      ctx.fillStyle = model.color;
      ctx.globalAlpha = 0.92;
      ctx.fillRect(x + 1, y, Math.max(barW - 2, 2), pad.top + plotH - y);
      ctx.globalAlpha = 1;
    });

    ctx.fillStyle = AXIS;
    ctx.fillText(metric.label, pad.left + gi * groupW + groupW / 2, h - 11);
  });

  ctx.strokeStyle = GRID;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotH + 0.5);
  ctx.lineTo(pad.left + plotW, pad.top + plotH + 0.5);
  ctx.stroke();
}

function drawTimeChart() {
  const canvas = $("#timeChart");
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);

  const pad = { top: 14, right: 14, bottom: 30, left: 34 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const maxHours = 12;
  const maxAp = 50;

  const toX = (v) => pad.left + (v / maxHours) * plotW;
  const toY = (v) => pad.top + plotH - (v / maxAp) * plotH;

  ctx.font = MONO;
  ctx.textBaseline = "middle";

  // horizontal grid
  for (let v = 0; v <= maxAp; v += 10) {
    const y = toY(v);
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y + 0.5);
    ctx.lineTo(pad.left + plotW, y + 0.5);
    ctx.stroke();

    ctx.fillStyle = AXIS;
    ctx.textAlign = "right";
    ctx.fillText(String(v), pad.left - 7, y);
  }

  // vertical grid + x ticks (hours)
  for (let hrs = 0; hrs <= maxHours; hrs += 3) {
    const x = toX(hrs);
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, pad.top);
    ctx.lineTo(x + 0.5, pad.top + plotH);
    ctx.stroke();

    ctx.fillStyle = AXIS;
    ctx.textAlign = "center";
    ctx.fillText(`${hrs}h`, x, h - 12);
  }

  // axis labels
  // (no inline y-axis title: the section caption already states the metric)

  MODELS.forEach((model) => {
    const x = toX(model.hours);
    const y = toY(model.ap);

    ctx.strokeStyle = model.color;
    ctx.globalAlpha = 0.22;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.fillStyle = model.color;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.textAlign = "center";
    ctx.fillStyle = "#e7eef7";
    ctx.fillText(model.short, x, y + 13);

    ctx.fillStyle = "#5c6b7f";
    ctx.fillText(`${model.hours} h`, x, y + 25);
  });
}

function drawCharts() {
  drawApChart();
  drawTimeChart();
}

let chartResizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(chartResizeTimer);
  chartResizeTimer = setTimeout(drawCharts, 120);
  if (state.cameraOn) resizeOverlay();
  if (videoUi.samples.length) drawVideoOverlay();
});

if (document.fonts?.ready) {
  document.fonts.ready.then(drawCharts).catch(() => {});
}
drawCharts();

/* ----------------------------------------------------------------- sliders */
$("#conf").addEventListener("input", (e) => {
  state.conf = Number(e.target.value);
  $("#confValue").textContent = fmt(state.conf);
});
$("#iou").addEventListener("input", (e) => {
  state.iou = Number(e.target.value);
  $("#iouValue").textContent = fmt(state.iou);
});
$("#videoFps").addEventListener("input", (e) => {
  $("#fpsValue").textContent = e.target.value;
});
$("#videoMaxFrames").addEventListener("input", (e) => {
  $("#maxFramesValue").textContent = e.target.value;
});

/* -------------------------------------------------------------------- tabs */
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => {
      t.classList.remove("is-active");
      t.setAttribute("aria-selected", "false");
    });
    $$(".panel").forEach((p) => p.classList.remove("is-active"));
    tab.classList.add("is-active");
    tab.setAttribute("aria-selected", "true");
    $(`#tab-${tab.dataset.tab}`).classList.add("is-active");

    if (tab.dataset.tab === "camera" && state.cameraOn) startCameraLoop();
    if (tab.dataset.tab !== "camera") stopCameraLoop();
    if (tab.dataset.tab === "video" && videoUi.samples.length) drawVideoOverlay();
  });
});

/* ------------------------------------------------------------- drag & drop */
function wireDropzone(zoneId, inputId, onFile) {
  const zone = $(zoneId);
  const input = $(inputId);

  input.addEventListener("change", (e) => {
    if (e.target.files[0]) onFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("is-over");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
}

/* ------------------------------------------------------------------ images */
wireDropzone("#imageDrop", "#imageInput", handleImage);

async function handleImage(file) {
  const hint = $("#imageHint");
  hint.className = "hint";
  hint.textContent = "Loading model…";

  try {
    const image = await createImageBitmap(file);
    const sw = image.width;
    const sh = image.height;

    await loadEngine();
    hint.textContent = "Running inference…";

    const result = await detect(image, sw, sh);

    const preview = $("#imagePreview");
    preview.src = annotate(image, sw, sh, result.detections);
    preview.hidden = false;
    $("#imageEmpty").hidden = true;

    $("#imageStats").innerHTML =
      statBox(result.detections.length, "detections", result.detections.length > 0) +
      statBox(`${result.inference_ms} ms`, "inference") +
      statBox(`${sw}×${sh}`, "frame");

    $("#imageDetections").innerHTML = result.detections.length
      ? result.detections
          .map(
            (d) =>
              `<li><span>${d.label}</span><span>conf ${fmt(d.confidence)}</span><span>[${d.box
                .map((v) => Math.round(v))
                .join(", ")}]</span></li>`
          )
          .join("")
      : `<li><span>no detections</span><span>try a lower confidence</span></li>`;

    hint.textContent = `${file.name || "image"} · ${(file.size / 1024).toFixed(0)} KB`;
    image.close?.();
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
    console.error(err);
  }
}

$("#imageSample").addEventListener("click", async () => {
  const hint = $("#imageHint");
  try {
    hint.className = "hint";
    hint.textContent = "Loading sample…";
    const res = await fetch("static/samples/frame.jpg");
    if (!res.ok) throw new Error(`Sample frame not found (${res.status}).`);
    const blob = await res.blob();
    await handleImage(new File([blob], "sample-frame.jpg", { type: blob.type || "image/jpeg" }));
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
  }
});

/* ------------------------------------------------------------------ videos */
const videoUi = { samples: [], url: null, file: null };

wireDropzone("#videoDrop", "#videoInput", (file) => {
  videoUi.file = file;
  $("#videoHint").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("#videoRun").disabled = false;
});

$("#videoSample").addEventListener("click", async () => {
  const hint = $("#videoHint");
  try {
    hint.className = "hint";
    hint.textContent = "Loading sample clip…";
    const res = await fetch("static/samples/clip.mp4");
    if (!res.ok) throw new Error(`Sample clip not found (${res.status}).`);
    const blob = await res.blob();
    videoUi.file = new File([blob], "sample-clip.mp4", { type: "video/mp4" });
    hint.textContent = `sample-clip.mp4 · ${(blob.size / 1024).toFixed(0)} KB`;
    $("#videoRun").disabled = false;
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
  }
});

$("#videoRun").addEventListener("click", runVideoAnalysis);

function seekVideo(video, time) {
  return new Promise((resolve, reject) => {
    const done = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("The browser could not seek in this clip."));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", fail);
    };
    video.addEventListener("seeked", done);
    video.addEventListener("error", fail);
    video.currentTime = Math.min(Math.max(time, 0), Math.max(video.duration - 0.05, 0));
  });
}

async function runVideoAnalysis() {
  const file = videoUi.file;
  if (!file) return;

  const btn = $("#videoRun");
  const progress = $("#videoProgress");
  const fill = $("#videoProgressFill");
  const label = $("#videoProgressLabel");
  const hint = $("#videoHint");
  const video = $("#videoResult");
  const stage = $("#videoFrame");
  const overlay = $("#videoOverlay");

  btn.disabled = true;
  hint.className = "hint";
  hint.textContent = "Loading model…";
  progress.hidden = false;
  fill.style.width = "2%";
  label.textContent = "Loading model";

  try {
    await loadEngine();

    if (videoUi.url) URL.revokeObjectURL(videoUi.url);
    videoUi.samples = [];
    videoUi.url = URL.createObjectURL(file);
    video.src = videoUi.url;
    video.muted = true;

    stage.hidden = false;
    $("#videoEmpty").hidden = true;
    $("#videoStats").innerHTML = "";
    $("#videoTimeline").innerHTML = "";
    $("#videoEvents").innerHTML = "";
    overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);

    await once(video, "loadedmetadata");
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      throw new Error("That clip has no readable duration.");
    }
    if (duration > 180) {
      throw new Error(`That clip is ${fmt(duration, 0)} s long — trim it to 3 minutes or less.`);
    }

    const fps = Number($("#videoFps").value);
    const maxFrames = Number($("#videoMaxFrames").value);
    const count = Math.max(1, Math.min(maxFrames, Math.floor(duration * fps)));

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);

    hint.textContent = `Analyzing ${count} frames…`;
    const started = performance.now();

    for (let i = 0; i < count; i += 1) {
      const t = ((i + 0.5) * duration) / count;
      await seekVideo(video, t);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const result = await detect(canvas, canvas.width, canvas.height);
      const peak = result.detections.reduce((max, d) => Math.max(max, d.confidence), 0);

      videoUi.samples.push({
        t,
        count: result.detections.length,
        maxConfidence: peak,
        detections: result.detections.map((d) => ({
          label: d.label,
          confidence: d.confidence,
          box: [
            d.box[0] / canvas.width,
            d.box[1] / canvas.height,
            d.box[2] / canvas.width,
            d.box[3] / canvas.height,
          ],
        })),
      });

      const pct = Math.round(((i + 1) / count) * 100);
      fill.style.width = `${pct}%`;
      label.textContent = `Frame ${i + 1} / ${count} · ${pct}%`;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const elapsed = (performance.now() - started) / 1000;
    fill.style.width = "100%";
    label.textContent = `Complete · ${fmt(elapsed, 1)} s`;
    hint.textContent = `${count} frames sampled at ${fps} fps. Press play to watch the overlay.`;

    video.currentTime = 0;
    renderVideoResult(duration);
    drawVideoOverlay();

    video.addEventListener("timeupdate", drawVideoOverlay);
    video.addEventListener("seeked", drawVideoOverlay);
    video.addEventListener("play", drawVideoOverlay);
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
    label.textContent = "Stopped";
    stage.hidden = true;
    $("#videoEmpty").hidden = false;
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

function nearestSample(time) {
  const samples = videoUi.samples;
  if (!samples.length) return null;

  let lo = 0;
  let hi = samples.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t < time) lo = mid + 1;
    else hi = mid;
  }

  const before = samples[Math.max(0, lo - 1)];
  const after = samples[lo];
  return Math.abs(before.t - time) <= Math.abs(after.t - time) ? before : after;
}

function drawVideoOverlay() {
  const video = $("#videoResult");
  const canvas = $("#videoOverlay");
  if (video.hidden || !video.videoWidth) return;

  const rect = video.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const sample = nearestSample(video.currentTime);
  if (!sample || !sample.detections.length) return;

  const boxes = sample.detections.map((d) => ({
    label: d.label,
    confidence: d.confidence,
    box: [
      d.box[0] * rect.width,
      d.box[1] * rect.height,
      d.box[2] * rect.width,
      d.box[3] * rect.height,
    ],
  }));

  drawLabels(
    ctx,
    boxes,
    rect.width,
    rect.height,
    rect.width,
    rect.height,
    Math.max(2, rect.width / 320),
    Math.max(12, rect.width / 46)
  );
}

function renderVideoResult(duration) {
  const samples = videoUi.samples;
  const total = samples.reduce((sum, s) => sum + s.count, 0);
  const hit = samples.filter((s) => s.count > 0).length;
  const peak = samples.reduce((max, s) => Math.max(max, s.maxConfidence), 0);

  $("#videoStats").innerHTML =
    statBox(total, "detections", total > 0) +
    statBox(hit, "frames hit") +
    statBox(`${fmt(duration, 1)} s`, "clip") +
    statBox(fmt(peak), "max conf", peak >= 0.5);

  renderTimeline(samples);
  renderEvents(eventsFromSamples(samples));
}

function renderTimeline(samples) {
  const host = $("#videoTimeline");
  host.innerHTML = "";
  if (!samples.length) return;

  const maxBars = 140;
  const step = Math.max(1, Math.ceil(samples.length / maxBars));
  for (let i = 0; i < samples.length; i += step) {
    const sample = samples[i];
    const bar = document.createElement("div");
    bar.className = `bar${sample.count ? " hit" : ""}`;
    bar.style.height = `${sample.count ? 22 + sample.maxConfidence * 78 : 8}%`;
    bar.title = `t=${fmt(sample.t)}s · ${sample.count} hit(s) · ${fmt(sample.maxConfidence)}`;
    host.appendChild(bar);
  }
}

function eventsFromSamples(samples) {
  const step = samples.length > 1 ? samples[1].t - samples[0].t : 0.33;
  const events = [];
  let current = null;

  samples.forEach((sample) => {
    if (!sample.count) {
      current = null;
      return;
    }
    if (current && sample.t - current.lastT <= step * 1.6) {
      current.end = Math.min(sample.t + step / 2, current.end + step);
      current.hits += sample.count;
      current.max = Math.max(current.max, sample.maxConfidence);
      current.lastT = sample.t;
    } else {
      current = {
        start: Math.max(0, sample.t - step / 2),
        end: sample.t + step / 2,
        hits: sample.count,
        max: sample.maxConfidence,
        lastT: sample.t,
      };
      events.push(current);
    }
  });

  return events.map(({ start, end, hits, max }) => ({
    start,
    end,
    hits,
    max_confidence: max,
  }));
}

function renderEvents(events) {
  const host = $("#videoEvents");
  if (!events.length) {
    host.innerHTML = `<li><span>no events detected</span><span>—</span></li>`;
    return;
  }
  host.innerHTML = events
    .map(
      (e) =>
        `<li><span>${fmt(e.start, 1)}s → ${fmt(e.end, 1)}s</span><span>${e.hits} hit(s)</span><span>max ${fmt(
          e.max_confidence
        )}</span></li>`
    )
    .join("");
}

/* ------------------------------------------------------------------ camera */
const camera = {
  video: $("#cameraVideo"),
  overlay: $("#cameraOverlay"),
  capture: document.createElement("canvas"),
  placeholder: $("#cameraPlaceholder"),
  stream: null,
};

camera.capture.width = 640;
camera.capture.height = 480;

$("#cameraStart").addEventListener("click", startCamera);
$("#cameraStop").addEventListener("click", stopCamera);

async function startCamera() {
  const hint = $("#cameraHint");
  hint.className = "hint";

  if (!window.isSecureContext) {
    hint.className = "hint error";
    hint.textContent =
      "Camera access needs HTTPS (or localhost). Open the GitHub Pages address of this page and try again.";
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    hint.className = "hint error";
    hint.textContent = "This browser does not expose a camera API.";
    return;
  }

  try {
    camera.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" },
      audio: false,
    });
    camera.video.srcObject = camera.stream;
    await camera.video.play();
    camera.placeholder.hidden = true;
    state.cameraOn = true;
    state.history = [];
    state.fpsSamples = [];
    $("#cameraStart").disabled = true;
    $("#cameraStop").disabled = false;
    setStatus("camera live", "ok");
    startCameraLoop();
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = `Camera error: ${err.message}`;
  }
}

function stopCamera() {
  state.cameraOn = false;
  stopCameraLoop();
  if (camera.stream) {
    camera.stream.getTracks().forEach((t) => t.stop());
    camera.stream = null;
  }
  camera.video.srcObject = null;
  camera.placeholder.hidden = false;
  $("#cameraStart").disabled = false;
  $("#cameraStop").disabled = true;
  $("#cameraAlert").hidden = true;
  clearOverlay();
  setStatus(engine.session ? "model ready" : "loading model", engine.session ? "ok" : "");
}

function stopCameraLoop() {
  if (state.cameraLoop) {
    clearTimeout(state.cameraLoop);
    state.cameraLoop = null;
  }
}

function startCameraLoop() {
  stopCameraLoop();
  if (!state.cameraOn) return;
  state.cameraLoop = setTimeout(cameraTick, 60);
}

/** Center-crop the source to the 4:3 capture aspect. */
function cropToAspect(video) {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const target = 4 / 3;
  let sw = vw;
  let sh = vh;
  let sx = 0;
  let sy = 0;
  if (vw / vh > target) {
    sw = vh * target;
    sx = (vw - sw) / 2;
  } else {
    sh = vw / target;
    sy = (vh - sh) / 2;
  }
  return { sx, sy, sw, sh };
}

async function cameraTick() {
  if (!state.cameraOn) return;
  const video = camera.video;
  if (video.readyState < 2) {
    state.cameraLoop = setTimeout(cameraTick, 150);
    return;
  }

  const ctx = camera.capture.getContext("2d");
  const { sx, sy, sw, sh } = cropToAspect(video);
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 640, 480);

  try {
    const result = await detect(camera.capture, 640, 480);
    updateCameraUi(result, result.inference_ms);
    drawOverlay(result.detections);
  } catch (err) {
    console.warn(err);
  }

  if (state.cameraOn) state.cameraLoop = setTimeout(cameraTick, 30);
}

function resizeOverlay() {
  const rect = camera.overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  camera.overlay.width = Math.max(1, Math.round(rect.width * dpr));
  camera.overlay.height = Math.max(1, Math.round(rect.height * dpr));
}

function clearOverlay() {
  const ctx = camera.overlay.getContext("2d");
  ctx.clearRect(0, 0, camera.overlay.width, camera.overlay.height);
}

function drawOverlay(detections) {
  resizeOverlay();
  const ctx = camera.overlay.getContext("2d");
  const { width: cw, height: ch } = camera.overlay;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!detections?.length) return;

  drawLabels(ctx, detections, 640, 480, cw, ch, Math.max(2, cw / 320), Math.max(12, cw / 50));
}

function updateCameraUi(data, inferenceMs) {
  const alert = $("#cameraAlert");
  if (data.detections.length > 0) {
    alert.hidden = false;
    $("#cameraAlertDetail").textContent = `${data.detections.length} region(s) · top ${fmt(
      data.detections[0].confidence
    )}`;
  } else {
    alert.hidden = true;
  }

  const top = data.detections[0]?.confidence || 0;
  $("#cameraMeter").style.width = `${Math.round(top * 100)}%`;

  state.history.push(data.detections.length ? top : 0);
  if (state.history.length > 60) state.history.shift();
  drawSparkline(state.history);

  const fps = 1000 / Math.max(inferenceMs, 1);
  state.fpsSamples.push(fps);
  if (state.fpsSamples.length > 10) state.fpsSamples.shift();
  const avgFps = state.fpsSamples.reduce((a, b) => a + b, 0) / state.fpsSamples.length;

  $("#cameraStats").innerHTML =
    statBox(data.detections.length, "in this frame", data.detections.length > 0) +
    statBox(`${Math.round(inferenceMs)} ms`, "inference") +
    statBox(`${fmt(avgFps, 1)} fps`, "analyzed") +
    statBox(fmt(top), "top confidence", top >= 0.5);

  $("#cameraDetections").innerHTML = data.detections.length
    ? data.detections
        .map(
          (d) =>
            `<li><span>${d.label}</span><span>conf ${fmt(d.confidence)}</span><span>${Math.round(
              d.box[2] - d.box[0]
            )}×${Math.round(d.box[3] - d.box[1])} px</span></li>`
        )
        .join("")
    : `<li><span>clear</span><span>no cigarette detected</span></li>`;
}

function drawSparkline(values) {
  const canvas = $("#cameraSparkline");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (values.length < 2) return;
  const step = w / Math.max(values.length - 1, 1);

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(53, 208, 165, 0.45)");
  grad.addColorStop(1, "rgba(53, 208, 165, 0)");

  ctx.beginPath();
  ctx.moveTo(0, h - values[0] * (h - 8));
  values.forEach((v, i) => ctx.lineTo(i * step, h - v * (h - 8)));
  ctx.lineTo((values.length - 1) * step, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  values.forEach((v, i) => {
    const x = i * step;
    const y = h - v * (h - 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#35d0a5";
  ctx.lineWidth = 2;
  ctx.stroke();
}
