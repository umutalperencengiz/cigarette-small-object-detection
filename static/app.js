/* Cigarette small-object detection — page controller + YOLO11 demo */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

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

async function postForm(url, formData) {
  const res = await fetch(url, { method: "POST", body: formData });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.error || `Request failed (${res.status})`);
  return payload;
}

/* ----------------------------------------------------------------- health */
(async function loadHealth() {
  try {
    const data = await (await fetch("/api/health")).json();
    const chip = $("#chipModel");
    if (chip) chip.textContent = `model: ${data.model}`;
    setStatus("model ready", "ok");
  } catch (err) {
    setStatus("model unavailable", "bad");
    console.error(err);
  }
})();

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
  hint.textContent = "Running inference…";

  try {
    const fd = new FormData();
    fd.append("image", file, file.name || "frame.jpg");
    fd.append("conf", state.conf);
    fd.append("iou", state.iou);
    fd.append("annotate", "1");

    const data = await postForm("/api/detect", fd);
    const preview = $("#imagePreview");
    preview.src = data.annotated || URL.createObjectURL(file);
    preview.hidden = false;
    $("#imageEmpty").hidden = true;

    $("#imageStats").innerHTML =
      statBox(data.count, "detections", data.count > 0) +
      statBox(`${data.inference_ms} ms`, "inference") +
      statBox(`${data.width}×${data.height}`, "frame");

    $("#imageDetections").innerHTML = data.detections.length
      ? data.detections
          .map(
            (d) =>
              `<li><span>${d.label}</span><span>conf ${fmt(d.confidence)}</span><span>[${d.box
                .map((v) => Math.round(v))
                .join(", ")}]</span></li>`
          )
      : `<li><span>no detections</span><span>try a lower confidence</span></li>`;

    hint.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
  }
}

/* ------------------------------------------------------------------ videos */
wireDropzone("#videoDrop", "#videoInput", (file) => {
  $("#videoHint").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("#videoRun").disabled = false;
});

$("#videoRun").addEventListener("click", async () => {
  const file = $("#videoInput").files[0];
  if (!file) return;

  const btn = $("#videoRun");
  const progress = $("#videoProgress");
  const fill = $("#videoProgressFill");
  const label = $("#videoProgressLabel");
  const hint = $("#videoHint");

  btn.disabled = true;
  hint.className = "hint";
  hint.textContent = "Uploading…";
  progress.hidden = false;
  fill.style.width = "2%";
  label.textContent = "Queued";

  try {
    const fd = new FormData();
    fd.append("video", file, file.name);
    fd.append("conf", state.conf);
    fd.append("iou", state.iou);
    fd.append("fps", $("#videoFps").value);
    fd.append("max_frames", $("#videoMaxFrames").value);

    const { job_id } = await postForm("/api/video", fd);
    pollJob(job_id, { fill, label, hint, btn });
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
    btn.disabled = false;
  }
});

async function pollJob(jobId, ui) {
  try {
    const job = await (await fetch(`/api/job/${jobId}`)).json();
    if (job.error) throw new Error(job.error);

    ui.fill.style.width = `${Math.max(job.progress || 0, 2)}%`;
    ui.label.textContent = job.message || job.status;

    if (job.status === "done") {
      ui.fill.style.width = "100%";
      ui.label.textContent = "Complete";
      ui.hint.textContent = `Analyzed ${job.result.summary.frames_analyzed} frames.`;
      renderVideoResult(job.result);
      ui.btn.disabled = false;
      return;
    }
    setTimeout(() => pollJob(jobId, ui), 800);
  } catch (err) {
    ui.hint.className = "hint error";
    ui.hint.textContent = err.message;
    ui.btn.disabled = false;
  }
}

function renderVideoResult(result) {
  const video = $("#videoResult");
  video.src = result.video_url;
  video.hidden = false;
  $("#videoEmpty").hidden = true;
  video.load();

  const s = result.summary;
  $("#videoStats").innerHTML =
    statBox(s.total_detections, "detections", s.total_detections > 0) +
    statBox(s.frames_with_detection, "frames hit") +
    statBox(`${fmt(s.duration_seconds, 1)} s`, "analyzed") +
    statBox(fmt(s.max_confidence), "max conf", s.max_confidence >= 0.5);

  renderTimeline(result.timeline);
  renderEvents(result.events);
}

function renderTimeline(timeline) {
  const host = $("#videoTimeline");
  host.innerHTML = "";
  if (!timeline.length) return;

  const maxBars = 140;
  const step = Math.max(1, Math.ceil(timeline.length / maxBars));
  for (let i = 0; i < timeline.length; i += step) {
    const sample = timeline[i];
    const bar = document.createElement("div");
    bar.className = `bar${sample.count ? " hit" : ""}`;
    bar.style.height = `${sample.count ? 22 + sample.max_confidence * 78 : 8}%`;
    bar.title = `t=${fmt(sample.t)}s · ${sample.count} hit(s) · ${fmt(sample.max_confidence)}`;
    host.appendChild(bar);
  }
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

  if (!navigator.mediaDevices?.getUserMedia) {
    hint.className = "hint error";
    hint.textContent = "This browser does not expose a camera API. Use localhost or HTTPS.";
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
  setStatus("model ready", "ok");
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

  const blob = await new Promise((resolve) =>
    camera.capture.toBlob(resolve, "image/jpeg", 0.7)
  );

  if (blob) {
    try {
      const t0 = performance.now();
      const fd = new FormData();
      fd.append("image", blob, "frame.jpg");
      fd.append("conf", state.conf);
      fd.append("iou", state.iou);
      fd.append("annotate", "0");
      const data = await postForm("/api/detect", fd);
      const rtt = performance.now() - t0;
      updateCameraUi(data, rtt);
      drawOverlay(data.detections);
    } catch (err) {
      console.warn(err);
    }
  }

  if (state.cameraOn) state.cameraLoop = setTimeout(cameraTick, 90);
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
  ctx.clearRect(0, 0, cw, ch);
  if (!detections?.length) return;

  const sx = cw / 640;
  const sy = ch / 480;
  ctx.lineWidth = Math.max(2, cw / 320);
  ctx.font = `${Math.max(12, cw / 50)}px ui-monospace, Consolas, monospace`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const colour = det.confidence >= 0.5 ? "#35d0a5" : "#ffb454";
    ctx.strokeStyle = colour;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

    const label = `${det.label} ${det.confidence.toFixed(2)}`;
    const textWidth = ctx.measureText(label).width;
    const labelY = Math.max(0, y1 * sy - 22);
    ctx.fillStyle = colour;
    ctx.fillRect(x1 * sx, labelY, textWidth + 10, 20);
    ctx.fillStyle = "#04120e";
    ctx.fillText(label, x1 * sx + 5, labelY + 3);
  }
}

function updateCameraUi(data, rtt) {
  const alert = $("#cameraAlert");
  if (data.count > 0) {
    alert.hidden = false;
    $("#cameraAlertDetail").textContent = `${data.count} region(s) · top ${fmt(
      data.detections[0].confidence
    )}`;
  } else {
    alert.hidden = true;
  }

  const top = data.detections[0]?.confidence || 0;
  $("#cameraMeter").style.width = `${Math.round(top * 100)}%`;

  state.history.push(data.count ? top : 0);
  if (state.history.length > 60) state.history.shift();
  drawSparkline(state.history);

  const fps = 1000 / Math.max(rtt, 1);
  state.fpsSamples.push(fps);
  if (state.fpsSamples.length > 10) state.fpsSamples.shift();
  const avgFps = state.fpsSamples.reduce((a, b) => a + b, 0) / state.fpsSamples.length;

  $("#cameraStats").innerHTML =
    statBox(data.count, "in this frame", data.count > 0) +
    statBox(`${Math.round(rtt)} ms`, "round trip") +
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
