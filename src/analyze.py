"""
AniVision Prototype
-------------------
Animation detection in video using CLIP + OpenCV + Whisper + Mistral.
Built as a proof of concept for the AniVision research project.
"""

import os
import threading

import cv2
import clip
import whisper
import ollama
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template_string, request, send_file
import torch


# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

app = Flask(__name__)

state = {
    "video_path": "",
    "results": [],
    "report": "",
    "status": {
        "stage": "Waiting for video...",
        "progress": 0,
        "total": 0,
        "done": False,
        "report_done": False,
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

print("Loading CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
LABELS = ["animation", "live action footage"]
label_tokens = clip.tokenize(LABELS).to(DEVICE)

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Ready → http://localhost:5000")


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def classify_frame(frame: np.ndarray) -> tuple[str, float]:
    """Classify a single frame using CLIP + OpenCV visual features."""

    # CLIP score
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = clip_preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = clip_model(image, label_tokens)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    clip_anim_score = float(probs[0])

    # OpenCV visual features
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise     = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges     = cv2.Canny(gray, 50, 150).mean()
    small     = cv2.resize(frame, (64, 64)).reshape(-1, 3)
    n_colors  = len(set(map(tuple, small)))

    noise_score = 1.0 / (1.0 + noise / 500)
    color_score = 1.0 / (1.0 + n_colors / 800)
    edge_score  = min(edges / 30, 1.0)
    cv_score    = noise_score * 0.4 + color_score * 0.4 + edge_score * 0.2

    combined = clip_anim_score * 0.5 + cv_score * 0.5
    label      = "animation" if combined > 0.5 else "live action footage"
    confidence = float(max(combined, 1.0 - combined))
    return label, confidence


def smooth(raw: list, window: int = 8) -> list:
    """Smooth per-frame labels using a confidence-weighted sliding window."""
    smoothed = []
    for i in range(len(raw)):
        chunk = raw[max(0, i - window): i + window + 1]
        anim_w = sum(c for _, l, c in chunk if l == "animation")
        live_w = sum(c for _, l, c in chunk if l == "live action footage")
        label  = "animation" if anim_w >= live_w else "live action footage"
        conf   = max(anim_w, live_w) / len(chunk)
        smoothed.append((raw[i][0], label, conf))
    return smoothed


def to_segments(smoothed: list, min_duration: float = 2.0) -> list:
    """Convert per-frame labels into timed segments, merging short ones."""
    if not smoothed:
        return []

    segments, cur_label, cur_start, cur_confs = [], smoothed[0][1], smoothed[0][0], [smoothed[0][2]]
    for ts, label, conf in smoothed[1:]:
        if label != cur_label:
            segments.append({"start": cur_start, "end": ts, "label": cur_label,
                              "confidence": round(float(np.mean(cur_confs)), 2)})
            cur_label, cur_start, cur_confs = label, ts, [conf]
        else:
            cur_confs.append(conf)
    segments.append({"start": cur_start, "end": smoothed[-1][0], "label": cur_label,
                     "confidence": round(float(np.mean(cur_confs)), 2)})

    # Merge segments shorter than min_duration into previous
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg["end"] - seg["start"] < min_duration:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def attach_transcripts(segments: list, words: list) -> None:
    """Attach Whisper word-level transcripts to each segment (in-place)."""
    for seg in segments:
        seg["transcript"] = " ".join(
            w["word"] for w in words if seg["start"] <= w["start"] < seg["end"]
        ).strip()


def fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

REPORT_PROMPT = """You are an analyst writing intelligence-style film reports for animation researchers.

Given the data below, produce a SHORT structured report. Use this exact format:

FILM TYPE: one line
PURPOSE: one line
ANIMATION USAGE: one line
ESTIMATED PERIOD: one line
LANGUAGE: one line
ANIMATION SHARE: {anim_pct}% of total runtime

SUMMARY:
Two or three sentences. Factual, no speculation beyond what the transcript supports.

RESEARCH NOTES:
One or two sentences on what is relevant for animation scholarship.

---
Duration: {duration}
Segments detected: {n_segments}
Animation segments: {n_anim} / Live action segments: {n_live}

Transcript excerpt:
{transcript}
"""


def generate_report(segments: list, transcript: str) -> None:
    """Call Mistral via Ollama and store the result in global state."""
    state["status"]["stage"] = "Generating report..."

    anim = [s for s in segments if s["label"] == "animation"]
    live = [s for s in segments if s["label"] == "live action footage"]
    total = segments[-1]["end"] if segments else 1
    anim_dur = sum(s["end"] - s["start"] for s in anim)

    prompt = REPORT_PROMPT.format(
        anim_pct   = int(anim_dur / total * 100),
        duration   = fmt(total),
        n_segments = len(segments),
        n_anim     = len(anim),
        n_live     = len(live),
        transcript = transcript[:2000],
    )

    try:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        state["report"] = response["message"]["content"]
    except Exception as exc:
        state["report"] = f"Report generation failed: {exc}"

    state["status"]["report_done"] = True


# ---------------------------------------------------------------------------
# Main analysis thread
# ---------------------------------------------------------------------------

def run_analysis(video_path: str) -> None:
    """Full pipeline: transcribe → classify frames → segment → report."""
    s = state["status"]
    state["results"] = []
    state["report"]  = ""
    s.update({"stage": "Transcribing audio...", "progress": 0, "total": 0,
               "done": False, "report_done": False})

    # 1. Transcription
    try:
        result = whisper_model.transcribe(video_path, word_timestamps=True)
    except Exception as exc:
        s["stage"] = f"Transcription failed: {exc}"
        return

    words = [
        {"word": w["word"], "start": w["start"], "end": w["end"]}
        for seg in result["segments"] if "words" in seg
        for w in seg["words"]
    ]
    full_transcript = " ".join(seg["text"] for seg in result["segments"])

    # 2. Frame classification
    s["stage"] = "Analyzing frames..."
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, int(fps * 0.5))
    s["total"]   = total_frames // sample_every

    raw, frame_idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            label, conf = classify_frame(frame)
            raw.append((frame_idx / fps, label, conf))
            s["progress"] = len(raw)
        frame_idx += 1
    cap.release()

    # 3. Segmentation
    s["stage"] = "Building segments..."
    segments = to_segments(smooth(raw), min_duration=2.0)
    attach_transcripts(segments, words)
    state["results"] = segments
    s["done"] = True

    # 4. Report
    generate_report(segments, full_transcript)


# ---------------------------------------------------------------------------
# HTML interface
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html>
<head>
<title>AniVision</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#eee;min-height:100vh}
  .header{background:#16213e;padding:20px 32px;border-bottom:1px solid #0f3460}
  .header h1{font-size:1.3em;color:#e94560}
  .header p{font-size:.8em;color:#888;margin-top:4px}
  .main{display:grid;grid-template-columns:1fr 400px;height:calc(100vh - 65px)}
  .left{padding:24px;overflow-y:auto}
  .right{background:#16213e;border-left:1px solid #0f3460;overflow-y:auto}
  .upload{border:2px dashed #0f3460;border-radius:12px;padding:48px;text-align:center;cursor:pointer;transition:.2s;margin-bottom:20px}
  .upload:hover,.upload.drag{border-color:#e94560;background:rgba(233,69,96,.05)}
  .upload h3{color:#e94560;margin-bottom:8px}
  .upload p{color:#666;font-size:.85em}
  #file-input{display:none}
  video{width:100%;border-radius:10px;display:none;margin-bottom:14px}
  .progress-wrap{background:#0f3460;border-radius:4px;height:4px;margin:12px 0 6px}
  .progress-bar{background:#e94560;height:4px;border-radius:4px;transition:width .4s;width:0%}
  #status{font-size:.8em;color:#666;margin-bottom:14px}
  .stats{display:flex;gap:10px;margin-bottom:14px}
  .stat{background:#16213e;border-radius:8px;padding:10px 14px;flex:1;text-align:center}
  .stat-v{font-size:1.4em;font-weight:bold;color:#e94560}
  .stat-l{font-size:.7em;color:#666;margin-top:2px}
  .seg{display:flex;flex-direction:column;padding:10px 14px;margin:4px 0;border-radius:8px;cursor:pointer;transition:transform .1s;border:2px solid transparent}
  .seg:hover{transform:translateX(3px)}
  .seg.active{border-color:rgba(255,255,255,.35)}
  .animation{background:#1b4332}
  .live{background:#1e3a5f}
  .seg-row{display:flex;align-items:center;gap:10px}
  .seg-label{font-weight:bold;font-size:.85em;flex:1}
  .seg-time{font-size:.8em;color:#aaa}
  .seg-conf{font-size:.75em;background:rgba(255,255,255,.1);padding:2px 7px;border-radius:8px;color:#ccc}
  .seg-text{margin-top:6px;font-size:.75em;color:#aaa;line-height:1.4;border-top:1px solid rgba(255,255,255,.08);padding-top:6px;font-style:italic}
  .seg-text:empty{display:none}
  .rpt-header{padding:16px 20px;border-bottom:1px solid #0f3460}
  .rpt-header h2{font-size:.95em;color:#e94560}
  .rpt-body{padding:16px 20px;font-size:.82em;line-height:1.75;color:#ccc;white-space:pre-wrap}
  .rpt-placeholder{padding:40px 20px;color:#444;font-size:.82em;text-align:center}
</style>
</head>
<body>
<div class="header">
  <h1>🎬 AniVision Prototype</h1>
  <p>CLIP · OpenCV · Whisper · Mistral</p>
</div>
<div class="main">
  <div class="left">
    <div class="upload" id="upload" onclick="document.getElementById('file-input').click()">
      <h3>Drop video here or click to upload</h3>
      <p>MP4 · AVI · MOV</p>
    </div>
    <input type="file" id="file-input" accept="video/*">
    <video id="player" controls></video>
    <div id="stats" class="stats" style="display:none">
      <div class="stat"><div class="stat-v" id="s-total">—</div><div class="stat-l">Segments</div></div>
      <div class="stat"><div class="stat-v" id="s-anim">—</div><div class="stat-l">Animation</div></div>
      <div class="stat"><div class="stat-v" id="s-live">—</div><div class="stat-l">Live Action</div></div>
    </div>
    <div class="progress-wrap"><div class="progress-bar" id="bar"></div></div>
    <div id="status">Upload a video to begin</div>
    <div id="segments"></div>
  </div>
  <div class="right">
    <div class="rpt-header"><h2>📋 Research Report</h2></div>
    <div id="report"><div class="rpt-placeholder">Report will appear after analysis</div></div>
  </div>
</div>
<script>
const player = document.getElementById('player');
let segs = [], polling = false;

const zone = document.getElementById('upload');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
document.getElementById('file-input').addEventListener('change', e => { if (e.target.files[0]) upload(e.target.files[0]); });

function upload(file) {
  const fd = new FormData(); fd.append('video', file);
  document.getElementById('status').textContent = 'Uploading...';
  document.getElementById('segments').innerHTML = '';
  document.getElementById('report').innerHTML = '<div class="rpt-placeholder">Waiting...</div>';
  document.getElementById('stats').style.display = 'none';
  fetch('/upload', { method: 'POST', body: fd }).then(r => r.json()).then(d => {
    if (d.ok) {
      player.src = '/video?' + Date.now();
      player.style.display = 'block';
      zone.style.display = 'none';
      if (!polling) { polling = true; poll(); }
    }
  });
}

const fmt = s => Math.floor(s/60).toString().padStart(2,'0') + ':' + Math.floor(s%60).toString().padStart(2,'0');

function poll() {
  fetch('/api/results').then(r => r.json()).then(d => {
    document.getElementById('bar').style.width = (d.total > 0 ? d.progress / d.total * 100 : 0) + '%';
    document.getElementById('status').textContent = d.stage + (d.total > 0 ? ' · ' + d.progress + '/' + d.total : '');

    if (d.done && d.segments.length) { segs = d.segments; renderSegs(d.segments); }

    if (d.report_done && d.report)
      document.getElementById('report').innerHTML = '<div class="rpt-body">' + d.report.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>') + '</div>';
    else if (d.done && !d.report_done)
      document.getElementById('report').innerHTML = '<div class="rpt-placeholder">⏳ Generating report...</div>';

    if (!d.done || !d.report_done) setTimeout(poll, 1500);
    else { polling = false; document.getElementById('status').textContent = '✓ Done — ' + d.segments.length + ' segments'; }
  });
}

function renderSegs(data) {
  const a = data.filter(s => s.label === 'animation');
  const l = data.filter(s => s.label === 'live action footage');
  document.getElementById('stats').style.display = 'flex';
  document.getElementById('s-total').textContent = data.length;
  document.getElementById('s-anim').textContent  = a.length;
  document.getElementById('s-live').textContent  = l.length;

  document.getElementById('segments').innerHTML = '';
  data.forEach((seg, i) => {
    const isA = seg.label === 'animation';
    const div = document.createElement('div');
    div.className = 'seg ' + (isA ? 'animation' : 'live');
    div.innerHTML = `<div class="seg-row">
      <span class="seg-label">${isA ? '🎨 Animation' : '🎥 Live Action'}</span>
      <span class="seg-time">${fmt(seg.start)} → ${fmt(seg.end)}</span>
      <span class="seg-conf">${Math.round(seg.confidence * 100)}%</span>
    </div><div class="seg-text">${seg.transcript || ''}</div>`;
    div.onclick = () => { player.currentTime = seg.start; player.play(); };
    document.getElementById('segments').appendChild(div);
  });
}

player.addEventListener('timeupdate', () => {
  const t = player.currentTime;
  document.querySelectorAll('.seg').forEach((el, i) => {
    el.classList.toggle('active', segs[i] && t >= segs[i].start && t < segs[i].end);
  });
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"ok": False})
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", f.filename)
    f.save(path)
    state["video_path"] = path
    t = threading.Thread(target=run_analysis, args=(path,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/video")
def video():
    return send_file(state["video_path"], mimetype="video/mp4")


@app.route("/api/results")
def api_results():
    return jsonify({
        **state["status"],
        "report":   state["report"],
        "segments": state["results"],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=False, port=5000)