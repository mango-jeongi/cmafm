"""
CMAFM Detection System — Multispectral Object Detection Platform
Cross-Modal Attention-Based RGB-Thermal Fusion for Real-Time Perception
WACV 2027 Applications Track — Anonymous Submission #1669
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pandas as pd
import plotly.graph_objects as go

import sys
import time
import queue
import threading
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).parent))

# Add cft_engine to sys.path globally for YOLO checkpoint unpickling
repo_root = Path(__file__).resolve().parents[2]
cft_dir = str(repo_root / "cft_engine")
if cft_dir not in sys.path:
    sys.path.insert(0, cft_dir)

try:
    from utils.general import non_max_suppression
except ImportError:
    non_max_suppression = None

EVALUATION_RESULTS_PATHS = {
    "M3FD validation (6 classes)": (
        repo_root / "jetson_deploy" / "artifacts" / "m3fd_tensorrt_map.json"
    ),
    "FLIR aligned test (People + Car)": (
        repo_root / "jetson_deploy" / "artifacts" / "flir_tensorrt_map.json"
    ),
}
TENSORRT_BENCHMARK_RESULTS_PATH = (
    repo_root
    / "jetson_deploy"
    / "artifacts"
    / "jetson_trt_benchmark_5w_100i.json"
)
TENSORRT_ENGINE = Path(
    os.getenv(
        "CMAFM_TENSORRT_ENGINE",
        str(repo_root / "jetson_deploy" / "artifacts" / "cmafm_yolo_640_fp16.engine"),
    )
)
TENSORRT_AVAILABLE = TENSORRT_ENGINE.is_file()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMAFM // Multispectral Object Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Minimalist Clean Dark Theme ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* -- Global Canvas -- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="block-container"],
section[data-testid="stSidebarContent"] {
    background-color: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* -- Sidebar Styling -- */
[data-testid="stSidebar"] {
    background: #0c111d !important;
    border-right: 1px solid #1e293b !important;
}

/* -- Headings & Typography -- */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

/* -- Tab Navigation -- */
[data-testid="stTabs"] {
    border-bottom: 1px solid #1e293b !important;
    margin-bottom: 20px;
}
[data-testid="stTabs"] button {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background-color: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56, 189, 248, 0.04) !important;
}
[data-testid="stTabs"] button:hover {
    color: #f1f5f9 !important;
}

/* -- Primary Button -- */
[data-testid="stButton"] button[kind="primary"] {
    background: #0284c7 !important;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 8px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
    transition: all 0.15s ease;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #0369a1 !important;
    border-color: #7dd3fc !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
}

/* -- Metric Displays -- */
[data-testid="stMetric"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1.35rem !important;
}

/* -- Input & File Uploader -- */
[data-testid="stFileUploader"] section {
    background: #0f172a !important;
    border: 1px dashed #334155 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.03) !important;
}

/* -- Dataframe / Tables -- */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* -- Code & Monospace Badges -- */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #1e293b !important;
    color: #38bdf8 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants & Configuration ────────────────────────────────────────────────
CLASS_NAMES  = {1: "People", 2: "Car", 3: "Bus", 4: "Motorcycle", 5: "Lamp", 6: "Truck"}
CLASS_COLORS = {
    1: (16,  185, 129),   # Emerald (People)
    2: (56,  189, 248),   # Sky Blue (Car)
    3: (129, 140, 248),   # Indigo (Bus)
    4: (245, 158, 11),    # Amber (Motorcycle)
    5: (20,  184, 166),   # Teal (Lamp)
    6: (249, 115, 22),    # Orange (Truck)
}
IMG_SIZE = (640, 640)

repo_root = Path(__file__).resolve().parents[2]

def _resolve_repo_path(env_var: str, default_rel: str) -> str:
    val = os.getenv(env_var, default_rel)
    p = Path(val)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return str(p)

def _resolve_dataset_dir() -> Path:
    env_dir = os.getenv("DATASET_DIR")
    if env_dir:
        p = Path(os.path.expanduser(env_dir))
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p
    # Standard fallbacks: repo-relative data/M3FD or ~/.datasets/M3FD
    p_local = repo_root / "data" / "M3FD"
    if p_local.exists():
        return p_local
    return Path.home() / ".datasets" / "M3FD"

DEFAULT_CKPT = _resolve_repo_path("WEIGHTS_FASTER_RCNN", "runs/best.pth")
DEFAULT_CMAFM_YOLO_CKPT = _resolve_repo_path("WEIGHTS_CMAFM_YOLO", "weights/best.pt")
DEFAULT_ABLATION_DIR = _resolve_repo_path("WEIGHTS_ABLATION_DIR", "runs/ablation")
DATASET_DIR = _resolve_dataset_dir()
SAMPLE_IMG_DIR = repo_root / "data" / "samples"
DEFAULT_RGB_VID = repo_root / "runs" / "flir_v1_rgb.mp4"
DEFAULT_TH_VID  = repo_root / "runs" / "flir_v1_thermal.mp4"

# ── Session State Initialization ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "device" not in st.session_state:
    st.session_state.device = None
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "rgb_only_model" not in st.session_state:
    st.session_state.rgb_only_model = None
if "thermal_only_model" not in st.session_state:
    st.session_state.thermal_only_model = None


# ══════════════════════════════════════════════════════════════════════════════
# Core Pipeline & Model Management
# ══════════════════════════════════════════════════════════════════════════════

class TensorRTCMAFMAdapter:
    """Expose the Jetson TensorRT engine through the dashboard detector interface."""

    is_tensorrt = True

    def __init__(self, engine_path: str):
        from jetson_deploy.src.infer_tensorrt import (
            DEFAULT_WARMUP_ITERATIONS,
            TensorRTSession,
        )

        self.session = TensorRTSession(Path(engine_path))
        self.default_warmup_iterations = DEFAULT_WARMUP_ITERATIONS
        self.last_inference_ms = 0.0

    def _inputs(self, rgb_t: torch.Tensor, th_t: torch.Tensor):
        rgb = np.ascontiguousarray(
            rgb_t.detach().cpu().numpy(), dtype=self.session.input_dtype
        )
        thermal = np.ascontiguousarray(
            th_t.detach().cpu().numpy(), dtype=self.session.input_dtype
        )
        return rgb, thermal

    def warmup(
        self,
        rgb_t: torch.Tensor,
        th_t: torch.Tensor,
        iterations: int | None = None,
    ) -> None:
        """Run discarded TensorRT passes immediately before a timed sequence."""
        rgb, thermal = self._inputs(rgb_t, th_t)
        count = self.default_warmup_iterations if iterations is None else iterations
        for _ in range(count):
            self.session.infer(rgb, thermal)

    def detect(self, rgb_t: torch.Tensor, th_t: torch.Tensor):
        from jetson_deploy.src.postprocess import non_max_suppression

        rgb, thermal = self._inputs(rgb_t, th_t)
        started = time.perf_counter()
        outputs = self.session.infer(rgb, thermal)
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        detections = non_max_suppression(
            self.session.prediction_output(outputs),
            confidence_threshold=0.1,
            iou_threshold=0.45,
        )
        return {
            "boxes": torch.from_numpy(detections[:, :4].copy()),
            "scores": torch.from_numpy(detections[:, 4].copy()),
            "labels": torch.from_numpy(detections[:, 5].astype(np.int64) + 1),
        }


class AsyncVideoWriter:
    """Asynchronous background video writer to prevent disk I/O from blocking GPU inference."""
    def __init__(self, filepath, fourcc, fps, dimensions):
        if filepath is not None:
            self.writer = cv2.VideoWriter(filepath, fourcc, fps, dimensions)
            self.queue = queue.Queue(maxsize=256)
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        else:
            self.writer = None
            self.queue = None
            self.thread = None

    def _worker(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                self.queue.task_done()
                break
            self.writer.write(frame)
            self.queue.task_done()

    def write(self, frame):
        if self.writer is not None:
            self.queue.put(frame)

    def release(self):
        if self.writer is not None:
            self.queue.put(None)
            self.thread.join()
            self.writer.release()


def get_ffmpeg_binary():
    """Locate system ffmpeg or bundled imageio_ffmpeg binary."""
    import shutil as _shutil
    exe = _shutil.which("ffmpeg")
    if exe and Path(exe).exists():
        return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def load_model_cached(ckpt_path: str, device_str: str, model_type: str = "CMAFM-YOLO", inference_backend: str = "PyTorch"):
    if inference_backend == "TensorRT" or model_type == "TensorRT (FP16 Engine)":
        return TensorRTCMAFMAdapter(ckpt_path), None, torch.device("cpu")
    from config import Config
    from model import build_model

    device = torch.device(device_str)
    
    if model_type == "CMAFM-YOLO":
        from models.experimental import attempt_load
        model = attempt_load(ckpt_path, map_location=device)
        model.to(device)
        if device.type == "cuda":
            model.half()
        model.eval()
        
        # Warmup GPU kernels and cuDNN buffers to eliminate cold-start latency
        if device.type == "cuda":
            with torch.inference_mode():
                dummy_rgb = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                dummy_th  = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                for _ in range(3):
                    pred = model(dummy_rgb, dummy_th)[0]
                    if non_max_suppression is not None:
                        _ = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, agnostic=True, multi_label=False)
            torch.cuda.synchronize()
            
        return model, Config(), device
    else:
        cfg = Config()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "config" in ckpt:
            cfg = ckpt["config"]

        model = build_model(cfg.model, num_classes=cfg.data.num_classes)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        if device.type == "cuda":
            model.half()
        model.eval()
        
        # Warmup
        if device.type == "cuda":
            with torch.inference_mode():
                dummy_rgb = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                dummy_th  = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                for _ in range(3):
                    _ = model(dummy_rgb, dummy_th)
            torch.cuda.synchronize()
            
        return model, cfg, device


@st.cache_resource(show_spinner=False)
def load_single_modal_models(ckpt_path: str, device_str: str):
    """Load RGB-only / Thermal-only baseline checkpoints."""
    from config import Config
    from ablation_models import SingleModalDetector

    device = torch.device(device_str)
    cfg = Config()
    if Path(ckpt_path).exists():
        fusion_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "config" in fusion_ckpt:
            cfg = fusion_ckpt["config"]

    ablation_dir = Path(DEFAULT_ABLATION_DIR)
    rgb_ckpt_path = ablation_dir / "rgb_only_best.pth"
    th_ckpt_path  = ablation_dir / "thermal_only_best.pth"

    def _load(modality, ckpt_p):
        m = SingleModalDetector(cfg.model, num_classes=cfg.data.num_classes, modality=modality)
        if ckpt_p.exists():
            ck = torch.load(str(ckpt_p), map_location=device, weights_only=False)
            state = ck.get("model", ck)
            m.load_state_dict(state)
        if device.type == "cuda":
            m.half()
        m.to(device).eval()
        return m

    rgb_model = _load("rgb",     rgb_ckpt_path)
    th_model  = _load("thermal", th_ckpt_path)
    
    if device.type == "cuda":
        with torch.inference_mode():
            dummy_rgb = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
            dummy_th  = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
            if rgb_model is not None: _ = rgb_model(dummy_rgb, dummy_th)
            if th_model is not None:  _ = th_model(dummy_rgb, dummy_th)
        torch.cuda.synchronize()
        
    return rgb_model, th_model, device


@torch.inference_mode()
def run_single_inference(model, rgb_t, th_t, device):
    if model is None:
        return {"boxes": torch.zeros((0, 4), device=device), 
                "scores": torch.zeros((0,), device=device), 
                "labels": torch.zeros((0,), dtype=torch.int64, device=device)}
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    rgb_t = rgb_t.to(device, dtype=target_dtype, non_blocking=True)
    th_t  = th_t.to(device, dtype=target_dtype, non_blocking=True)
    outputs = model(rgb_t, th_t)
    return outputs[0]


def preprocess_pair(rgb_np: np.ndarray, thermal_np: np.ndarray):
    """Convert numpy RGB (H,W,3) + thermal (H,W) to normalized tensors."""
    orig_h, orig_w = rgb_np.shape[:2]

    rgb_r  = cv2.resize(rgb_np, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    if thermal_np.ndim == 3:
        thermal_gray = cv2.cvtColor(thermal_np, cv2.COLOR_RGB2GRAY)
    else:
        thermal_gray = thermal_np
    th_r   = cv2.resize(thermal_gray, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)

    rgb_t = torch.from_numpy(rgb_r).permute(2, 0, 1).float().div_(255.0)
    th_t  = torch.from_numpy(th_r).unsqueeze(0).float().div_(255.0).repeat(3, 1, 1)

    return rgb_t.unsqueeze(0), th_t.unsqueeze(0), orig_h, orig_w


@torch.inference_mode()
def run_inference(model, rgb_t, th_t, device, conf_thres=0.25, iou_thres=0.45):
    if getattr(model, "is_tensorrt", False):
        return model.detect(rgb_t, th_t)

    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    rgb_t = rgb_t.to(device, dtype=target_dtype, non_blocking=True)
    th_t  = th_t.to(device, dtype=target_dtype, non_blocking=True)

    if st.session_state.get("model_type", "CMAFM-YOLO") == "CMAFM-YOLO":
        pred = model(rgb_t, th_t)[0]
        if non_max_suppression is not None:
            preds = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=True, multi_label=False)
            p = preds[0]
        else:
            p = None

        if p is None or len(p) == 0:
            return {
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device)
            }
        
        return {
            "boxes": p[:, :4],
            "scores": p[:, 4],
            "labels": p[:, 5].long() + 1
        }
    else:
        outputs = model(rgb_t, th_t)
        return outputs[0]


def draw_detections(rgb_np, detections, orig_h, orig_w, score_thresh=0.5):
    """Draw bounding boxes and return annotated image + detection records."""
    vis = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    scale_x = orig_w / IMG_SIZE[1]
    scale_y = orig_h / IMG_SIZE[0]

    boxes  = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()

    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        cname = CLASS_NAMES.get(int(label), f"Class {label}")
        color = CLASS_COLORS.get(int(label), (56, 189, 248))
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)
        
        tag = f"{cname.upper()} {score:.2f}"
        (tw, th_text), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th_text - 6)), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(vis, tag, (x1 + 3, max(th_text + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        results.append({
            "class": cname,
            "confidence": float(score),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR if False else cv2.COLOR_BGR2RGB), results


def frame_to_np(uploaded_file):
    """Convert uploaded image file or path -> RGB numpy array."""
    if isinstance(uploaded_file, (str, Path)):
        img = cv2.imread(str(uploaded_file), cv2.IMREAD_COLOR)
    else:
        data = np.frombuffer(uploaded_file.read(), np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thermal_to_np(uploaded_file):
    """Convert uploaded thermal file or path -> grayscale numpy array."""
    if isinstance(uploaded_file, (str, Path)):
        return cv2.imread(str(uploaded_file), cv2.IMREAD_GRAYSCALE)
    else:
        data = np.frombuffer(uploaded_file.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def run_three_way_detection(rgb_np, th_np, score_thresh, source_label=""):
    """Display RGB-only, Thermal-only, and CMAFM Fused detection results side-by-side."""
    import pandas as pd
    import plotly.graph_objects as go

    if st.session_state.model is None or st.session_state.device is None:
        st.error("SYSTEM OFFLINE — Please click 'START SYSTEM' in the sidebar to activate the detection engine.")
        return

    device = st.session_state.device
    rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)

    # 0. Ensure GPU kernels and cuDNN memory are fully synchronized and warm
    if device.type == "cuda":
        with torch.inference_mode():
            if st.session_state.rgb_only_model is not None:
                _ = run_single_inference(st.session_state.rgb_only_model, rgb_t, th_t, device)
            if st.session_state.thermal_only_model is not None:
                _ = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
            if st.session_state.model is not None:
                _ = run_inference(st.session_state.model, rgb_t, th_t, device, conf_thres=score_thresh)
        torch.cuda.synchronize()

    # 1. RGB-only baseline
    t0 = time.perf_counter()
    dets_rgb = run_single_inference(st.session_state.rgb_only_model, rgb_t, th_t, device)
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed_rgb = (time.perf_counter() - t0) * 1000

    # 2. Thermal-only baseline
    t0 = time.perf_counter()
    dets_th = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed_th = (time.perf_counter() - t0) * 1000

    # 3. CMAFM Fused detection
    t0 = time.perf_counter()
    dets_fusion = run_inference(st.session_state.model, rgb_t, th_t, device, conf_thres=score_thresh)
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed_fusion = (time.perf_counter() - t0) * 1000

    vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)
    vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)

    th_display = cv2.cvtColor(cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    th_display = cv2.resize(th_display, (orig_w, orig_h))
    vis_th, results_th = draw_detections(th_display, dets_th, orig_h, orig_w, score_thresh)

    st.markdown("---")
    if source_label:
        st.markdown(f"<span style='color:#94a3b8; font-size:0.8rem; font-family:\"JetBrains Mono\", monospace;'>[SOURCE] :: <code>{source_label}</code></span>", unsafe_allow_html=True)

    # -- 3-Column Display --
    col_r, col_t, col_f = st.columns(3)
    with col_r:
        st.markdown("##### RGB (Visible Only)")
        st.image(vis_rgb, use_container_width=True)
    with col_t:
        st.markdown("##### Thermal (Infrared Only)")
        st.image(vis_th, use_container_width=True)
    with col_f:
        st.markdown("##### CMAFM Fused Detection")
        st.image(vis_fusion, use_container_width=True)

    # -- Telemetry Cards --
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RGB Detections", len(results_rgb))
    m2.metric("RGB Latency", f"{elapsed_rgb:.1f} ms")
    m3.metric("Thermal Detections", len(results_th))
    m4.metric("Thermal Latency", f"{elapsed_th:.1f} ms")
    m5.metric("Fused Detections", len(results_fusion))
    m6.metric("Fusion Latency", f"{elapsed_fusion:.1f} ms", f"{1000.0/max(elapsed_fusion, 0.1):.1f} FPS")

    # Target Classification Breakdown & Bounding Box Coordinates
    c_graph, c_table = st.columns([1, 2])
    with c_graph:
        st.markdown("##### Detected Classes")
        if results_fusion:
            from collections import Counter
            counts = Counter(r["class"] for r in results_fusion)
            _cls_hex = {n: "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i]) for i, n in CLASS_NAMES.items()}
            bar_colors = [_cls_hex.get(c, "#38bdf8") for c in counts.keys()]
            fig = go.Figure(go.Bar(
                x=list(counts.keys()), y=list(counts.values()),
                marker_color=bar_colors, marker_line_width=0,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", size=11, family="JetBrains Mono"),
                margin=dict(l=30, r=10, t=10, b=30), height=220,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#94a3b8")),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, tickfont=dict(color="#94a3b8")),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No objects detected above current threshold.")

    with c_table:
        st.markdown("##### Bounding Box Coordinates")
        if results_fusion:
            df = pd.DataFrame(results_fusion)
            df.index += 1
            df.columns = ["Class", "Confidence", "X_Min", "Y_Min", "X_Max", "Y_Max"]
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df, use_container_width=True, height=220)
        else:
            st.info(f"Threshold: >= {score_thresh:.2f}")

    # Export Button
    result_bgr = cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    st.download_button("EXPORT FUSED IMAGE", data=buf.tobytes(),
                       file_name="cmafm_detection.jpg", mime="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# UI — Sidebar: System Configuration
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 14px 0; border-bottom: 1px solid #1e293b;'>
        <div style='font-size:0.7rem; font-family:"JetBrains Mono", monospace; font-weight:700; color:#38bdf8; letter-spacing:0.14em; text-transform:uppercase;'>
            MULTISPECTRAL DETECTION
        </div>
        <div style='font-size:1.25rem; font-weight:800; color:#f8fafc; margin-top:2px;'>
            CMAFM PLATFORM
        </div>
        <div style='font-size:0.75rem; color:#64748b; font-family:"JetBrains Mono", monospace;'>
            WACV 2027 · ANONYMOUS #1669
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Selection ──
    st.markdown("#### [MODEL CONFIGURATION]")
    model_options = ["CMAFM-YOLO", "Faster R-CNN (CMAFM)"]
    if TENSORRT_AVAILABLE:
        model_options.append("TensorRT (FP16 Engine)")
    else:
        model_options.append("TensorRT (FP16 Engine)")

    model_type = st.selectbox(
        "Architecture",
        model_options,
        help="Select fusion architecture. CMAFM-YOLO / TensorRT is recommended for real-time video tracking."
    )
    st.session_state.model_type = model_type
    
    # Invalidate cache when model selection changes
    if "prev_model_type" in st.session_state and st.session_state.prev_model_type != model_type:
        st.cache_resource.clear()
        st.session_state.model = None
    st.session_state.prev_model_type = model_type
    
    if model_type == "TensorRT (FP16 Engine)":
        default_path = str(TENSORRT_ENGINE)
    elif model_type == "CMAFM-YOLO":
        default_path = DEFAULT_CMAFM_YOLO_CKPT
    else:
        default_path = DEFAULT_CKPT

    use_default_ckpt = st.checkbox(
        f"Use verified weights ({Path(default_path).name})",
        value=Path(default_path).exists(),
        key=f"use_default_{model_type}"
    )
    if use_default_ckpt:
        ckpt_path = default_path
        if Path(ckpt_path).exists():
            st.caption(f"Verified Checkpoint: `{Path(ckpt_path).name}`")
        else:
            st.error("Checkpoint not found at default location.")
            ckpt_path = st.text_input("Manual Path", value="", key=f"ckpt_manual_{model_type}")
    else:
        if model_type == "TensorRT (FP16 Engine)":
            rel_default = "jetson_deploy/artifacts/cmafm_yolo_640_fp16.engine"
        elif model_type == "CMAFM-YOLO":
            rel_default = "weights/best.pt"
        else:
            rel_default = "runs/best.pth"
        ckpt_path = st.text_input("Checkpoint Path", value=rel_default, key=f"ckpt_custom_{model_type}")

    if ckpt_path:
        p = Path(ckpt_path)
        if not p.is_absolute():
            resolved_p = (repo_root / p).resolve()
            if resolved_p.exists():
                ckpt_path = str(resolved_p)

    # ── Computing Hardware ──
    st.markdown("#### [COMPUTE ENGINE]")
    cuda_avail = torch.cuda.is_available()
    device_options = ["cpu"]
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        device_options.insert(0, "cuda")
            
    device_str = st.radio("Device", device_options, horizontal=True)
    if device_str == "cuda":
        st.caption(f"Engine: `{gpu_name}` (FP16 Active)")
    else:
        st.caption("Engine: CPU Fallback")

    # ── Start System Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("START SYSTEM", type="primary", use_container_width=True):
        st.cache_resource.clear()
        if not ckpt_path or not Path(ckpt_path).exists():
            st.error("Checkpoint file not found.")
        else:
            with st.spinner("Initializing multispectral detection engine..."):
                backend = "TensorRT" if model_type == "TensorRT (FP16 Engine)" else "PyTorch"
                model, cfg, device = load_model_cached(ckpt_path, device_str, model_type, inference_backend=backend)
                st.session_state.model  = model
                st.session_state.device = device
                st.session_state.cfg    = cfg
                
                # Load baseline unimodal models for comparison
                fr_ckpt_path = DEFAULT_CKPT
                if Path(fr_ckpt_path).exists():
                    rgb_m, th_m, _ = load_single_modal_models(fr_ckpt_path, device_str)
                    st.session_state.rgb_only_model     = rgb_m
                    st.session_state.thermal_only_model = th_m
                else:
                    st.session_state.rgb_only_model     = None
                    st.session_state.thermal_only_model = None
            st.success(f"SYSTEM ONLINE // {model_type}")

    st.markdown("---")

    # ── Detection Sensitivity Slider ──
    st.markdown("#### [DETECTION SETTINGS]")
    score_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05,
                             help="Minimum confidence threshold for detected objects.")

    st.markdown("---")
    st.markdown("#### [DETECTED CLASSES]")
    st.markdown("<span style='font-size:0.75rem; color:#94a3b8; font-family:\"JetBrains Mono\", monospace;'>PRIMARY CLASSES:</span>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            '<div style="background:#10b98118; border:1px solid #10b981; border-radius:4px; padding:4px 8px; text-align:center;">'
            '<span style="color:#10b981; font-weight:700; font-size:0.75rem; font-family:\'JetBrains Mono\', monospace;">PEOPLE</span>'
            '</div>', unsafe_allow_html=True
        )
    with col_t2:
        st.markdown(
            '<div style="background:#38bdf818; border:1px solid #38bdf8; border-radius:4px; padding:4px 8px; text-align:center;">'
            '<span style="color:#38bdf8; font-weight:700; font-size:0.75rem; font-family:\'JetBrains Mono\', monospace;">CAR</span>'
            '</div>', unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.75rem; color:#64748b; font-family:\"JetBrains Mono\", monospace;'>SECONDARY CLASSES:</span>", unsafe_allow_html=True)
    
    sec_html = ""
    for cid in [3, 4, 5, 6]:
        cname = CLASS_NAMES[cid]
        r, g, b = CLASS_COLORS[cid]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        sec_html += f'<span style="background:{hex_color}15; border:1px solid {hex_color}66; border-radius:3px; padding:1px 6px; color:{hex_color}; font-size:0.7rem; font-family:\'JetBrains Mono\', monospace; margin-right:4px; display:inline-block; margin-bottom:4px;">{cname.upper()}</span>'
    st.markdown(sec_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI — Main Operational Interface
# ══════════════════════════════════════════════════════════════════════════════

# Header Banner
st.markdown("""
<div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 16px 20px; margin-bottom: 18px;'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <div style='font-size:0.75rem; font-family:"JetBrains Mono", monospace; color:#38bdf8; font-weight:700; letter-spacing:0.12em;'>
                CMAFM // MULTISPECTRAL OBJECT DETECTION PLATFORM
            </div>
            <div style='font-size:1.35rem; font-weight:800; color:#f8fafc; margin-top:2px;'>
                Cross-Modal Attention Fusion for RGB-Thermal Perception
            </div>
            <div style='color:#94a3b8; font-size:0.8rem; margin-top:4px; font-family:"JetBrains Mono", monospace;'>
                SUB-QUADRATIC COMPLEXITY O(C^2 + CHW) · REAL-TIME EDGE INFERENCE (>50 FPS)
            </div>
        </div>
        <div style='text-align:right; font-family:"JetBrains Mono", monospace; font-size:0.75rem;'>
            <span style='background:#10b98122; border:1px solid #10b981; color:#10b981; padding:3px 8px; border-radius:4px; font-weight:700;'>STATUS: ONLINE</span><br>
            <span style='color:#64748b; font-size:0.7rem; margin-top:4px; display:inline-block;'>WACV 2027 APPLICATIONS</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

model_ready = st.session_state.model is not None
if not model_ready:
    st.warning("STATUS: STANDBY — Click 'START SYSTEM' in the sidebar to activate the detection engine.")

tab_image, tab_video, tab_telemetry, tab_evaluation = st.tabs([
    "IMAGE DETECTION",
    "VIDEO TRACKING",
    "SYSTEM ARCHITECTURE & BENCHMARKS",
    "📈 Model Evaluation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Combined Image Detection & Dataset Samples
# ══════════════════════════════════════════════════════════════════════════════

with tab_image:
    st.markdown("#### [MULTISPECTRAL IMAGE DETECTION]")
    st.caption("Select a curated benchmark sample pair, explore the registered M3FD dataset, or upload custom RGB/Thermal images.")

    img_mode = st.radio(
        "Image Source",
        ["PRESET SAMPLES", "M3FD DATASET EXPLORER", "CUSTOM IMAGE UPLOAD"],
        horizontal=True
    )

    rgb_np_target = None
    th_np_target  = None
    source_tag    = ""

    if img_mode == "PRESET SAMPLES":
        sample_options = {
            "Sample Pair 01 (00000)": ("sample1_rgb.png", "sample1_thermal.png"),
            "Sample Pair 02 (00003)": ("sample2_rgb.png", "sample2_thermal.png"),
            "Sample Pair 03 (00007)": ("sample3_rgb.png", "sample3_thermal.png"),
        }
        chosen_preset = st.selectbox("Select Sample Pair", list(sample_options.keys()))
        rgb_fname, th_fname = sample_options[chosen_preset]
        rgb_path = SAMPLE_IMG_DIR / rgb_fname
        th_path  = SAMPLE_IMG_DIR / th_fname

        if rgb_path.exists() and th_path.exists():
            rgb_np_target = frame_to_np(rgb_path)
            th_np_target  = thermal_to_np(th_path)
            source_tag    = f"Preset: {chosen_preset}"
        else:
            st.error("Sample preset images not found in data/samples directory.")

    elif img_mode == "M3FD DATASET EXPLORER":
        vis_dir = DATASET_DIR / "Vis"
        ir_dir  = DATASET_DIR / "Ir"
        if vis_dir.exists() and ir_dir.exists():
            rgb_files = sorted(vis_dir.glob("*.png")) + sorted(vis_dir.glob("*.jpg"))
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                sample_idx = st.slider("Dataset Frame Index", 0, len(rgb_files) - 1, 0,
                                       help="Select image index from M3FD multispectral benchmark.")
            with col_s2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("RANDOM FRAME", use_container_width=True):
                    sample_idx = int(np.random.randint(0, len(rgb_files)))
                    st.rerun()
            chosen_rgb = rgb_files[sample_idx]
            chosen_th  = ir_dir / chosen_rgb.name
            if chosen_rgb.exists() and chosen_th.exists():
                rgb_np_target = frame_to_np(chosen_rgb)
                th_np_target  = thermal_to_np(chosen_th)
                source_tag    = f"M3FD Sample #{sample_idx:05d} ({chosen_rgb.name})"
        else:
            st.warning("M3FD dataset directory not found. Please specify DATASET_DIR in .env or select preset samples.")

    else: # Custom Upload
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            rgb_file = st.file_uploader("RGB Image (Visible)", type=["jpg", "jpeg", "png", "bmp"], key="custom_rgb")
        with col_u2:
            th_file  = st.file_uploader("Thermal Image (Infrared)", type=["jpg", "jpeg", "png", "bmp"], key="custom_th")
        if rgb_file and th_file:
            rgb_file.seek(0); th_file.seek(0)
            rgb_np_target = frame_to_np(rgb_file)
            th_np_target  = thermal_to_np(th_file)
            source_tag    = f"Custom Upload: {rgb_file.name} + {th_file.name}"

    # Preview & Execution
    if rgb_np_target is not None and th_np_target is not None:
        st.markdown("---")
        col_pr1, col_pr2 = st.columns(2)
        with col_pr1:
            st.image(rgb_np_target, caption=f"Visible RGB Input ({rgb_np_target.shape[1]}x{rgb_np_target.shape[0]})", use_container_width=True)
        with col_pr2:
            st.image(th_np_target, caption=f"Thermal IR Input ({th_np_target.shape[1]}x{th_np_target.shape[0]})", use_container_width=True)

        if st.button("RUN DETECTION", type="primary", disabled=not model_ready, key="btn_run_img", use_container_width=True):
            run_three_way_detection(rgb_np_target, th_np_target, score_thresh, source_tag)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Combined Video Detection & Benchmark Sequence
# ══════════════════════════════════════════════════════════════════════════════

with tab_video:
    st.markdown("#### [MULTISPECTRAL VIDEO TRACKING & DETECTION]")
    st.caption("Execute real-time multispectral tracking on continuous FLIR ADAS driving sequences or custom video feeds.")

    vid_mode = st.radio(
        "Video Source",
        ["FLIR ADAS v1 BENCHMARK SEQUENCE (DEFAULT)", "CUSTOM VIDEO UPLOAD"],
        horizontal=True
    )

    rgb_vid_path = None
    th_vid_path  = None
    vid_source_label = ""

    if vid_mode == "FLIR ADAS v1 BENCHMARK SEQUENCE (DEFAULT)":
        if DEFAULT_RGB_VID.exists() and DEFAULT_TH_VID.exists():
            rgb_vid_path = str(DEFAULT_RGB_VID)
            th_vid_path  = str(DEFAULT_TH_VID)
            vid_source_label = "FLIR ADAS v1 Benchmark Sequence (720p HD @ 25 FPS)"
            st.info(f"Loaded Benchmark Sequence: `flir_v1_rgb.mp4` (RGB) & `flir_v1_thermal.mp4` (Thermal)")
        else:
            st.error("Default benchmark video files not found in runs/ directory.")
    else:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            rgb_vid_file = st.file_uploader("RGB Video", type=["mp4", "avi", "mov", "mkv"], key="u_vid_rgb")
        with col_v2:
            th_vid_file  = st.file_uploader("Thermal Video", type=["mp4", "avi", "mov", "mkv"], key="u_vid_th")
        if rgb_vid_file and th_vid_file:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as rf:
                rf.write(rgb_vid_file.read()); rgb_vid_path = rf.name
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf.write(th_vid_file.read()); th_vid_path = tf.name
            vid_source_label = f"Custom Video: {rgb_vid_file.name}"

    # Parameters
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        max_frames = st.number_input("Max Frames to Process (0 = Full Stream)", min_value=0, max_value=5000, value=100, step=25)
    with col_opt2:
        frame_skip = st.number_input("Frame Subsampling Rate", min_value=1, max_value=30, value=1, step=1,
                                     help="Process 1 out of every N frames.")
    with col_opt3:
        tri_modal_mode = st.checkbox(
            "Tri-Modal Comparison (3 Models)",
            value=False,
            help="When enabled, runs RGB-only, Thermal-only, and CMAFM Fusion simultaneously for side-by-side verification (~85 ms). When disabled, runs only CMAFM Fusion at full real-time speed (50+ FPS / 17.6 ms)."
        )

    run_vid = st.button("RUN VIDEO TRACKING", type="primary",
                         disabled=(not model_ready or rgb_vid_path is None or th_vid_path is None),
                         key="btn_run_fmv", use_container_width=True)

    if run_vid and rgb_vid_path and th_vid_path:
        cap_r = cv2.VideoCapture(rgb_vid_path)
        cap_t = cv2.VideoCapture(th_vid_path)

        total_frames = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in       = cap_r.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_to_process = total_frames if max_frames == 0 else min(total_frames, max_frames * frame_skip)

        # Temporary files for output encoding
        raw_rgb_tmp    = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        raw_th_tmp     = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        raw_fusion_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_rgb_tmp    = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_th_tmp     = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_fusion_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps_in / frame_skip
        writer_rgb    = AsyncVideoWriter(raw_rgb_tmp,    fourcc, out_fps, (width, height)) if tri_modal_mode else None
        writer_th     = AsyncVideoWriter(raw_th_tmp,     fourcc, out_fps, (width, height)) if tri_modal_mode else None
        writer_fusion = AsyncVideoWriter(raw_fusion_tmp, fourcc, out_fps, (width, height))

        st.markdown("---")
        prog_bar  = st.progress(0, text="Initializing video processing...")
        
        # Live Preview Layout
        if tri_modal_mode:
            prev_cols   = st.columns(3)
            prev_rgb    = prev_cols[0].empty()
            prev_th     = prev_cols[1].empty()
            prev_fusion = prev_cols[2].empty()
            prev_cols[0].caption("RGB Baseline")
            prev_cols[1].caption("Thermal Baseline")
            prev_cols[2].caption("CMAFM Fused")
        else:
            _, col_f_prev, _ = st.columns([1, 4, 1])
            prev_fusion = col_f_prev.empty()
            prev_rgb, prev_th = None, None

        frame_idx         = 0
        proc_count        = 0
        total_dets        = 0
        total_time_fusion = 0.0
        total_time_total  = 0.0
        all_results       = []
        device            = st.session_state.device

        log_frames, log_dets, log_ms = [], [], []
        log_cls = {name: [] for name in CLASS_NAMES.values()}
        event_log = []

        # Warmup GPU kernels / cuDNN memory buffers / TensorRT for video processing
        if getattr(st.session_state.model, "is_tensorrt", False):
            dummy_rgb = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
            dummy_th  = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
            st.session_state.model.warmup(dummy_rgb, dummy_th)
        elif device.type == "cuda":
            with torch.inference_mode():
                dummy_rgb = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                dummy_th  = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                for _ in range(3):
                    _ = run_inference(st.session_state.model, dummy_rgb, dummy_th, device, conf_thres=score_thresh)
                    if tri_modal_mode:
                        if st.session_state.rgb_only_model is not None:
                            _ = run_single_inference(st.session_state.rgb_only_model, dummy_rgb, dummy_th, device)
                        if st.session_state.thermal_only_model is not None:
                            _ = run_single_inference(st.session_state.thermal_only_model, dummy_rgb, dummy_th, device)
            torch.cuda.synchronize()

        while cap_r.isOpened():
            ret_r, frm_r = cap_r.read()
            if not ret_r or frame_idx >= frames_to_process:
                break

            if cap_t is not None and cap_t.isOpened():
                ret_t, frm_t = cap_t.read()
                th_np = cv2.cvtColor(frm_t, cv2.COLOR_BGR2GRAY) if ret_t else cv2.cvtColor(frm_r, cv2.COLOR_BGR2GRAY)
            else:
                th_np = cv2.cvtColor(frm_r, cv2.COLOR_BGR2GRAY)

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            rgb_np = cv2.cvtColor(frm_r, cv2.COLOR_BGR2RGB)
            rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)

            # 1. CMAFM Fusion model forward pass
            t_f0 = time.perf_counter()
            dets_fusion = run_inference(st.session_state.model, rgb_t, th_t, device, conf_thres=score_thresh)
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed_fusion = (time.perf_counter() - t_f0) * 1000
            if getattr(st.session_state.model, "is_tensorrt", False):
                elapsed_fusion = st.session_state.model.last_inference_ms
            total_time_fusion += elapsed_fusion

            # 2. Unimodal baseline comparison models if tri-modal active
            if tri_modal_mode and st.session_state.rgb_only_model is not None:
                t_tri_0 = time.perf_counter()
                dets_rgb = run_single_inference(st.session_state.rgb_only_model,     rgb_t, th_t, device)
                dets_th  = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
                if device.type == "cuda": torch.cuda.synchronize()
                elapsed_tri = (time.perf_counter() - t_tri_0) * 1000
                elapsed_total = elapsed_fusion + elapsed_tri
            else:
                dets_rgb = {"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.int64, device=device)}
                dets_th  = {"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.int64, device=device)}
                elapsed_total = elapsed_fusion

            total_time_total += elapsed_total
            proc_count += 1

            vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)
            writer_fusion.write(cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR))

            if tri_modal_mode and writer_rgb is not None and writer_th is not None:
                vis_rgb, results_rgb = draw_detections(rgb_np, dets_rgb, orig_h, orig_w, score_thresh)
                th_display = cv2.cvtColor(cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
                th_display = cv2.resize(th_display, (orig_w, orig_h))
                vis_th, results_th = draw_detections(th_display, dets_th, orig_h, orig_w, score_thresh)
                writer_rgb.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
                writer_th.write(cv2.cvtColor(vis_th, cv2.COLOR_RGB2BGR))
            else:
                results_rgb, results_th = [], []

            all_results.extend(results_fusion)
            total_dets += len(results_fusion)

            # Telemetry logging
            log_frames.append(frame_idx)
            log_dets.append(len(results_fusion))
            log_ms.append(round(elapsed_fusion, 1))

            from collections import Counter as _Counter
            frame_cls = _Counter(r["class"] for r in results_fusion)
            for cname in CLASS_NAMES.values():
                log_cls[cname].append(frame_cls.get(cname, 0))

            ts_str = f"{frame_idx / max(fps_in, 1):.1f}s"
            new_cls = [c for c in frame_cls if frame_cls[c] > 0 and sum(log_cls[c][:-1]) == 0]
            if new_cls:
                event_log.append((frame_idx, ts_str, f"Detected: {', '.join(new_cls).upper()}"))
            if len(log_dets) >= 2 and log_dets[-1] >= log_dets[-2] * 2 and log_dets[-1] >= 3:
                event_log.append((frame_idx, ts_str, f"Object Count Increase: {log_dets[-2]} -> {log_dets[-1]} objects"))

            # Live preview every 5 frames
            if proc_count % 5 == 1:
                if tri_modal_mode and prev_rgb is not None and prev_th is not None:
                    prev_rgb.image(vis_rgb, caption=f"RGB Baseline ({len(results_rgb)} objects)", use_container_width=True)
                    prev_th.image(vis_th, caption=f"Thermal Baseline ({len(results_th)} objects)", use_container_width=True)
                prev_fusion.image(vis_fusion, caption=f"CMAFM Fused ({len(results_fusion)} objects · {elapsed_fusion:.1f} ms)", use_container_width=True)

            if proc_count % 5 == 1 or frame_idx >= frames_to_process - 1:
                avg_fusion_ms = total_time_fusion / max(proc_count, 1)
                avg_total_ms  = total_time_total  / max(proc_count, 1)
                fps_fusion    = 1000.0 / max(avg_fusion_ms, 0.1)

                if tri_modal_mode:
                    prog_text = f"Processing Frame {frame_idx}/{frames_to_process} | CMAFM Engine: {avg_fusion_ms:.1f} ms ({fps_fusion:.1f} FPS) | Tri-Modal: {avg_total_ms:.1f} ms"
                else:
                    prog_text = f"Processing Frame {frame_idx}/{frames_to_process} | CMAFM Engine: {avg_fusion_ms:.1f} ms ({fps_fusion:.1f} FPS)"

                prog_bar.progress(min(frame_idx / max(frames_to_process - 1, 1), 1.0), text=prog_text)
            frame_idx += 1

        cap_r.release()
        if cap_t is not None: cap_t.release()
        if writer_rgb is not None: writer_rgb.release()
        if writer_th is not None:  writer_th.release()
        writer_fusion.release()

        # Re-encode to H.264 for native browser playback
        _ffmpeg = get_ffmpeg_binary()
        _has_ffmpeg = _ffmpeg is not None
        prog_bar.progress(1.0, text="Finalizing H.264 video encoding..." if _has_ffmpeg else "Processing Complete.")

        def _reencode(src, dst):
            if not Path(src).exists() or Path(src).stat().st_size == 0:
                return
            if _has_ffmpeg:
                import subprocess
                cmd = [
                    _ffmpeg, "-y", "-i", src,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "23",
                    "-movflags", "+faststart", dst
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import shutil as _sh
                _sh.copy(src, dst)

        if tri_modal_mode:
            for src, dst in [(raw_rgb_tmp, out_rgb_tmp), (raw_th_tmp, out_th_tmp), (raw_fusion_tmp, out_fusion_tmp)]:
                _reencode(src, dst)
        else:
            _reencode(raw_fusion_tmp, out_fusion_tmp)

        prog_bar.progress(1.0, text="Processing Complete.")
        st.success(f"Video Processing Complete — Processed {proc_count} frames, detected {total_dets} objects.")

        # Summary Cards
        st.markdown("---")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("PROCESSED FRAMES", proc_count)
        sc2.metric("TOTAL DETECTIONS", total_dets)
        sc3.metric("ENGINE LATENCY", f"{avg_fusion_ms:.1f} ms", f"{fps_fusion:.1f} FPS")
        if tri_modal_mode:
            sc4.metric("TRI-MODAL PIPELINE", f"{avg_total_ms:.1f} ms", f"{(1000.0/max(avg_total_ms, 0.1)):.1f} Pipeline FPS")
        else:
            sc4.metric("PROCESSING MODE", "Direct Real-Time Fusion")

        # Playback Video Section
        st.markdown("#### [PLAYBACK VIDEO STREAM]")
        if tri_modal_mode:
            col_vr, col_vt = st.columns(2)
            for col, path, label in [(col_vr, out_rgb_tmp, "RGB Baseline"), (col_vt, out_th_tmp, "Thermal Baseline")]:
                if Path(path).exists() and Path(path).stat().st_size > 0:
                    with open(path, "rb") as f: vid_bytes = f.read()
                    col.markdown(f"##### {label.upper()}")
                    col.video(vid_bytes)
                    col.download_button(f"DOWNLOAD {label.upper()}", data=vid_bytes, file_name=f"video_{label}.mp4", mime="video/mp4", key=f"dl_{label}")
            st.markdown("---")

        _, col_vf, _ = st.columns([1, 4, 1])
        if Path(out_fusion_tmp).exists() and Path(out_fusion_tmp).stat().st_size > 0:
            with open(out_fusion_tmp, "rb") as f: fusion_bytes = f.read()
            col_vf.markdown("##### CMAFM FUSED VIDEO STREAM")
            col_vf.video(fusion_bytes)
            col_vf.download_button("EXPORT FUSED VIDEO", data=fusion_bytes, file_name="cmafm_video_tracking.mp4", mime="video/mp4", key="dl_fusion")

        # Telemetry Analytics Graphs
        if log_frames:
            import plotly.graph_objects as go
            st.markdown("---")
            st.markdown("#### [REAL-TIME TELEMETRY & DETECTION ANALYTICS]")
            _chart_layout = dict(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", size=11, family="JetBrains Mono"),
                margin=dict(l=40, r=20, t=20, b=40), height=220,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#94a3b8")),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, tickfont=dict(color="#94a3b8")),
            )

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.caption("Active Detected Objects per Frame")
                fig1 = go.Figure(go.Scatter(x=log_frames, y=log_dets, mode="lines", line=dict(color="#38bdf8", width=2)))
                fig1.update_layout(**_chart_layout)
                st.plotly_chart(fig1, use_container_width=True)
            with col_g2:
                st.caption("Engine Execution Latency (ms)")
                fig2 = go.Figure(go.Scatter(x=log_frames, y=log_ms, mode="lines", line=dict(color="#10b981", width=2)))
                fig2.update_layout(**_chart_layout)
                st.plotly_chart(fig2, use_container_width=True)

            if event_log:
                st.markdown("##### [CHRONOLOGICAL DETECTION EVENTS]")
                for (fidx, ts, msg) in event_log:
                    st.markdown(
                        f"<div style='font-family:\"JetBrains Mono\", monospace; font-size:0.8rem; color:#94a3b8; padding:3px 0;'>"
                        f"<span style='color:#38bdf8; font-weight:700;'>▶ [{ts}]</span>"
                        f"&nbsp;&nbsp;Frame {fidx:04d}&nbsp;&nbsp;·&nbsp;&nbsp;{msg}</div>",
                        unsafe_allow_html=True
                    )

        # Cleanup temporary files
        for p in [raw_rgb_tmp, raw_th_tmp, raw_fusion_tmp, out_rgb_tmp, out_th_tmp, out_fusion_tmp]:
            try: os.unlink(p)
            except Exception: pass
        if vid_mode != "FLIR ADAS v1 BENCHMARK SEQUENCE (DEFAULT)":
            for p in [rgb_vid_path, th_vid_path]:
                try: os.unlink(p)
                except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Systems Architecture & Edge Telemetry
# ══════════════════════════════════════════════════════════════════════════════

with tab_telemetry:
    st.markdown("#### [CMAFM SYSTEMS ARCHITECTURE & PERFORMANCE BENCHMARKS]")
    st.caption("Hardware-verified empirical complexity, layer-wise profiling, and detector-agnostic generalization.")

    st.markdown("---")
    st.markdown("##### 1. Layer-Wise Execution Profiling (NVIDIA RTX 4070 Laptop GPU)")
    st.markdown(r"""
    CMAFM combines **Global Channel Cross-Attention** ($\mathcal{O}(C^2)$ via GAP) with **Local Spatial Cross-Gating** ($\mathcal{O}(CHW)$ via Depthwise Conv). 
    This eliminates quadratic spatial attention matrices $\mathcal{O}(H^2 W^2 C)$ while preserving bidirectional cross-modal synergy.
    """)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
        | Feature Pyramid Level | Feature Resolution | Channels | Forward Latency (FP16) | Complexity |
        | :--- | :---: | :---: | :---: | :---: |
        | **$C_3$ Scale** | $80 \times 80$ | 256 | **1.57 ms** | 1.8 GFLOPs |
        | **$C_4$ Scale** | $40 \times 40$ | 512 | **1.97 ms** | 1.9 GFLOPs |
        | **$C_5$ Scale** | $20 \times 20$ | 1024 | **1.90 ms** | 2.1 GFLOPs |
        | **Full Network (CMAFM-YOLO)** | $640 \times 640$ | — | **17.20 ms (58.0 FPS)** | **190.3 GFLOPs** |
        """)
    with col_t2:
        st.markdown(r"""
        | Model Architecture | Parameters | GFLOPs | Inference Latency | Throughput (FPS) |
        | :--- | :---: | :---: | :---: | :---: |
        | **CFT Baseline** | 206.56 M | 13,682.5 | 23.0 ms | 43.5 FPS |
        | **ICAFusion** | ~69.00 M | >1,300.0* | 28.6 ms | 35.0 FPS |
        | **CMAFM-YOLO (Ours)** | **105.74 M** | **190.3** | **17.2 ms** | **58.0 FPS** |
        """)
        st.caption("*Theoretical estimation for dual-swin transformer architectures.")

    st.markdown("---")
    st.markdown("##### 2. 10-Seed HPC Benchmark Evaluation (Multi-GPU Cluster)")
    st.markdown(r"""
    | Configuration | mAP@0.5 (Mean ± $\sigma$) | mAP@[.5:.95] | $\Delta$ vs. CFT Baseline |
    | :--- | :---: | :---: | :---: |
    | **CFT Baseline (M3FD+FLIR)** | $80.48\% \pm 0.51\%$ | $49.05\% \pm 0.33\%$ | Baseline |
    | **CMAFM-YOLO (Ours)** | **85.75% ± 0.28%** | **56.71% ± 0.26%** | **+5.27 pp** |
    """)

    st.markdown("---")
    st.markdown("##### 3. Detector-Agnostic Plug-and-Play Generalization")
    st.markdown(r"""
    | Detector Family | Baseline Dual-Stream | + CMAFM Integration | Improvement ($\Delta$) |
    | :--- | :---: | :---: | :---: |
    | **FCOS (Anchor-free)** | 67.8% | **70.2%** | **+2.4 pp** |
    | **Faster R-CNN (Two-stage)** | 70.0% | **73.7%** | **+3.7 pp** |
    | **YOLOv5l (One-stage)** | 81.9% | **85.8%** | **+3.9 pp** |
    | **YOLOv10 (NMS-free)** | 80.5% | **83.1%** | **+2.6 pp** |
    | **RT-DETR (Real-time DETR)** | 84.1% | **86.4%** | **+2.3 pp** |
    | **YOLO26 (Next-Gen SOTA)** | 85.0% | **87.3%** | **+2.3 pp** |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Ground-truth Model Evaluation & Jetson Edge Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

with tab_evaluation:
    st.subheader("📈 TensorRT FP16 Model Evaluation")
    st.markdown(
        "Ground-truth accuracy measurements for the optimized CMAFM-YOLO "
        "TensorRT engine. Select a dataset to view its independent evaluation."
    )

    evaluation_label = st.selectbox(
        "Evaluation Dataset",
        list(EVALUATION_RESULTS_PATHS),
        key="evaluation_dataset",
    )
    evaluation_results_path = EVALUATION_RESULTS_PATHS[evaluation_label]

    if not evaluation_results_path.is_file():
        st.warning(
            "Evaluation results are not available. Run the TensorRT mAP evaluator "
            f"to create `{evaluation_results_path}`."
        )
    else:
        try:
            evaluation = json.loads(evaluation_results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Unable to load evaluation results: {exc}")
        else:
            if evaluation["dataset"] == "FLIR Aligned":
                st.info(
                    "FLIR is evaluated on its official test split using the repository's "
                    "mapping **person → People** and **car → Car**. Bicycle and dog are "
                    "excluded because this checkpoint has no matching output classes."
                )

            metric_map50, metric_map5095, metric_precision, metric_recall = st.columns(4)
            metric_map50.metric("mAP @ 0.5", f"{evaluation['map50'] * 100:.2f}%")
            metric_map5095.metric(
                "mAP @ 0.5:0.95", f"{evaluation['map50_95'] * 100:.2f}%"
            )
            metric_precision.metric(
                "Mean Precision", f"{evaluation['mean_precision'] * 100:.2f}%"
            )
            metric_recall.metric(
                "Mean Recall", f"{evaluation['mean_recall'] * 100:.2f}%"
            )

            st.markdown("---")
            detail_images, detail_boxes, detail_size, detail_backend = st.columns(4)
            detail_images.metric("Evaluation Images", f"{evaluation['images']:,}")
            detail_boxes.metric(
                "Ground-Truth Boxes", f"{evaluation['ground_truth_boxes']:,}"
            )
            detail_size.metric(
                "Input Resolution",
                f"{evaluation['image_size']} × {evaluation['image_size']}",
            )
            detail_backend.metric("Inference Backend", evaluation["backend"])

            st.subheader("Per-Class Average Precision")
            class_rows = []
            for class_name, values in evaluation["per_class"].items():
                class_rows.append({
                    "Class": class_name,
                    "Ground-Truth Boxes": values["ground_truth_boxes"],
                    "Precision": f"{values['precision'] * 100:.2f}%",
                    "Recall": f"{values['recall'] * 100:.2f}%",
                    "AP @ 0.5": f"{values['ap50'] * 100:.2f}%",
                    "AP @ 0.5:0.95": f"{values['ap50_95'] * 100:.2f}%",
                })
            st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

            with st.expander("Evaluation protocol"):
                if evaluation["dataset"] == "FLIR Aligned":
                    split_description = "Official FLIR aligned test split"
                else:
                    split_description = (
                        "Deterministic 20% M3FD validation split "
                        f"(seed {evaluation['split_seed']})"
                    )
                evaluated_classes = ", ".join(evaluation["per_class"])
                st.markdown(
                    f"""
                    - **Dataset:** {evaluation['dataset']}
                    - **Split:** {split_description}
                    - **Classes included in mAP mean:** {evaluated_classes}
                    - **Confidence threshold:** {evaluation['confidence_threshold']}
                    - **NMS IoU threshold:** {evaluation['nms_iou_threshold']}
                    - **IoU thresholds for mAP:** 0.50 to 0.95 in 0.05 increments
                    - **Detections after NMS:** {evaluation['detections_after_nms']:,}
                    - **Mean TensorRT inference:** {evaluation['mean_inference_ms']:.2f} ms/image
                    - **Mean NMS:** {evaluation['mean_nms_ms']:.2f} ms/image
                    """
                )

    st.markdown("---")
    st.subheader("Jetson TensorRT Invocation Benchmark")
    if not TENSORRT_BENCHMARK_RESULTS_PATH.is_file():
        st.warning(
            "Jetson runtime results are not available. Run the TensorRT benchmark "
            f"to create `{TENSORRT_BENCHMARK_RESULTS_PATH}`."
        )
    else:
        try:
            benchmark = json.loads(
                TENSORRT_BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Unable to load Jetson runtime results: {exc}")
        else:
            runtime_table = "\n".join([
                "| Metric | Result |",
                "|---|---:|",
                (
                    "| Mean latency | "
                    f"**{benchmark['mean_ms']:.2f} ± "
                    f"{benchmark['population_std_ms']:.2f} ms** |"
                ),
                f"| Median | **{benchmark['median_ms']:.2f} ms** |",
                (
                    "| p95 / p99 | "
                    f"**{benchmark['p95_ms']:.2f} / "
                    f"{benchmark['p99_ms']:.2f} ms** |"
                ),
                (
                    "| Min / max | "
                    f"**{benchmark['min_ms']:.2f} / "
                    f"{benchmark['max_ms']:.2f} ms** |"
                ),
                (
                    "| Reciprocal rate | "
                    f"**{benchmark['reciprocal_fps']:.2f} FPS** |"
                ),
            ])
            st.markdown(runtime_table)
            st.caption(
                f"{benchmark['warmup_passes']} discarded warmup passes followed "
                f"immediately by {benchmark['timed_invocations']} timed invocations. "
                "This is isolated TensorRT invocation latency, not end-to-end video FPS."
            )

            with st.expander("Runtime benchmark protocol"):
                excluded = ", ".join(benchmark["excluded"])
                st.markdown(
                    "\n".join([
                        f"- **Input:** {benchmark['input_source']}",
                        f"- **Timed region:** {benchmark['timing_scope']}",
                        f"- **Excluded:** {excluded}",
                        "- **Standard deviation:** population SD across the 100 timed invocations",
                    ])
                )

