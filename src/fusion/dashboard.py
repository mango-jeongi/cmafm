"""
CMAFM Dashboard — RGB+Thermal Multispectral Object Detection
Real-time video/image upload and object detection visualization
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMAFM Tactical Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Military Dark Navy Theme ──────────────────────────────────────────────────
st.markdown("""
<style>
/* -- Overall Background & Base Text -- */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
.main, .block-container,
section[data-testid="stSidebarContent"],
div[data-testid="stVerticalBlock"] {
    background-color: #1e1e1e !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
}
[data-testid="stHeader"] {
    background-color: #1e1e1e !important;
}

/* -- Sidebar -- */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #252526 !important;
    border-right: 2px solid #6b8f5e !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] * {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Main Text Bold -- */
*, p, label, span, div, li {
    font-weight: bold !important;
}

/* -- Headings -- */
h1, h2, h3, h4, h5, h6 {
    color: #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}

/* -- Captions -- */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #b0c4d8 !important;
    font-weight: bold !important;
}

/* -- Tab Container Background -- */
[data-testid="stTabs"],
[data-testid="stTabs"] > div,
[data-testid="stTabs"] > div > div,
[data-testid="stTabs"] > div > div > div,
div[role="tablist"],
div[role="tablist"]::before,
div[role="tablist"]::after {
    background-color: #1e1e1e !important;
    border-bottom: 1px solid #6b8f5e !important;
}

/* -- Tab Button: Light Background + Dark Text -- */
[data-testid="stTabs"] button,
div[role="tablist"] button {
    background-color: #d4d4d4 !important;
    color: #000000 !important;
    border: 1px solid #c0c0c0 !important;
    border-radius: 4px 4px 0 0 !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}
/* Force tab button text black */
[data-testid="stTabs"] button p,
[data-testid="stTabs"] button span,
[data-testid="stTabs"] button div,
div[role="tablist"] button p,
div[role="tablist"] button span,
div[role="tablist"] button div {
    color: #000000 !important;
    font-weight: 900 !important;
}
/* -- Selected Tab: Gold Background + Dark Text -- */
[data-testid="stTabs"] button[aria-selected="true"],
div[role="tablist"] button[aria-selected="true"] {
    background-color: #6b8f5e !important;
    color: #000000 !important;
    border-bottom: 3px solid #4a6741 !important;
    font-weight: 900 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] p,
[data-testid="stTabs"] button[aria-selected="true"] span,
[data-testid="stTabs"] button[aria-selected="true"] div,
div[role="tablist"] button[aria-selected="true"] p,
div[role="tablist"] button[aria-selected="true"] span,
div[role="tablist"] button[aria-selected="true"] div {
    color: #000000 !important;
    font-weight: 900 !important;
}

/* -- Tab Content Area Background -- */
[data-testid="stTabsTabPanel"],
div[role="tabpanel"] {
    background-color: #1e1e1e !important;
}

/* -- Default Button -- */
[data-testid="stButton"] button {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    border: 1px solid #37373d !important;
    font-weight: bold !important;
}
/* -- Primary Button -- */
[data-testid="stButton"] button[kind="primary"] {
    background-color: #37373d !important;
    color: #6b8f5e !important;
    border: 2px solid #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #6b8f5e !important;
    color: #1e1e1e !important;
}
/* -- Download Button -- */
[data-testid="stDownloadButton"] button {
    background-color: #252526 !important;
    color: #6b8f5e !important;
    border: 1px solid #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: bold !important;
}

/* -- Metric Cards -- */
[data-testid="stMetric"] {
    background-color: #252526 !important;
    border: 1px solid #37373d !important;
    border-left: 4px solid #6b8f5e !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
}
[data-testid="stMetricLabel"] {
    color: #b0c4d8 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.75rem !important;
    font-weight: bold !important;
}
[data-testid="stMetricValue"] {
    color: #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
}

/* -- Input Fields -- */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
textarea {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    border: 1px solid #37373d !important;
    font-weight: bold !important;
}
[data-testid="stSlider"] { accent-color: #6b8f5e; }

/* -- DataFrame -- */
[data-testid="stDataFrame"] { border: 1px solid #37373d !important; }
.stDataFrame thead th {
    background-color: #37373d !important;
    color: #6b8f5e !important;
    font-weight: 900 !important;
}
.stDataFrame tbody td {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Notification Box -- */
[data-testid="stAlert"] {
    background-color: #252526 !important;
    border-left: 4px solid #6b8f5e !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Divider -- */
hr { border-color: #37373d !important; }

/* -- Checkbox / Slider -- */
[data-testid="stCheckbox"] { accent-color: #6b8f5e; }

/* -- Progress Bar -- */
[data-testid="stProgressBar"] > div > div {
    background-color: #6b8f5e !important;
}

/* -- File Uploader -- */
[data-testid="stFileUploader"] {
    background-color: #252526 !important;
    border: 2px dashed #37373d !important;
    border-radius: 4px !important;
}

/* -- Radio Button -- */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span,
[data-testid="stRadio"] p {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Checkbox Text -- */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] p {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Selectbox Text -- */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #252526 !important;
    border: 1px solid #37373d !important;
}

/* -- Slider Label & Value -- */
[data-testid="stSlider"] label,
[data-testid="stSlider"] span,
[data-testid="stSlider"] p {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── number input ── */
[data-testid="stNumberInput"] input {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
    border: 1px solid #37373d !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- File Uploader Text -- */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}
[data-testid="stFileUploader"] section {
    background-color: #252526 !important;
    border: 2px dashed #37373d !important;
}

/* -- Text Input Label -- */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Text Inside Tab Panels -- */
[data-testid="stTabsTabPanel"] p,
[data-testid="stTabsTabPanel"] span,
[data-testid="stTabsTabPanel"] label,
[data-testid="stTabsTabPanel"] div {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Markdown Text -- */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Box Inside Text (Success/Warning/Error/Info) -- */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* -- Spinner Text -- */
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {
    color: #6b8f5e !important;
    font-weight: bold !important;
}

/* -- Progress Bar Text -- */
[data-testid="stProgressBar"] + div,
[data-testid="stProgressBar"] ~ p {
    color: #6b8f5e !important;
    font-weight: bold !important;
}

/* -- Empty State / Guidelines -- */
[data-testid="stEmpty"] p {
    color: #b0c4d8 !important;
    font-weight: bold !important;
}

/* -- Bar Chart Label -- */
[data-testid="stVegaLiteChart"] text {
    fill: #d4d4d4 !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = {1: "People", 2: "Car", 3: "Bus", 4: "Motorcycle", 5: "Lamp", 6: "Truck"}
CLASS_COLORS = {
    1: (0,   255,  80),   # green
    2: (255, 80,   0),    # orange-red
    3: (0,   160, 255),   # blue
    4: (255, 220,  0),    # yellow
    5: (200,   0, 255),   # purple
    6: (0,   220, 200),   # teal
}
IMG_SIZE = (640, 640)

DEFAULT_CKPT = str(Path(__file__).resolve().parents[2] / "runs" / "best.pth")

# ── Session state ─────────────────────────────────────────────────────────────
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
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model...")
def load_model_cached(ckpt_path: str, device_str: str, model_type: str = "Faster R-CNN (CMAFM)"):
    from config import Config
    from model import build_model

    device = torch.device(device_str)
    
    if model_type == "CMAFM-YOLO":
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        cft_dir = str(repo_root / "cft_engine")
        if cft_dir not in sys.path:
            sys.path.append(cft_dir)
            
        from models.experimental import attempt_load
        model = attempt_load(ckpt_path, map_location=device)
        model.to(device)
        model.eval()
        return model, Config(), device
    else:
        cfg = Config()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "config" in ckpt:
            cfg = ckpt["config"]

        model = build_model(cfg.model, num_classes=cfg.data.num_classes)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model, cfg, device


@st.cache_resource(show_spinner="Loading single-modality model...")
def load_single_modal_models(ckpt_path: str, device_str: str):
    """Load RGB-only / Thermal-only ablation checkpoints."""
    from config import Config
    from ablation_models import SingleModalDetector

    device = torch.device(device_str)

    # Extract config from fused model checkpoint
    cfg = Config()
    fusion_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in fusion_ckpt:
        cfg = fusion_ckpt["config"]

    ablation_dir = Path(ckpt_path).parent / "ablation"
    rgb_ckpt_path = ablation_dir / "rgb_only_best.pth"
    th_ckpt_path  = ablation_dir / "thermal_only_best.pth"

    def _load(modality, ckpt_p):
        m = SingleModalDetector(cfg.model, num_classes=cfg.data.num_classes, modality=modality)
        if ckpt_p.exists():
            ck = torch.load(str(ckpt_p), map_location=device, weights_only=False)
            state = ck.get("model", ck)
            m.load_state_dict(state)
        m.to(device).eval()
        return m

    rgb_model = _load("rgb",     rgb_ckpt_path)
    th_model  = _load("thermal", th_ckpt_path)
    return rgb_model, th_model, device


@torch.no_grad()
def run_single_inference(model, rgb_t, th_t, device):
    rgb_t = rgb_t.to(device)
    th_t  = th_t.to(device)
    outputs = model(rgb_t, th_t)
    return outputs[0]


def preprocess_pair(rgb_np: np.ndarray, thermal_np: np.ndarray):
    """numpy RGB (H,W,3) + thermal (H,W) or (H,W,3) → tensors."""
    orig_h, orig_w = rgb_np.shape[:2]

    rgb_r  = cv2.resize(rgb_np,      (IMG_SIZE[1], IMG_SIZE[0]))
    if thermal_np.ndim == 3:
        thermal_gray = cv2.cvtColor(thermal_np, cv2.COLOR_RGB2GRAY)
    else:
        thermal_gray = thermal_np
    th_r   = cv2.resize(thermal_gray, (IMG_SIZE[1], IMG_SIZE[0]))

    rgb_t = torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0
    th_t  = torch.from_numpy(th_r).unsqueeze(0).float() / 255.0
    th_t  = th_t.repeat(3, 1, 1)

    return rgb_t.unsqueeze(0), th_t.unsqueeze(0), orig_h, orig_w


@torch.no_grad()
def run_inference(model, rgb_t, th_t, device):
    if st.session_state.get("model_type", "Faster R-CNN (CMAFM)") == "CMAFM-YOLO":
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        cft_dir = str(repo_root / "cft_engine")
        if cft_dir not in sys.path:
            sys.path.append(cft_dir)
        from utils.general import non_max_suppression

        is_half = next(model.parameters()).dtype == torch.float16
        rgb_t = rgb_t.to(device)
        th_t  = th_t.to(device)
        if is_half:
            rgb_t = rgb_t.half()
            th_t  = th_t.half()
        else:
            rgb_t = rgb_t.float()
            th_t  = th_t.float()
            
        pred = model(rgb_t, th_t)[0]
        preds = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45)
        p = preds[0]
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
        rgb_t = rgb_t.to(device)
        th_t  = th_t.to(device)
        outputs = model(rgb_t, th_t)
        return outputs[0]


def draw_detections(rgb_np, detections, orig_h, orig_w, score_thresh=0.5):
    """Returns annotated BGR image + list of detection dicts."""
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

        color = CLASS_COLORS.get(int(label), (255, 255, 255))
        name  = CLASS_NAMES.get(int(label), f"cls{label}")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        results.append({"class": name, "score": float(score),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), results


def frame_to_np(uploaded_file):
    """Convert uploaded image file → RGB numpy array."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thermal_to_np(uploaded_file):
    """Convert uploaded thermal file → grayscale numpy array."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def detection_summary(results):
    """Count detections per class."""
    from collections import Counter
    counts = Counter(r["class"] for r in results)
    return dict(counts)


def run_three_way_detection(rgb_np, th_np, score_thresh, thermal_source=""):
    """Display RGB-only, Thermal-only, and Fused detection results in 3 columns."""
    import pandas as pd

    device = st.session_state.device
    rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)

    t0 = time.perf_counter()
    dets_rgb = run_single_inference(st.session_state.rgb_only_model, rgb_t, th_t, device)
    elapsed_rgb = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dets_th = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
    elapsed_th = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dets_fusion = run_inference(st.session_state.model, rgb_t, th_t, device)
    elapsed_fusion = (time.perf_counter() - t0) * 1000

    vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)
    vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)

    th_display = cv2.cvtColor(cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    th_display = cv2.resize(th_display, (orig_w, orig_h))
    vis_th, results_th = draw_detections(th_display, dets_th, orig_h, orig_w, score_thresh)

    st.markdown("---")
    if thermal_source:
        st.caption(f"Thermal source: {thermal_source}")

    # -- Result Images (Top) --
    col_r, col_t, col_f = st.columns(3)
    with col_r:
        st.markdown("##### RGB-only")
        st.image(vis_rgb, use_container_width=True)
    with col_t:
        st.markdown("##### Thermal-only")
        st.image(vis_th, use_container_width=True)
    with col_f:
        st.markdown("##### RGB+Thermal Fusion (CMAFM)")
        st.image(vis_fusion, use_container_width=True)

    # -- Quantitative Metrics (Bottom) --
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RGB Detections", len(results_rgb))
    m2.metric("RGB Inference", f"{elapsed_rgb:.1f} ms")
    m3.metric("Thermal Detections", len(results_th))
    m4.metric("Thermal Inference", f"{elapsed_th:.1f} ms")
    m5.metric("Fusion Detections", len(results_fusion))
    m6.metric("Fusion Inference", f"{elapsed_fusion:.1f} ms")

    # Class distribution graph
    if results_fusion:
        import plotly.graph_objects as go
        summary = detection_summary(results_fusion)
        _cls_hex = {n: "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i])
                    for i, n in CLASS_NAMES.items()}
        bar_colors = [_cls_hex.get(c, "#6b8f5e") for c in summary.keys()]
        fig = go.Figure(go.Bar(
            x=list(summary.keys()), y=list(summary.values()),
            marker_color=bar_colors, marker_line_width=0,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d4d4d4", size=11),
            margin=dict(l=40, r=20, t=20, b=40), height=220,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#aaa")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(color="#888")),
            showlegend=False,
        )
        st.caption("Detections by Class (Fusion Model)")
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Fusion Table
    st.subheader("Fusion Model Detailed Results")
    if results_fusion:
        df = pd.DataFrame(results_fusion)
        df.index += 1
        df.columns = ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.3f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"No detections found with confidence >= {score_thresh:.2f}.")

    # Download fusion result image
    result_bgr = cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    st.download_button("💾 Download Fusion Result Image", data=buf.tobytes(),
                       file_name="detection_fusion.jpg", mime="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# UI — Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <div style='font-size:2.5rem;'>🎯</div>
        <div style='color:#6b8f5e; font-family:Courier New; font-size:1.1rem; font-weight:bold; letter-spacing:2px;'>
            TACTICAL SYSTEM
        </div>
        <div style='color:#a0b4c8; font-family:Courier New; font-size:0.7rem; letter-spacing:1px;'>
            CMAFM · RGB+LWIR FUSION
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Model ──
    st.subheader("🔧 Model Deployment")
    
    model_type = st.selectbox(
        "Model Architecture",
        ["Faster R-CNN (CMAFM)", "CMAFM-YOLO"],
        help="Select the fusion model architecture to deploy."
    )
    st.session_state.model_type = model_type
    
    # Invalidate cache when model selection changes
    if "prev_model_type" in st.session_state and st.session_state.prev_model_type != model_type:
        st.cache_resource.clear()
    st.session_state.prev_model_type = model_type
    
    if model_type == "Faster R-CNN (CMAFM)":
        default_path = DEFAULT_CKPT
    else:
        default_path = str(Path(__file__).resolve().parents[2] / "weights" / "CMAFM_Pretrained.pt")

    use_default_ckpt = st.checkbox(f"Use default path ({Path(default_path).name})",
                                   value=Path(default_path).exists())
    if use_default_ckpt:
        ckpt_path = default_path
        if Path(ckpt_path).exists():
            st.success(f"✅ Model weights located: `{Path(ckpt_path).name}`")
        else:
            st.error(f"❌ default path not found — Enter path manually")
            ckpt_path = st.text_input("Checkpoint Path", value="")
    else:
        ckpt_path = st.text_input("Checkpoint Path", value=default_path)

    # -- Device (Auto selection) --
    st.subheader("⚡ Computing Device")
    cuda_avail = torch.cuda.is_available()
    device_str = "cuda" if cuda_avail else "cpu"
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"CUDA ✔  {gpu_name}")
    else:
        st.info("CPU Mode (No CUDA)")

    # ── Load model button ──
    if st.button("🚀 Start System", type="primary", use_container_width=True):
        if not ckpt_path or not Path(ckpt_path).exists():
            st.error("Checkpoint file not found.")
        else:
            with st.spinner("Loading model..."):
                model, cfg, device = load_model_cached(ckpt_path, device_str, model_type)
                st.session_state.model  = model
                st.session_state.device = device
                st.session_state.cfg    = cfg
                
                # Load unimodal Faster R-CNN models for comparison
                fr_ckpt_path = DEFAULT_CKPT
                rgb_m, th_m, _ = load_single_modal_models(fr_ckpt_path, device_str)
                st.session_state.rgb_only_model     = rgb_m
                st.session_state.thermal_only_model = th_m
            st.success(f"✅ System initialised ({model_type} + RGB-only + Thermal-only)")

    st.markdown("---")

    # ── Inference params ──
    st.subheader("🎚️ Sensitivity Settings")
    score_thresh = st.slider("Score Threshold", 0.1, 0.95, 0.5, 0.05)

    st.markdown("---")
    st.subheader("🏷️ Target Classes")
    for cid, cname in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS[cid]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(
            f'<span style="background:{hex_color};border-radius:3px;'
            f'padding:3px 12px;color:#000;font-weight:bold;'
            f'font-family:Courier New;letter-spacing:1px;">{cname}</span>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# UI — Main Area
# ══════════════════════════════════════════════════════════════════════════════

_photo_path = Path(__file__).parent / "ltc_author_a.png"
_photo_b64 = ""
if _photo_path.exists():
    import base64 as _b64
    _photo_b64 = _b64.b64encode(_photo_path.read_bytes()).decode()

_photo_html = (
    f"<img src='data:image/png;base64,{_photo_b64}' "
    f"style='width:90px; height:110px; object-fit:cover; object-position:top; "
    f"border-radius:4px; opacity:0.85; filter:sepia(30%);'/>"
    if _photo_b64 else ""
)

st.markdown(f"""
<div style='border:2px solid #6b8f5e; border-radius:6px; padding:16px 24px 10px 24px; margin-bottom:8px;
            background:linear-gradient(90deg,#252526 0%,#0a1628 100%);
            display:flex; align-items:center; justify-content:space-between;'>
    <div>
        <div style='font-size:2.6rem; font-weight:900; color:#6b8f5e;
                    font-family:Courier New; letter-spacing:4px;'>
            🎯 CMAFM TACTICAL DETECTION SYSTEM
        </div>
        <div style='color:#a0b4c8; font-family:Courier New; font-size:0.85rem; margin-top:4px; letter-spacing:1px;'>
            ◆ CROSS-MODAL ATTENTION FUSION &nbsp;|&nbsp; RGB + LWIR MULTISPECTRAL &nbsp;|&nbsp; M3FD DATASET
        </div>
        <div style='color:#556677; font-family:Courier New; font-size:0.7rem; margin-top:8px; letter-spacing:1px;'>
            by. LTC Bansungwoo
        </div>
    </div>
    <div style='margin-left:24px; flex-shrink:0;'>
        {_photo_html}
    </div>
</div>
""", unsafe_allow_html=True)

model_ready = st.session_state.model is not None

if not model_ready:
    st.warning("⚠️ System standby — Click **Start System** in the sidebar to activate.")

# ── Mode selection ──
tab_image, tab_video, tab_webcam = st.tabs(["📡 Image Detection", "📹 Video Tracking", "🎖️ Sample Test"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Image Detection
# ══════════════════════════════════════════════════════════════════════════════

with tab_image:
    st.subheader("📡 Static Image Target Detection")

    st.caption("Upload RGB and Thermal images respectively.")
    col_upload_rgb, col_upload_th = st.columns(2)
    with col_upload_rgb:
        rgb_file = st.file_uploader("📷 RGB Image",
                                     type=["jpg", "jpeg", "png", "bmp"],
                                     key="img_rgb")
    with col_upload_th:
        th_file  = st.file_uploader("🌡️ Thermal Image",
                                     type=["jpg", "jpeg", "png", "bmp"],
                                     key="img_th")

    if rgb_file and th_file:
        col_prev_r, col_prev_t = st.columns(2)
        rgb_file.seek(0); th_file.seek(0)
        with col_prev_r:
            st.image(rgb_file, caption="RGB Input", use_container_width=True)
        with col_prev_t:
            st.image(th_file,  caption="Thermal Input", use_container_width=True)

    run_img = st.button("🔍 Run Detection", type="primary",
                         disabled=(not model_ready or rgb_file is None or th_file is None),
                         key="btn_img")

    if run_img and rgb_file and th_file:
        rgb_file.seek(0); th_file.seek(0)
        rgb_np = frame_to_np(rgb_file)
        th_np  = thermal_to_np(th_file)
        run_three_way_detection(rgb_np, th_np, score_thresh, "Uploaded Image")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Video Detection
# ══════════════════════════════════════════════════════════════════════════════

with tab_video:
    st.subheader("📹 Dynamic Target Tracking & Detection")

    st.markdown("> ⚠️ **Both videos must have the same number of frames and resolution.**")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        rgb_vid = st.file_uploader("📹 RGB Video", type=["mp4", "avi", "mov", "mkv"],
                                    key="vid_rgb")
    with col_v2:
        th_vid  = st.file_uploader("🌡️ Thermal Video", type=["mp4", "avi", "mov", "mkv"],
                                    key="vid_th")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        max_frames = st.number_input("Maximum frames to process (0 = all)",
                                      min_value=0, max_value=10000, value=100, step=10)
    with col_opt2:
        frame_skip = st.number_input("Frame skip (process 1 in every N frames)",
                                      min_value=1, max_value=30, value=1, step=1)

    vid_ready = model_ready and rgb_vid is not None and th_vid is not None
    run_vid = st.button("🎬 Start Video Detection", type="primary",
                         disabled=not vid_ready,
                         key="btn_vid")

    if run_vid and rgb_vid and th_vid:
        # Save RGB/Thermal to temp files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as rf:
            rf.write(rgb_vid.read()); rgb_tmp = rf.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(th_vid.read()); th_tmp = tf.name

        cap_r = cv2.VideoCapture(rgb_tmp)
        cap_t = cv2.VideoCapture(th_tmp) if th_tmp else None

        total_frames = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in       = cap_r.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_process = total_frames if max_frames == 0 else min(total_frames, max_frames * frame_skip)

        # Output video
        # 3 video output files (record with mp4v, then re-encode with ffmpeg to H.264)
        raw_rgb_tmp    = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        raw_th_tmp     = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        raw_fusion_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_rgb_tmp    = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_th_tmp     = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_fusion_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps_in / frame_skip
        writer_rgb    = cv2.VideoWriter(raw_rgb_tmp,    fourcc, out_fps, (width, height))
        writer_th     = cv2.VideoWriter(raw_th_tmp,     fourcc, out_fps, (width, height))
        writer_fusion = cv2.VideoWriter(raw_fusion_tmp, fourcc, out_fps, (width, height))

        st.markdown("---")
        prog_bar  = st.progress(0, text="Processing...")
        # Live preview 3 columns
        prev_cols   = st.columns(3)
        prev_rgb    = prev_cols[0].empty()
        prev_th     = prev_cols[1].empty()
        prev_fusion = prev_cols[2].empty()
        prev_cols[0].caption("RGB-only")
        prev_cols[1].caption("Thermal-only")
        prev_cols[2].caption("Fusion (CMAFM)")

        frame_idx    = 0
        proc_count   = 0
        total_dets   = 0
        total_time   = 0.0
        all_results  = []
        device       = st.session_state.device
        # Time-series data for plotting
        log_frames, log_dets, log_ms = [], [], []
        # Class-wise frame time-series
        log_cls = {name: [] for name in CLASS_NAMES.values()}
        # Event log (chronological annotations)
        event_log = []   # [(frame, timestamp_str, event_str)]

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

            t0 = time.perf_counter()
            dets_rgb    = run_single_inference(st.session_state.rgb_only_model,     rgb_t, th_t, device)
            dets_th     = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
            dets_fusion = run_inference(st.session_state.model,                     rgb_t, th_t, device)
            elapsed = (time.perf_counter() - t0) * 1000
            total_time += elapsed

            vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)
            vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)

            th_display = cv2.cvtColor(cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            th_display = cv2.resize(th_display, (orig_w, orig_h))
            vis_th, results_th = draw_detections(th_display, dets_th, orig_h, orig_w, score_thresh)

            writer_rgb.write(cv2.cvtColor(vis_rgb,    cv2.COLOR_RGB2BGR))
            writer_th.write(cv2.cvtColor(vis_th,     cv2.COLOR_RGB2BGR))
            writer_fusion.write(cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR))

            all_results.extend(results_fusion)
            total_dets += len(results_fusion)
            proc_count += 1

            # Time-series log
            log_frames.append(frame_idx)
            log_dets.append(len(results_fusion))
            log_ms.append(round(elapsed, 1))

            # Class-wise counts
            from collections import Counter as _Counter
            frame_cls = _Counter(r["class"] for r in results_fusion)
            for cname in CLASS_NAMES.values():
                log_cls[cname].append(frame_cls.get(cname, 0))

            # Event: record new class appearance or detection spike
            ts_str = f"{frame_idx / max(fps_in, 1):.1f}s"
            new_cls = [c for c in frame_cls if frame_cls[c] > 0 and
                       sum(log_cls[c][:-1]) == 0]  # First appearance in this frame
            if new_cls:
                event_log.append((frame_idx, ts_str,
                                  f"First Detection: {', '.join(new_cls)}"))
            if len(log_dets) >= 2 and log_dets[-1] >= log_dets[-2] * 2 and log_dets[-1] >= 3:
                event_log.append((frame_idx, ts_str,
                                  f"Detection Spike: {log_dets[-2]} -> {log_dets[-1]} targets"))

            # Live preview every 5 frames
            if proc_count % 5 == 1:
                prev_rgb.image(vis_rgb,    caption=f"Frame {frame_idx} | {len(results_rgb)} targets",    use_container_width=True)
                prev_th.image(vis_th,     caption=f"Frame {frame_idx} | {len(results_th)} targets",     use_container_width=True)
                prev_fusion.image(vis_fusion, caption=f"Frame {frame_idx} | {len(results_fusion)} targets", use_container_width=True)

            avg_ms = total_time / proc_count
            prog_bar.progress(
                min(frame_idx / max(frames_to_process - 1, 1), 1.0),
                text=f"Frame {frame_idx}/{frames_to_process} | Average {avg_ms:.1f} ms | Fusion Detections {total_dets}"
            )
            frame_idx += 1

        cap_r.release()
        if cap_t is not None:
            cap_t.release()
        writer_rgb.release()
        writer_th.release()
        writer_fusion.release()

        # mp4v -> H.264 re-encoding (browser playback compatibility)
        import shutil as _shutil
        _ffmpeg = _shutil.which("ffmpeg")
        _has_ffmpeg = _ffmpeg is not None and Path(_ffmpeg).exists()

        prog_bar.progress(1.0, text="Encoding to H.264..." if _has_ffmpeg else "Done!")

        def _reencode(src, dst):
            import subprocess
            subprocess.run(
                [_ffmpeg, "-y", "-i", src,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", dst],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        if _has_ffmpeg:
            for src, dst in [(raw_rgb_tmp, out_rgb_tmp),
                              (raw_th_tmp,  out_th_tmp),
                              (raw_fusion_tmp, out_fusion_tmp)]:
                _reencode(src, dst)
        else:
            # If ffmpeg is missing, use the raw files directly
            import shutil as _sh
            for src, dst in [(raw_rgb_tmp, out_rgb_tmp),
                              (raw_th_tmp,  out_th_tmp),
                              (raw_fusion_tmp, out_fusion_tmp)]:
                _sh.copy(src, dst)

        prog_bar.progress(1.0, text="Done!")
        st.success(f"✅ Video processing complete — {proc_count} frames, {total_dets} fusion detections")

        # Playback Result Video
        st.subheader("Playback Result Video")

        # Top row (2 columns): RGB-only / Thermal-only
        col_r, col_t = st.columns(2)
        for col, path, label in [
            (col_r, out_rgb_tmp, "RGB-only"),
            (col_t, out_th_tmp,  "Thermal-only"),
        ]:
            with open(path, "rb") as f:
                vid_bytes = f.read()
            col.markdown(f"##### {label}")
            col.video(vid_bytes)
            col.download_button(f"💾 Download {label}",
                                data=vid_bytes,
                                file_name=f"detection_{label}.mp4",
                                mime="video/mp4",
                                key=f"dl_{label}")

        # Bottom row (1 column): Fusion (centered & scaled)
        st.markdown("---")
        _, col_f, _ = st.columns([1, 4, 1])
        with open(out_fusion_tmp, "rb") as f:
            fusion_bytes = f.read()
        col_f.markdown("##### RGB+Thermal Fusion (CMAFM)")
        col_f.video(fusion_bytes)
        col_f.download_button("💾 Download Fusion (CMAFM)",
                              data=fusion_bytes,
                              file_name="detection_fusion_cmafm.mp4",
                              mime="video/mp4",
                              key="dl_fusion")

        # -- Frame-wise detection graphs --
        if log_frames:
            import pandas as pd
            import plotly.graph_objects as go
            st.markdown("---")
            st.subheader("📊 Frame-wise Detection Statistics")

            _chart_layout = dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d4d4d4", size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                height=220,
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False, showline=False, tickfont=dict(color="#888"),
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False, showline=False, tickfont=dict(color="#888"),
                ),
                showlegend=False,
            )

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.caption("Total detected objects per frame")
                fig1 = go.Figure(go.Scatter(
                    x=log_frames, y=log_dets, mode="lines",
                    line=dict(color="#ffffaa", width=1.5),
                    fill=None,
                ))
                fig1.update_layout(**_chart_layout)
                st.plotly_chart(fig1, use_container_width=True)
            with col_g2:
                st.caption("Inference time per frame (ms)")
                fig2 = go.Figure(go.Scatter(
                    x=log_frames, y=log_ms, mode="lines",
                    line=dict(color="#5e7a8f", width=1.5),
                    fill=None,
                ))
                fig2.update_layout(**_chart_layout)
                st.plotly_chart(fig2, use_container_width=True)

            # -- Class-wise line graphs --
            st.caption("Class-wise frame detections (line)")
            _cls_hex = {n: "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i])
                        for i, n in CLASS_NAMES.items()}
            fig_cls = go.Figure()
            for cname, vals in log_cls.items():
                if any(v > 0 for v in vals):
                    fig_cls.add_trace(go.Scatter(
                        x=log_frames, y=vals, mode="lines",
                        name=cname,
                        line=dict(color=_cls_hex.get(cname, "#aaa"), width=1.5),
                    ))
            _cls_layout = dict(**_chart_layout)
            _cls_layout["height"] = 260
            _cls_layout["showlegend"] = True
            _cls_layout["legend"] = dict(
                font=dict(color="#d4d4d4", size=10),
                bgcolor="rgba(0,0,0,0)",
                orientation="h", yanchor="bottom", y=1.02,
            )
            fig_cls.update_layout(**_cls_layout)
            st.plotly_chart(fig_cls, use_container_width=True)

            # -- Event Log --
            if event_log:
                st.caption("📋 Event log by timestamp")
                for (fidx, ts, msg) in event_log:
                    st.markdown(
                        f"<div style='font-family:Courier New; font-size:0.78rem; "
                        f"color:#a0b4c8; padding:2px 0;'>"
                        f"<span style='color:#6b8f5e;'>▶ {ts}</span>"
                        f"&nbsp;&nbsp;Frame {fidx:04d}&nbsp;&nbsp;{msg}</div>",
                        unsafe_allow_html=True
                    )

        # Class distribution
        if all_results:
            import pandas as pd
            import plotly.graph_objects as go
            from collections import Counter
            counts = Counter(r["class"] for r in all_results)
            df_sum = pd.DataFrame(counts.items(), columns=["Class", "Detections"]).sort_values("Detections", ascending=False)
            st.subheader("Overall Class-wise Detection Statistics (Fusion Model)")
            _cls_hex = {n: "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i])
                        for i, n in CLASS_NAMES.items()}
            bar_colors3 = [_cls_hex.get(c, "#6b8f5e") for c in df_sum["Class"]]
            fig3 = go.Figure(go.Bar(
                x=df_sum["Class"], y=df_sum["Detections"],
                marker_color=bar_colors3, marker_line_width=0,
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d4d4d4", size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                height=260,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#aaa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(color="#888")),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Clean up temporary files
        for p in [rgb_tmp, raw_rgb_tmp, raw_th_tmp, raw_fusion_tmp,
                  out_rgb_tmp, out_th_tmp, out_fusion_tmp]:
            try:
                os.unlink(p)
            except Exception:
                pass
        if th_tmp:
            try:
                os.unlink(th_tmp)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sample Test (random sample from dataset)
# ══════════════════════════════════════════════════════════════════════════════

with tab_webcam:
    st.subheader("🎖️ Tactical Dataset Sample Test")
    st.markdown("Select a sample from the M3FD dataset to compare RGB-only, Thermal-only, and Fused detections.")

    data_root = Path(__file__).parent / "data" / "M3FD"
    vis_dir   = data_root / "Vis"
    ir_dir    = data_root / "Ir"

    has_data = vis_dir.exists() and ir_dir.exists()
    if not has_data:
        st.warning("⚠️ `data/M3FD/Vis` or `data/M3FD/Ir` directories are missing. "
                   "Please place the dataset first.")
    else:
        rgb_files = sorted(vis_dir.glob("*.png")) + sorted(vis_dir.glob("*.jpg"))
        st.caption(f"Dataset: {len(rgb_files)} samples found")

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            sample_idx = st.slider("Sample Index", 0, max(0, len(rgb_files) - 1), 0)
        with col_s2:
            if st.button("🎲 Select Random", use_container_width=True):
                sample_idx = int(np.random.randint(0, len(rgb_files)))
                st.rerun()

        if rgb_files:
            chosen_rgb = rgb_files[sample_idx]
            chosen_th  = ir_dir / chosen_rgb.name

            col_s_r, col_s_t = st.columns(2)
            with col_s_r:
                st.image(str(chosen_rgb), caption=f"RGB: {chosen_rgb.name}",
                          use_container_width=True)
            with col_s_t:
                if chosen_th.exists():
                    st.image(str(chosen_th), caption=f"Thermal: {chosen_th.name}",
                              use_container_width=True)
                else:
                    st.error("Corresponding Thermal image not found.")

            run_sample = st.button("🔍 Run Sample Detection", type="primary",
                                    disabled=(not model_ready or not chosen_th.exists()),
                                    key="btn_sample")

            if run_sample:
                rgb_np = cv2.cvtColor(cv2.imread(str(chosen_rgb)), cv2.COLOR_BGR2RGB)
                th_np  = cv2.imread(str(chosen_th), cv2.IMREAD_GRAYSCALE)

                rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)
                device = st.session_state.device

                # -- Three models inference --
                t0 = time.perf_counter()
                dets_fusion = run_inference(st.session_state.model, rgb_t, th_t, device)
                elapsed_fusion = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                dets_rgb = run_single_inference(st.session_state.rgb_only_model, rgb_t, th_t, device)
                elapsed_rgb = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                dets_th = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
                elapsed_th = (time.perf_counter() - t0) * 1000

                vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)
                vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)

                # Display thermal result on thermal background
                th_display = cv2.cvtColor(
                    cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
                )
                th_display_resized = cv2.resize(th_display, (orig_w, orig_h))
                vis_th, results_th = draw_detections(th_display_resized, dets_th, orig_h, orig_w, score_thresh)

                st.markdown("---")

                # -- Metrics Row --
                st.subheader("Detection Comparison")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("RGB-only Detections", len(results_rgb))
                m2.metric("RGB Inference", f"{elapsed_rgb:.1f} ms")
                m3.metric("Thermal-only Detections", len(results_th))
                m4.metric("Thermal Inference", f"{elapsed_th:.1f} ms")
                m5.metric("Fusion Detections", len(results_fusion))
                m6.metric("Fusion Inference", f"{elapsed_fusion:.1f} ms")

                # -- 3-column Result Images --
                col_r, col_t, col_f = st.columns(3)
                with col_r:
                    st.markdown("##### RGB-only")
                    st.image(vis_rgb, use_container_width=True)
                    st.caption(f"Detections: {len(results_rgb)}")
                with col_t:
                    st.markdown("##### Thermal-only")
                    st.image(vis_th, use_container_width=True)
                    st.caption(f"Detections: {len(results_th)}")
                with col_f:
                    st.markdown("##### RGB+Thermal Fusion (CMAFM)")
                    st.image(vis_fusion, use_container_width=True)
                    st.caption(f"Detections: {len(results_fusion)}")

                # -- Detailed Fusion Results Table --
                st.markdown("---")
                st.subheader("Fusion Model Detailed Results")
                if results_fusion:
                    import pandas as pd
                    df = pd.DataFrame(results_fusion)
                    df.index += 1
                    df.columns = ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
                    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.3f}")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning(f"No detections found with confidence >= {score_thresh:.2f}.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#a0b4c8; font-family:Courier New; font-size:0.75rem; letter-spacing:1px;'>
    🎯 CMAFM &nbsp;·&nbsp; CROSS-MODAL ATTENTION FUSION MODEL &nbsp;·&nbsp; M3FD DATASET<br>
    ResNet-50 DUAL BACKBONE + FPN + Faster R-CNN &nbsp;·&nbsp; RGB + LWIR MULTISPECTRAL
</div>
""", unsafe_allow_html=True)
