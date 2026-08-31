# WACV 2027 Applications Track — Anonymous Supplementary Reproduction Container
# Submission ID: 3068
# Paper: "CMAFM: Cross-Modal Attention Fusion Module for Efficient
#         Real-Time RGB-Thermal Object Detection on Edge Systems"

# ── Base image: PyTorch 2.4 with CUDA 12.1 & cuDNN 8 ──────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="CMAFM — WACV 2027 Supplementary" \
      org.opencontainers.image.description="RGB-Thermal cross-modal attention fusion for object detection" \
      org.opencontainers.image.version="wacv2027-round2"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# KMP workaround for OpenMP conflict on multi-core systems
ENV KMP_DUPLICATE_LIB_OK=TRUE

# ── System dependencies for OpenCV headless & FFmpeg ──────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python dependencies (CPU-compatible subset; GPU picked up from base image) ─
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir \
    numpy<2.0.0 \
    pandas<2.2.0 \
    scipy \
    opencv-python-headless \
    Pillow<11.0.0 \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    streamlit \
    albumentations \
    pycocotools \
    thop \
    psutil \
    python-dotenv \
    plotly>=5,<7 \
    requests \
    setuptools<70.0.0

# ── Copy anonymized codebase ───────────────────────────────────────────────────
COPY . /workspace/

# ── Healthcheck: Streamlit exposes /healthz at runtime ────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# ── Expose Streamlit default port ─────────────────────────────────────────────
EXPOSE 8501

# ── Default: architecture verification (no weights / no GPU required) ─────────
# Override with: docker run ... streamlit run src/fusion/dashboard.py ...
CMD ["python", "run_demo.py"]
