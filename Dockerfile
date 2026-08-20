# WACV 2027 Applications Track — Anonymous Supplementary Reproduction Container
# Base Image with CUDA & PyTorch
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for OpenCV and GUI-free headless rendering
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

# Install Python requirements
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
    plotly

# Copy anonymized codebase
COPY . /workspace/

# Expose Streamlit default port
EXPOSE 8501

# Default command: launch interactive dashboard
CMD ["streamlit", "run", "src/fusion/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
