FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other Python packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pillow \
    numpy \
    matplotlib \
    pandas \
    git+https://github.com/facebookresearch/segment-anything.git \
    git+https://github.com/openai/CLIP.git

# Set matplotlib to use Agg backend for headless operation
ENV MPLBACKEND=Agg

# Download SAM model checkpoints (both vit_h and lighter vit_b)
RUN mkdir -p /app/checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /app/checkpoints/sam_vit_h_4b8939.pth && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O /app/checkpoints/sam_vit_b_01ec64.pth

# Copy scripts
COPY extract_sam_clip_features.py .
COPY sam_config.py .
COPY sam_config_lite.py .
COPY test_sam_inference.py .
COPY sam_segment_clip_pipeline.py .

CMD ["python", "extract_sam_clip_features.py"]
