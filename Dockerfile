FROM python:3.9-slim


RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


RUN pip install --no-cache-dir \
    torch \
    torchvision \
    pillow \
    numpy \
    opencv-python \
    matplotlib \
    'elasticsearch>=8.0,<9.0' \
    git+https://github.com/openai/CLIP.git \
    git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint
RUN mkdir -p /app/sam_models && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O /app/sam_models/sam_vit_b_01ec64.pth

COPY extract_clip_features.py .
COPY search_with_segments.py .


CMD ["python", "extract_clip_features.py"]