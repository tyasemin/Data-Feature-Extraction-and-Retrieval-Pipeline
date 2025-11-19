FROM python:3.9-slim


RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


RUN pip install --no-cache-dir \
    torch \
    torchvision \
    pillow \
    numpy \
    git+https://github.com/openai/CLIP.git


COPY extract_clip_features.py .


RUN mkdir -p /app/images


CMD ["python", "extract_clip_features.py"]