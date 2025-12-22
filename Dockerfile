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
    'elasticsearch>=8.0,<9.0' \
    git+https://github.com/openai/CLIP.git


COPY extract_clip_features.py .


CMD ["python", "extract_clip_features.py"]