#!/bin/bash

# Usage: ./run_sam_extraction.sh [input_folder] [output_folder]
# Default: ./run_sam_extraction.sh dataset features_sam

INPUT_FOLDER="${1:-dataset}"
OUTPUT_FOLDER="${2:-features_sam}"

echo ""
echo "============================================"
echo "SAM + CLIP Feature Extraction"
echo "============================================"
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "============================================"
echo ""

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "❌ Error: Input folder '$INPUT_FOLDER' not found!"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Build Docker image if needed
echo "Building SAM Docker image..."
cd SAM
docker build -t sam-clip-extractor .
if [ $? -ne 0 ]; then
    echo "❌ Error: Docker build failed!"
    exit 1
fi
cd ..

echo "✓ Docker image built"
echo ""

# Run Docker with GPU support
echo "Running SAM + CLIP extraction..."
echo ""

docker run --rm \
  --gpus all \
  -v "$(pwd)/$INPUT_FOLDER":/dataset:ro \
  -v "$(pwd)/$OUTPUT_FOLDER":/app/features \
  sam-clip-extractor \
  python extract_sam_clip_features.py /dataset /app/features

echo ""
echo "============================================"
echo "Feature extraction completed!"
echo "Segment features saved to: $OUTPUT_FOLDER/"
echo "============================================"
