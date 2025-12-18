#!/bin/bash

# Usage: ./run_clip_extraction.sh [input_folder] [output_folder]
# Default: ./run_clip_extraction.sh dataset features

INPUT_FOLDER="${1:-dataset}"
OUTPUT_FOLDER="${2:-features}"

echo ""
echo "Running CLIP feature extraction..."
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo ""

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' not found!"
    exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

docker run --rm \
  --gpus all \
  -v "$(pwd)/$INPUT_FOLDER":/dataset:ro \
  -v "$(pwd)/$OUTPUT_FOLDER":/app/features \
  -v "$(pwd)/label_cleaned.csv":/app/label_cleaned.csv:ro \
  clip-extractor \
  python extract_clip_features.py /dataset /app/features

echo ""
echo "Feature extraction completed!"
echo "Individual JSON files saved to: $OUTPUT_FOLDER/"
