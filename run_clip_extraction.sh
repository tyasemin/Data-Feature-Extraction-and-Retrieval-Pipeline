#!/bin/bash

echo ""
echo "Running CLIP feature extraction..."
docker run --rm \
  -v "$(pwd)/istanbul_landmarks_images_extended:/app/images" \
  -v "$(pwd):/app" \
  clip-extractor

echo ""
echo "Feature extraction completed!"
echo "Results saved to: istanbul_landmarks_clip_features.json"