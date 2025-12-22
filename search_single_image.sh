#!/bin/bash

# Usage: ./search_single_image.sh <image_path> [top_k] [visualize]
# Example: ./search_single_image.sh test_images/my_photo.jpg 5
# Example: ./search_single_image.sh test_images/my_photo.jpg 3 --viz

if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_path> [top_k] [--viz]"
    echo ""
    echo "Arguments:"
    echo "  image_path    Path to the test image (required)"
    echo "  top_k         Number of similar images to return (default: 3)"
    echo "  --viz         Create visualization image (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 test_images/my_photo.jpg"
    echo "  $0 test_images/my_photo.jpg 5"
    echo "  $0 test_images/my_photo.jpg 3 --viz"
    exit 1
fi

IMAGE_PATH="$1"
TOP_K="${2:-3}"
VISUALIZE=""

# Check if --viz flag is present
if [ "$3" = "--viz" ] || [ "$2" = "--viz" ]; then
    VISUALIZE="--visualize"
    if [ "$2" = "--viz" ]; then
        TOP_K="3"
    fi
fi

OUTPUT="temp_test/similarity_results.json"
VIZ_OUTPUT="temp_test/similarity_visualization.png"

echo ""
echo "============================================"
echo "Image Similarity Search"
echo "============================================"
echo "Image: $IMAGE_PATH"
echo "Top K: $TOP_K"
echo "Output: $OUTPUT"
if [ -n "$VISUALIZE" ]; then
    echo "Visualization: $VIZ_OUTPUT"
fi
echo "============================================"
echo ""

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Check if Elasticsearch is running
if ! curl -s http://localhost:9201 > /dev/null; then
    echo "❌ Error: Elasticsearch is not running on localhost:9201"
    echo "Start it with: docker compose up -d"
    exit 1
fi

echo "✓ Elasticsearch is running"
echo ""

# Get absolute path for the image
IMAGE_FULL_PATH="$(cd "$(dirname "$IMAGE_PATH")" && pwd)/$(basename "$IMAGE_PATH")"

# Check if the image is inside the workspace
WORKSPACE_DIR="$(pwd)"
if [[ "$IMAGE_FULL_PATH" != "$WORKSPACE_DIR"* ]]; then
    echo "❌ Error: Image must be inside the workspace directory"
    echo "Image path: $IMAGE_FULL_PATH"
    echo "Workspace: $WORKSPACE_DIR"
    exit 1
fi

# Calculate relative path from workspace
IMAGE_REL_PATH="${IMAGE_FULL_PATH#$WORKSPACE_DIR/}"

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT")"

echo "Running similarity search using Docker..."
echo ""

# Run Docker with the image file and search script
# Use the same network as Elasticsearch and connect via container name
# Mount temp_test as writable for output
docker run --rm \
  --network data-feature-extraction-and-retrieval-pipeline_default \
  -v "$WORKSPACE_DIR":/workspace:ro \
  -v "$WORKSPACE_DIR/temp_test":/workspace/temp_test:rw \
  -w /workspace \
  -e ES_HOST=elasticsearch \
  -e ES_PORT=9200 \
  clip-extractor \
  python search_test_image.py "$IMAGE_REL_PATH" --top-k "$TOP_K" --output "$OUTPUT" $VISUALIZE --viz-output "$VIZ_OUTPUT"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && [ -n "$VISUALIZE" ] && [ -f "$VIZ_OUTPUT" ]; then
    echo ""
    echo "✓ Visualization created: $VIZ_OUTPUT"
fi

exit $EXIT_CODE
