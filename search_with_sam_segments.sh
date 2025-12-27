#!/bin/bash

# Usage: ./search_with_sam_segments.sh <image_path> [mode] [top_k] [tags...]
# Example: ./search_with_sam_segments.sh test_images/my_photo.jpg hybrid 10
# Example: ./search_with_sam_segments.sh test_images/my_photo.jpg segment 5 --tags mosque tower

if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_path> [mode] [top_k] [--tags tag1 tag2 ...]"
    echo ""
    echo "Arguments:"
    echo "  image_path    Path to the test image (required)"
    echo "  mode          Search mode: whole|segment|hybrid|tags (default: hybrid)"
    echo "  top_k         Number of similar images to return (default: 10)"
    echo "  --tags        Optional tags to filter by or search for"
    echo ""
    echo "Examples:"
    echo "  $0 test_images/my_photo.jpg"
    echo "  $0 test_images/my_photo.jpg segment 10"
    echo "  $0 test_images/my_photo.jpg hybrid 5 --tags mosque dome"
    echo "  $0 --mode tags --tags mosque tower minaret"
    exit 1
fi

# Parse arguments
IMAGE_PATH=""
MODE="hybrid"
TOP_K="10"
TAGS=""
PARSE_TAGS=false

for arg in "$@"; do
    if [ "$PARSE_TAGS" = true ]; then
        TAGS="$TAGS $arg"
    elif [ "$arg" = "--tags" ]; then
        PARSE_TAGS=true
    elif [ "$arg" = "--mode" ]; then
        continue
    elif [ -z "$IMAGE_PATH" ] && [ "$arg" != "tags" ]; then
        IMAGE_PATH="$arg"
    elif [[ "$arg" =~ ^(whole|segment|hybrid|tags)$ ]]; then
        MODE="$arg"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        TOP_K="$arg"
    fi
done

echo ""
echo "============================================"
echo "Advanced Image Search with SAM Segments"
echo "============================================"
echo "Mode: $MODE"
echo "Top K: $TOP_K"
if [ -n "$TAGS" ]; then
    echo "Tags filter:$TAGS"
fi
if [ -n "$IMAGE_PATH" ]; then
    echo "Query image: $IMAGE_PATH"
fi
echo "============================================"
echo ""

# Check if Elasticsearch is running
if ! curl -s http://localhost:9201 > /dev/null; then
    echo " Error: Elasticsearch is not running on localhost:9201"
    echo "Start it with: docker compose up -d"
    exit 1
fi

echo " Elasticsearch is running"
echo ""

# Get workspace directory
WORKSPACE_DIR="$(pwd)"

# For tag-only search, no image needed
if [ "$MODE" = "tags" ]; then
    if [ -z "$TAGS" ]; then
        echo " Error: --tags required for tag mode"
        exit 1
    fi
    
    echo "Running tag-based search in Docker..."
    docker run --rm \
        --network host \
        -v "$WORKSPACE_DIR":/workspace \
        -w /workspace \
        -e ES_HOST=localhost \
        -e ES_PORT=9201 \
        sam_hybrid_testing \
        python3 search_with_segments.py --mode tags --top-k "$TOP_K" --tags $TAGS
    exit $?
fi

# Image-based search modes require image
if [ -z "$IMAGE_PATH" ]; then
    echo " Error: image_path required for whole, segment, or hybrid modes"
    exit 1
fi

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo " Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Get absolute path and convert to relative path from workspace
IMAGE_FULL_PATH="$(cd "$(dirname "$IMAGE_PATH")" && pwd)/$(basename "$IMAGE_PATH")"

# Check if the image is inside the workspace
if [[ "$IMAGE_FULL_PATH" != "$WORKSPACE_DIR"* ]]; then
    echo " Error: Image must be inside the workspace directory"
    echo "Image path: $IMAGE_FULL_PATH"
    echo "Workspace: $WORKSPACE_DIR"
    exit 1
fi

# Calculate relative path from workspace
IMAGE_REL_PATH="${IMAGE_FULL_PATH#$WORKSPACE_DIR/}"

echo "Running search with mode: $MODE"
echo ""

# Build Docker command
DOCKER_CMD="docker run --rm \
    --network host \
    -v \"$WORKSPACE_DIR\":/workspace \
    -w /workspace \
    -e ES_HOST=localhost \
    -e ES_PORT=9201 \
    sam_hybrid_testing \
    python3 search_with_segments.py --image \"$IMAGE_REL_PATH\" --mode $MODE --top-k $TOP_K"

if [ -n "$TAGS" ]; then
    DOCKER_CMD="$DOCKER_CMD --tags $TAGS"
fi

# Execute
eval $DOCKER_CMD

exit $?
