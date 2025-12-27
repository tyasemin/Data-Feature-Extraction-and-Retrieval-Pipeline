# SAM Inference Testing

This directory contains scripts for testing SAM (Segment Anything Model) inference on random images from the dataset.

## Files

- `test_sam_inference.py` - Main testing script that runs SAM inference on randomly selected images
- `docker-compose-test.yml` - Docker Compose configuration for running tests
- `run_test_inference.sh` - Convenient bash script to build and run tests
- `Dockerfile` - Docker image with SAM, CLIP, and all dependencies
- `sam_config.py` - Configuration for SAM parameters

## Quick Start

### Option 1: Using the convenience script (Recommended)

```bash
cd SAM
./run_test_inference.sh
```

This will:
1. Build the Docker image
2. Run inference on 20 randomly selected images
3. Save visualizations to `./sam_inference/`
4. Display a summary of results

### Option 2: Using Docker Compose directly

```bash
cd SAM

# Build the image
docker build -t sam_clip:latest .

# Run the test
docker-compose -f docker-compose-test.yml up

# Clean up
docker-compose -f docker-compose-test.yml down
```

### Option 3: Manual Docker run

```bash
cd SAM

# Build the image
docker build -t sam_clip:latest .

# Create output directory
mkdir -p sam_inference

# Run inference
docker run --gpus all --rm \
  -v /home/yasemin/Desktop/personal/yüksek/homework/Data-Feature-Extraction-and-Retrieval-Pipeline:/workspace \
  sam_clip:latest \
  python test_sam_inference.py \
    --csv /workspace/label_cleaned.csv \
    --num-images 20 \
    --output-dir /workspace/SAM/sam_inference \
    --seed 42
```

## Script Options

The `test_sam_inference.py` script accepts the following arguments:

- `--csv`: Path to the CSV file (default: `/workspace/label_cleaned.csv`)
- `--num-images`: Number of images to test (default: 20)
- `--output-dir`: Output directory for results (default: `/workspace/SAM/sam_inference`)
- `--seed`: Random seed for reproducibility (default: 42)

### Custom number of images

```bash
# Test with 50 images
docker-compose -f docker-compose-test.yml run --rm sam_test \
  python test_sam_inference.py --num-images 50
```

### Different random seed

```bash
# Use a different random selection
docker-compose -f docker-compose-test.yml run --rm sam_test \
  python test_sam_inference.py --seed 123
```

## Output

The script generates visualization images showing:

1. **Left panel**: Original image
2. **Right panel**: Image with SAM segmentation masks overlaid
3. **Statistics**: Image name, resolution, number of segments, and area statistics

Output files are saved as: `{original_filename}_sam_inference.png`

Example output structure:
```
sam_inference/
├── tophane_sam_inference.png
├── unkapani-koprusunde-at-arabasi-1953_sam_inference.png
├── halic-unkapani-suleymaniye-1945_sam_inference.png
└── ...
```

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA 11.8+ support
- nvidia-docker2 or Docker with `--gpus` flag support

## SAM Configuration

Edit `sam_config.py` to adjust SAM parameters:

- `POINTS_PER_SIDE`: Number of sampling points (higher = more segments)
- `PRED_IOU_THRESH`: IoU threshold for predictions
- `STABILITY_SCORE_THRESH`: Stability score threshold
- `MIN_MASK_REGION_AREA`: Minimum area for valid segments

## Troubleshooting

### No GPU detected

If running without GPU:
```bash
# Remove --gpus flag or nvidia runtime
docker run --rm -v ... sam_clip:latest python test_sam_inference.py ...
```

Note: Inference will be much slower on CPU.

### Permission denied on script

```bash
chmod +x run_test_inference.sh
```

### Output directory issues

Make sure the SAM directory has write permissions:
```bash
chmod 755 /home/yasemin/Desktop/personal/yüksek/homework/Data-Feature-Extraction-and-Retrieval-Pipeline/SAM
```

## Notes

- The script randomly selects images from `label_cleaned.csv`
- Images that don't exist or can't be loaded are skipped
- Inference time depends on image resolution and GPU performance
- Each image typically takes 5-30 seconds depending on size
- Total runtime for 20 images: approximately 5-10 minutes
