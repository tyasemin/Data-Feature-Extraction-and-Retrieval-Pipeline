# SAM + CLIP Feature Extraction

This directory contains scripts for extracting features using SAM (Segment Anything Model) + CLIP.

## Overview

SAM segments images into regions of interest, and CLIP extracts features from each segment. This provides:
- **Better matching**: Features from specific objects (e.g., towers, buildings)
- **Robustness**: Less affected by background variations
- **Multi-scale features**: Both segment-level and global image features

## Quick Start

### 1. Build Docker Image
```bash
cd SAM
docker build -t sam-clip-extractor .
```

### 2. Run Feature Extraction
```bash
# From the project root directory
./SAM/run_sam_extraction.sh dataset features_sam
```

Or with custom paths:
```bash
./SAM/run_sam_extraction.sh /path/to/images /path/to/output
```

## Configuration

Edit `sam_config.py` to customize:

```python
# SAM Model (vit_h = highest quality, vit_b = fastest)
SAM_MODEL_TYPE = "vit_h"

# Segmentation quality
POINTS_PER_SIDE = 32  # More points = better segmentation, slower
PRED_IOU_THRESH = 0.88  # Quality threshold (0-1)

# Output settings
MAX_SEGMENTS_PER_IMAGE = 10  # Max segments to extract features from
MIN_SEGMENT_SIZE = 0.01  # Minimum segment size (1% of image)
EXTRACT_GLOBAL_FEATURES = True  # Also extract full image features
```

## Output Format

Each image produces a JSON file with:

```json
{
  "filename": "image.jpg",
  "image_path": "/dataset/image.jpg",
  "num_segments": 5,
  "segments": [
    {
      "segment_id": 0,
      "area": 12500,
      "bbox": [100, 200, 150, 200],
      "stability_score": 0.95,
      "features": [0.123, -0.456, ...]  // 512-dim CLIP features
    }
  ],
  "global_features": [0.789, -0.234, ...],  // Full image features
  "feature_dimension": 512
}
```

## SAM Models

Three model sizes available:

| Model | Checkpoint Size | Quality | Speed |
|-------|----------------|---------|-------|
| vit_h | 2.4 GB | Best | Slow |
| vit_l | 1.2 GB | Good | Medium |
| vit_b | 375 MB | Fast | Fast |

Default: `vit_h` (best quality)

## GPU Requirements

- Recommended: NVIDIA GPU with 8GB+ VRAM
- Falls back to CPU if GPU unavailable (much slower)

## How It Works

1. **SAM Segmentation**: Divides image into distinct objects/regions
2. **Segment Filtering**: Keeps top segments by size and quality
3. **CLIP Feature Extraction**: Extracts features from each segment
4. **Global Features**: Also extracts features from full image
5. **JSON Output**: Saves all features with metadata

## Integration with Search

To use SAM features in similarity search:

1. Upload SAM features to Elasticsearch (segment-level index)
2. Modify search to query both global and segment features
3. Aggregate results (e.g., best matching segment + global similarity)

## Troubleshooting

### Out of Memory
- Reduce `POINTS_PER_SIDE` in config
- Use smaller model (vit_b instead of vit_h)
- Process fewer segments: lower `MAX_SEGMENTS_PER_IMAGE`

### Slow Processing
- Use vit_b model (fastest)
- Reduce `POINTS_PER_SIDE` to 16
- Enable GPU support

### Poor Segmentation
- Increase `POINTS_PER_SIDE` to 64
- Lower quality thresholds slightly
- Use vit_h model (best quality)
