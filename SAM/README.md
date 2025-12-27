# SAM Segment-Based Feature Extraction & Similarity Search

Complete pipeline for extracting SAM segments, generating CLIP features, and performing similarity search with optional tag filtering.

## 📁 Files Overview

### Core Pipeline
- **`sam_segment_clip_pipeline.py`** - Main pipeline: SAM segmentation → CLIP features → Tag generation
- **`sam_config.py`** - Configuration for full SAM model (vit_h)
- **`sam_config_lite.py`** - Configuration for lightweight SAM model (vit_b)

### ElasticSearch Integration
- **`setup_elasticsearch_sam.py`** - Create `foto_atlas_sam` index
- **`upload_segments_to_elasticsearch.py`** - Upload segment features to ElasticSearch
- **`test_similarity_search.py`** - Test similarity search with optional tag filtering

### Testing
- **`test_sam_inference.py`** - Test SAM inference with visualization

### Docker
- **`Dockerfile`** - Docker image with SAM + CLIP
- **`docker-compose-pipeline.yml`** - Run full pipeline
- **`docker-compose-test.yml`** - Run inference tests

## 🏷️ Tag Categories (50 tags)

**Architecture (20 tags):**
mosque, church, tower, minaret, dome, palace, castle, bridge, gate, arch, column, fortress, monument, building, wall, roof, window, balcony, courtyard, fountain

**Nature (10 tags):**
water, sea, sky, mountain, tree, cloud, river, hill, garden, vegetation

**Objects (20 tags):**
people, person, crowd, boat, ship, vehicle, flag, statue, sculpture, decoration, ornament, painting, sign, text, street, square, market, harbor, panorama, cityscape

## 🚀 Usage

### Step 1: Run SAM + CLIP Pipeline

```bash
# Process 20 random images with 5 segments each
cd SAM
docker compose -f docker-compose-pipeline.yml run --rm sam_segment_clip \
  python sam_segment_clip_pipeline.py \
  --num-images 20 \
  --max-segments 5 \
  --seed 42
```

**Outputs:**
- `segmented_features/` - CLIP features (512-dim) for each segment
- `segmented_tags/` - Tags with confidence scores
- `segmented_images/` - Segment images as PNG files

### Step 2: Setup ElasticSearch Index

```bash
python setup_elasticsearch_sam.py
```

Creates `foto_atlas_sam` index with:
- Segment-level CLIP features (512-dim dense vector)
- Whole-image CLIP features (512-dim dense vector)
- Tags with categories and confidence scores
- Multi-level search support

### Step 3: Upload Segments to ElasticSearch

```bash
python upload_segments_to_elasticsearch.py \
  --features-dir ./segmented_features \
  --batch-size 100
```

### Step 4: Test Similarity Search

**Without tag filtering (pure similarity):**
```bash
python test_similarity_search.py \
  --query-image /path/to/query/image.jpg \
  --top-k 10 \
  --search-type segment
```

**With tag filtering (filtered similarity):**
```bash
# Find images with mosque AND minaret
python test_similarity_search.py \
  --query-image /path/to/query/image.jpg \
  --tags mosque minaret \
  --top-k 10 \
  --search-type segment
```

**Search types:**
- `segment` - Search at segment level (default)
- `whole` - Search using whole-image features
- `both` - Combine segment and whole-image search

**Save results to file:**
```bash
python test_similarity_search.py \
  --query-image /path/to/query/image.jpg \
  --tags dome architecture \
  --top-k 20 \
  --output search_results.json
```

## 📊 Index Structure

```json
{
  "original_image": "/workspace/dataset/2024/04/image.jpg",
  "filename": "image",
  "segment_id": 1,
  "segment_image": "image_1.png",
  "segment_area": 12345,
  "segment_bbox": [x, y, width, height],
  "stability_score": 0.95,
  
  "clip_features": [512-dim vector],
  "whole_image_features": [512-dim vector],
  
  "tags": [
    {"tag": "mosque", "confidence": 0.85, "category": "architecture"},
    {"tag": "dome", "confidence": 0.72, "category": "architecture"}
  ],
  "tag_list": ["mosque", "dome"],
  
  "has_architecture": true,
  "has_nature": false,
  "has_objects": false
}
```

## 🔍 Search Examples

### 1. Find similar segments (no filter)
```bash
python test_similarity_search.py \
  --query-image dataset/2024/04/hagia_sophia.jpg \
  --top-k 10
```

### 2. Find segments with architecture
```bash
python test_similarity_search.py \
  --query-image dataset/2024/04/hagia_sophia.jpg \
  --tags dome minaret \
  --top-k 10
```

### 3. Find segments with water/nature
```bash
python test_similarity_search.py \
  --query-image dataset/2024/04/bosphorus.jpg \
  --tags water sea \
  --top-k 10
```

### 4. Whole-image similarity
```bash
python test_similarity_search.py \
  --query-image dataset/2024/04/panorama.jpg \
  --search-type whole \
  --top-k 10
```

## 🎯 Benefits of Segment-Based Search

1. **Region-level matching** - Find similar objects/regions, not just whole images
2. **Multi-object handling** - Images with multiple objects can match different queries
3. **Better precision** - Match specific architectural elements (domes, minarets, columns)
4. **Compositional queries** - Combine tags: "mosque AND water" finds mosques near water
5. **Multi-level search** - Search at both segment and whole-image levels

## 📝 Notes

- Uses SAM vit_b model (lightweight, ~6GB GPU)
- CLIP ViT-B/32 for features
- Sequential GPU loading: SAM → unload → CLIP
- Segments resized to 800px before CLIP encoding
- Confidence scores from zero-shot classification
- ElasticSearch cosine similarity for vector search

## 🔧 Configuration

Edit `sam_segment_clip_pipeline.py` to modify:
- Number of segments per image (`--max-segments`)
- Tag categories (currently 50 curated tags)
- Image resize dimensions (currently 800px)
- Random seed for reproducibility

Edit `setup_elasticsearch_sam.py` to modify:
- Index name (currently `foto_atlas_sam`)
- Number of shards/replicas
- Vector dimensions (currently 512 for CLIP ViT-B/32)

## 📦 Docker Commands

**Build image:**
```bash
docker compose -f docker-compose-pipeline.yml build
```

**Run pipeline:**
```bash
docker compose -f docker-compose-pipeline.yml run --rm sam_segment_clip \
  python sam_segment_clip_pipeline.py --num-images 20 --max-segments 5
```

**Run test inference:**
```bash
docker compose -f docker-compose-test.yml run --rm sam_test \
  python test_sam_inference.py --num-images 10
```
