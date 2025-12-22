"""
SAM Configuration
"""

# SAM Model Configuration
SAM_CHECKPOINT = "/app/checkpoints/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"  # Options: vit_h, vit_l, vit_b

# Segmentation Configuration
POINTS_PER_SIDE = 32  # Grid points for automatic mask generation
PRED_IOU_THRESH = 0.88  # Quality threshold for masks
STABILITY_SCORE_THRESH = 0.95  # Stability threshold
MIN_MASK_REGION_AREA = 100  # Minimum area for a segment (pixels)

# CLIP Configuration
CLIP_MODEL = "ViT-B/32"
CLIP_DEVICE = "cuda"  # or "cpu"

# Processing Configuration
MAX_SEGMENTS_PER_IMAGE = 10  # Maximum number of segments to extract features from
MIN_SEGMENT_SIZE = 0.01  # Minimum segment size as fraction of image (1% of image)
EXTRACT_GLOBAL_FEATURES = True  # Also extract features from the full image

# Output Configuration
FEATURE_DIMENSION = 512  # CLIP ViT-B/32 feature dimension
SAVE_VISUALIZATION = False  # Save segmentation masks visualization
SAVE_SEGMENT_IMAGES = False  # Save individual segment images
