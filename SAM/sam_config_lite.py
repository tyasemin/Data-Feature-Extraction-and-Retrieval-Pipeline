"""
SAM Configuration - Lightweight version for limited GPU memory
"""

# SAM Model Configuration (using smaller vit_b model)
SAM_CHECKPOINT = "/app/checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"  # Using vit_b instead of vit_h for less memory

# Segmentation Configuration (reduced for memory efficiency)
POINTS_PER_SIDE = 16  # Reduced from 32
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.95
MIN_MASK_REGION_AREA = 500  # Increased to get fewer but better segments

# CLIP Configuration
CLIP_MODEL = "ViT-B/32"
CLIP_DEVICE = "cuda"

# Processing Configuration
MAX_SEGMENTS_PER_IMAGE = 10
MIN_SEGMENT_SIZE = 0.01
EXTRACT_GLOBAL_FEATURES = True

# Output Configuration
FEATURE_DIMENSION = 512
SAVE_VISUALIZATION = False
SAVE_SEGMENT_IMAGES = True
