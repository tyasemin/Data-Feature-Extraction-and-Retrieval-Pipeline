#!/usr/bin/env python3
"""
Extract SAM segments and CLIP features from images
Combines SAM segmentation with CLIP feature extraction for better similarity search
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import clip
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sam_config as config


def load_models():
    """Load SAM and CLIP models"""
    print("Loading models...")
    
    # Load SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.POINTS_PER_SIDE,
        pred_iou_thresh=config.PRED_IOU_THRESH,
        stability_score_thresh=config.STABILITY_SCORE_THRESH,
        min_mask_region_area=config.MIN_MASK_REGION_AREA,
    )
    
    # Load CLIP
    clip_model, clip_preprocess = clip.load(config.CLIP_MODEL, device=device)
    
    print(f"✓ SAM model loaded: {config.SAM_MODEL_TYPE}")
    print(f"✓ CLIP model loaded: {config.CLIP_MODEL}")
    
    return mask_generator, clip_model, clip_preprocess, device


def segment_image(image_path, mask_generator):
    """Segment image using SAM"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    return image, masks


def filter_segments(masks, image_shape):
    """Filter segments based on size and quality"""
    h, w = image_shape[:2]
    total_pixels = h * w
    min_pixels = int(total_pixels * config.MIN_SEGMENT_SIZE)
    
    filtered = []
    for mask in masks:
        if mask['area'] >= min_pixels:
            filtered.append(mask)
            if len(filtered) >= config.MAX_SEGMENTS_PER_IMAGE:
                break
    
    return filtered


def extract_segment_features(image, mask, clip_model, clip_preprocess, device):
    """Extract CLIP features from a segmented region"""
    # Apply mask to image
    segmentation = mask['segmentation']
    
    # Create masked image (keep only the segment)
    masked_image = image.copy()
    masked_image[~segmentation] = 0  # Set background to black
    
    # Find bounding box of the segment
    y_indices, x_indices = np.where(segmentation)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Crop to bounding box
    cropped = masked_image[y_min:y_max+1, x_min:x_max+1]
    
    # Convert to PIL and preprocess for CLIP
    pil_image = Image.fromarray(cropped)
    image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().flatten().tolist()


def extract_global_features(image_path, clip_model, clip_preprocess, device):
    """Extract CLIP features from the entire image"""
    image = Image.open(image_path).convert('RGB')
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = clip_model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().flatten().tolist()


def process_single_image(image_path, mask_generator, clip_model, clip_preprocess, device, output_folder):
    """Process a single image: segment with SAM and extract CLIP features"""
    
    filename = os.path.basename(image_path)
    print(f"\nProcessing: {filename}")
    
    try:
        # Segment image
        print("  Segmenting with SAM...")
        image, masks = segment_image(image_path, mask_generator)
        print(f"  Found {len(masks)} segments")
        
        # Filter segments
        filtered_masks = filter_segments(masks, image.shape)
        print(f"  Kept {len(filtered_masks)} segments after filtering")
        
        # Extract features for each segment
        segment_features = []
        for i, mask in enumerate(filtered_masks):
            features = extract_segment_features(image, mask, clip_model, clip_preprocess, device)
            if features is not None:
                segment_features.append({
                    'segment_id': i,
                    'area': int(mask['area']),
                    'bbox': mask['bbox'],  # [x, y, w, h]
                    'stability_score': float(mask['stability_score']),
                    'features': features
                })
        
        print(f"  Extracted features from {len(segment_features)} segments")
        
        # Extract global features
        global_features = None
        if config.EXTRACT_GLOBAL_FEATURES:
            print("  Extracting global image features...")
            global_features = extract_global_features(image_path, clip_model, clip_preprocess, device)
        
        # Prepare output
        output_data = {
            'filename': filename,
            'image_path': image_path,
            'num_segments': len(segment_features),
            'segments': segment_features,
            'global_features': global_features,
            'feature_dimension': config.FEATURE_DIMENSION
        }
        
        # Save to JSON
        output_filename = os.path.splitext(filename)[0] + '.json'
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {filename}: {e}")
        return False


def process_images_folder(images_folder, output_folder, allowed_filenames=None):
    """Find all images in folder recursively"""
    image_files = []
    for root, dirs, files in os.walk(images_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                if allowed_filenames is None or filename in allowed_filenames:
                    filepath = os.path.join(root, filename)
                    image_files.append(filepath)
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description='Extract SAM segments and CLIP features')
    parser.add_argument('images_folder', help='Folder containing images to process')
    parser.add_argument('output_folder', help='Folder to save feature JSON files')
    parser.add_argument('--csv-file', default=None, help='CSV file to filter images (optional)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SAM + CLIP Feature Extraction")
    print("="*80)
    print(f"Input folder: {args.images_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"CSV filter: {args.csv_file or 'None (process all images)'}")
    print("="*80)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load models
    mask_generator, clip_model, clip_preprocess, device = load_models()
    
    # Get list of images to process
    allowed_filenames = None
    if args.csv_file:
        print(f"\nReading image list from {args.csv_file}...")
        # TODO: Implement CSV reading if needed
        pass
    
    image_files = process_images_folder(args.images_folder, args.output_folder, allowed_filenames)
    
    if not image_files:
        print("\n✗ No images found!")
        return 1
    
    print(f"\n✓ Found {len(image_files)} images to process")
    print("="*80)
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]")
        if process_single_image(image_path, mask_generator, clip_model, clip_preprocess, device, args.output_folder):
            success_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(image_files) - success_count}")
    print(f"Output folder: {args.output_folder}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
