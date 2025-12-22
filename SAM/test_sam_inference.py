#!/usr/bin/env python3
"""
Test SAM Inference Script
Randomly selects 20 images from label_cleaned.csv, runs SAM inference,
and outputs visualizations with segmentation masks
"""

import os
import sys
import csv
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sam_config as config


def load_sam_model():
    """Load SAM model"""
    print("Loading SAM model...")
    
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
    
    print(f"✓ SAM model loaded: {config.SAM_MODEL_TYPE}")
    return mask_generator, device


def read_csv_and_select_images(csv_path, num_images=20, random_seed=42):
    """Read CSV file and randomly select images"""
    print(f"\nReading CSV file: {csv_path}")
    
    # Get the base directory (CSV file's parent directory)
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    
    image_paths = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        for row in reader:
            # Get the first column which contains image path
            # Access the first value from the row (handles any column name variations)
            first_column = list(row.keys())[0]
            image_path = row[first_column]
            
            if image_path:
                # Handle both absolute and relative paths
                if not os.path.isabs(image_path):
                    image_path = os.path.join(csv_dir, image_path)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
    
    print(f"Found {len(image_paths)} valid images in CSV")
    
    # Randomly select images
    random.seed(random_seed)
    selected = random.sample(image_paths, min(num_images, len(image_paths)))
    
    print(f"Selected {len(selected)} images for testing")
    return selected


def show_anns(anns, ax):
    """Visualize SAM annotations"""
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)


def run_inference(image_path, mask_generator, output_dir):
    """Run SAM inference on a single image and save visualization"""
    print(f"\nProcessing: {Path(image_path).name}")
    
    try:
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Failed to load image: {image_path}")
            return False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if image is too large
        max_size = 1024
        h, w = image_rgb.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"  Resized from {w}x{h} to {new_w}x{new_h}")
        
        # Generate masks
        print("  Running SAM inference...")
        masks = mask_generator.generate(image_rgb)
        print(f"  Generated {len(masks)} segments")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # Image with masks
        axes[1].imshow(image_rgb)
        show_anns(masks, axes[1])
        axes[1].set_title(f'SAM Segmentation ({len(masks)} segments)', 
                         fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # Add statistics text
        stats_text = f"Image: {Path(image_path).name}\n"
        stats_text += f"Resolution: {image_rgb.shape[1]}x{image_rgb.shape[0]}\n"
        stats_text += f"Segments: {len(masks)}\n"
        
        if masks:
            areas = [m['area'] for m in masks]
            stats_text += f"Avg area: {np.mean(areas):.0f} px²\n"
            stats_text += f"Min area: {np.min(areas):.0f} px²\n"
            stats_text += f"Max area: {np.max(areas):.0f} px²"
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 family='monospace')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save output
        output_filename = Path(image_path).stem + '_sam_inference.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  ✓ Saved to: {output_filename}")
        return True
        
    except torch.cuda.OutOfMemoryError:
        print(f"  ✗ GPU out of memory - skipping this image")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        plt.close('all')
        return False
    except Exception as e:
        print(f"  ✗ Error processing image: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        plt.close('all')
        return False


def main():
    parser = argparse.ArgumentParser(description='Test SAM inference on random images')
    parser.add_argument('--csv', type=str, 
                       default='/workspace/label_cleaned.csv',
                       help='Path to label_cleaned.csv')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of images to test (default: 20)')
    parser.add_argument('--output-dir', type=str, 
                       default='/workspace/SAM/sam_inference',
                       help='Output directory for inference results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load SAM model
    mask_generator, device = load_sam_model()
    
    # Select random images from CSV
    selected_images = read_csv_and_select_images(
        args.csv, 
        num_images=args.num_images,
        random_seed=args.seed
    )
    
    if not selected_images:
        print("\n✗ No valid images found!")
        return 1
    
    # Run inference on selected images
    print(f"\n{'='*60}")
    print("Starting SAM inference on selected images...")
    print(f"{'='*60}")
    
    success_count = 0
    for i, image_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{len(selected_images)}]", end=" ")
        if run_inference(image_path, mask_generator, args.output_dir):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(selected_images)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(selected_images) - success_count}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
