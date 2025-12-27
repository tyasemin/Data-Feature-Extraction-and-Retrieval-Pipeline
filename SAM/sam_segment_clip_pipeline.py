#!/usr/bin/env python3
"""
SAM Segment + CLIP Feature Extraction + Tag Generation
Processes images: SAM segmentation → save segments → CLIP features → generate tags
Uses actual tags from label_cleaned.csv dataset
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
import clip
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Use lightweight config for limited GPU memory
try:
    import sam_config_lite as config
    print("Using lightweight SAM configuration")
except ImportError:
    import sam_config as config


def get_pretrained_tags():
    """Get curated tags grouped by category for CLIP's zero-shot classification"""
    print(f"\n{'='*80}")
    print("USING CLIP PRE-TRAINED CATEGORIES (GROUPED)")
    print(f"{'='*80}")
    
    # Architecture (20 tags) - Buildings and architectural elements
    architecture_tags = [
        "mosque", "church", "tower", "minaret", "dome", "palace", "castle", "bridge",
        "gate", "arch", "column", "fortress", "monument", "building", "wall",
        "roof", "window", "balcony", "courtyard", "fountain"
    ]
    
    # Nature (10 tags) - Natural elements and landscape
    nature_tags = [
        "water", "sea", "sky", "mountain", "tree", "cloud", "river", "hill",
        "garden", "vegetation"
    ]
    
    # Objects (20 tags) - People, vehicles, and other objects
    object_tags = [
        "people", "person", "crowd", "boat", "ship", "vehicle", "flag", "statue",
        "sculpture", "decoration", "ornament", "painting", "sign", "text",
        "street", "square", "market", "harbor", "panorama", "cityscape"
    ]
    
    # Combine all tags
    all_tags = architecture_tags + nature_tags + object_tags
    
    print(f" Architecture tags: {len(architecture_tags)}")
    print(f" Nature tags: {len(nature_tags)}")
    print(f" Object tags: {len(object_tags)}")
    print(f" Total: {len(all_tags)} curated categories")
    
    return all_tags


def load_sam_model():
    """Load only SAM model on GPU"""
    print("Loading SAM model on GPU...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM device: {device}")
    
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT)
    sam.to(device=device)
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=800,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )
    
    print(f" SAM model loaded: {config.SAM_MODEL_TYPE} on {device}")
    return mask_generator, sam, device


def unload_sam_model(sam, mask_generator):
    """Unload SAM from GPU and clear memory"""
    print("  Unloading SAM from GPU...")
    del sam
    del mask_generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("   SAM unloaded, GPU memory cleared")


def load_clip_model():
    """Load CLIP model on GPU after SAM is unloaded"""
    print("\nLoading CLIP model on GPU...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CLIP device: {device}")
    
    clip_model, clip_preprocess = clip.load(config.CLIP_MODEL, device=device)
    clip_model.eval()
    
    print(f" CLIP model loaded: {config.CLIP_MODEL} on {device}")
    return clip_model, clip_preprocess, device


def read_csv_and_select_images(csv_path, num_images=20, random_seed=42, row_range=None):
    """Read CSV file and select images (by row range or random selection)
    
    Args:
        csv_path: Path to CSV file
        num_images: Number of images to randomly select (ignored if row_range is set)
        random_seed: Random seed for selection (ignored if row_range is set)
        row_range: Tuple of (start, end) row indices (1-based, inclusive). If set, overrides random selection.
    """
    print(f"\nReading CSV file: {csv_path}")
    
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    
    image_paths = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row[list(row.keys())[0]]
            full_path = os.path.join(csv_dir, image_path)
            if os.path.exists(full_path):
                image_paths.append(full_path)
    
    print(f"Found {len(image_paths)} valid images in CSV")
    
    # Select images based on row_range or random selection
    if row_range:
        start_idx, end_idx = row_range
        # Convert from 1-based to 0-based indexing
        start_idx = max(0, start_idx - 1)
        end_idx = min(len(image_paths), end_idx)
        selected = image_paths[start_idx:end_idx]
        print(f"Selected rows {start_idx + 1}-{end_idx} ({len(selected)} images)")
    else:
        random.seed(random_seed)
        selected = random.sample(image_paths, min(num_images, len(image_paths)))
        print(f"Randomly selected {len(selected)} images (seed={random_seed})")
    
    return selected


def segment_image_with_sam(image_path, mask_generator):
    """Segment image using SAM on GPU"""
    # Clear GPU cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load and resize image if too large
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to moderate size for GPU processing
    max_size = 800
    h, w = image_rgb.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
        print(f"  Resized from {w}x{h} to {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    
    # Generate masks
    masks = mask_generator.generate(image_rgb)
    
    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    return image_rgb, masks


def extract_segment_image(image, mask_data):
    """Extract segment as a separate image"""
    segmentation = mask_data['segmentation']
    bbox = mask_data['bbox']  # [x, y, w, h]
    
    # Create masked image
    masked = image.copy()
    masked[~segmentation] = 255  # White background
    
    # Crop to bounding box
    x, y, w, h = [int(v) for v in bbox]
    cropped = masked[y:y+h, x:x+w]
    
    return cropped


def extract_clip_features(segment_image, clip_model, clip_preprocess, device):
    """Extract CLIP features from segment image on GPU"""
    try:
        pil_image = Image.fromarray(segment_image)
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"    Warning: Failed to extract features: {e}")
        return None


def generate_tags(segment_image, clip_model, clip_preprocess, device, tag_list, top_k=5):
    """Generate tags for segment using CLIP zero-shot classification on GPU"""
    try:
        pil_image = Image.fromarray(segment_image)
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
        
        text_prompts = [tag for tag in tag_list]
        batch_size = 100
        all_similarities = []
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for i in range(0, len(text_prompts), batch_size):
                batch_prompts = text_prompts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_prompts, truncate=True).to(device)
                
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T)
                all_similarities.append(similarity)
        
        all_similarities = torch.cat(all_similarities, dim=1)
        similarity_probs = all_similarities.softmax(dim=-1)
        values, indices = similarity_probs[0].topk(top_k)
        
        tags = []
        for value, index in zip(values, indices):
            tags.append({
                'tag': tag_list[index],
                'confidence': float(value)
            })
        
        return tags
    except Exception as e:
        print(f"    Warning: Failed to generate tags: {e}")
        return []


def save_segment_image(segment_image, output_path):
    """Save segment image to disk"""
    try:
        pil_image = Image.fromarray(segment_image)
        pil_image.save(output_path)
        return True
    except Exception as e:
        print(f"    Warning: Failed to save segment: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='SAM segmentation + CLIP feature extraction + tag generation'
    )
    parser.add_argument('--csv', type=str,
                       default='/workspace/label_cleaned.csv',
                       help='Path to label_cleaned.csv')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of images to randomly process (default: 20, ignored if --row-range is set)')
    parser.add_argument('--row-range', type=str, default=None,
                       help='Row range to process from CSV (e.g., "1-1000" or "2000-5000"). Overrides --num-images.')
    parser.add_argument('--max-segments', type=int, default=10,
                       help='Maximum segments per image (default: 10)')
    parser.add_argument('--features-dir', type=str,
                       default='/workspace/SAM/segmented_features',
                       help='Output directory for features')
    parser.add_argument('--tags-dir', type=str,
                       default='/workspace/SAM/segmented_tags',
                       help='Output directory for tags')
    parser.add_argument('--segments-dir', type=str,
                       default='/workspace/SAM/segmented_images',
                       help='Output directory for segment images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for random selection (default: 42, ignored if --row-range is set)')
    
    args = parser.parse_args()
    
    # Parse row range if provided
    row_range = None
    if args.row_range:
        try:
            start, end = args.row_range.split('-')
            row_range = (int(start), int(end))
            if row_range[0] < 1 or row_range[1] < row_range[0]:
                print(f" Invalid row range: {args.row_range}. Start must be >= 1 and end must be >= start.")
                return 1
        except ValueError:
            print(f" Invalid row range format: {args.row_range}. Use format: START-END (e.g., 1-1000)")
            return 1
    
    # Create output directories
    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.tags_dir, exist_ok=True)
    os.makedirs(args.segments_dir, exist_ok=True)
    
    print("="*80)
    print("SAM SEGMENTATION + CLIP FEATURES + TAG GENERATION")
    print("="*80)
    print(f"Features output: {args.features_dir}")
    print(f"Tags output: {args.tags_dir}")
    print(f"Segments output: {args.segments_dir}")
    print("="*80)
    
    # Get pretrained tags for CLIP classification
    tag_list = get_pretrained_tags()
    
    # Select images (by row range or random selection)
    if row_range:
        selected_images = read_csv_and_select_images(
            args.csv,
            row_range=row_range
        )
    else:
        selected_images = read_csv_and_select_images(
            args.csv,
            num_images=args.num_images,
            random_seed=args.seed
        )
    
    if not selected_images:
        print("\n✗ No valid images found!")
        return 1
    
    # Process images - load SAM, process all images, then load CLIP
    print(f"\n{'='*80}")
    print("PHASE 1: SAM SEGMENTATION ON GPU")
    print(f"{'='*80}")
    
    # Load SAM model
    mask_generator, sam_model, sam_device = load_sam_model()
    
    # Store segmentation metadata (not full images)
    segmentation_metadata = []
    
    for i, image_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{len(selected_images)}] Segmenting: {Path(image_path).stem}")
        try:
            image_rgb, masks = segment_image_with_sam(image_path, mask_generator)
            if image_rgb is not None and masks:
                filename = Path(image_path).stem
                num_segments = min(len(masks), args.max_segments)
                
                # Save segment images to disk
                segment_files = []
                for j in range(num_segments):
                    mask = masks[j]
                    segment_num = j + 1
                    segment_image = extract_segment_image(image_rgb, mask)
                    
                    segment_filename = f"{filename}_{segment_num}.png"
                    segment_path = os.path.join(args.segments_dir, segment_filename)
                    save_segment_image(segment_image, segment_path)
                    
                    segment_files.append({
                        'segment_id': segment_num,
                        'filename': segment_filename,
                        'path': segment_path,
                        'area': int(mask['area']),
                        'bbox': [float(x) for x in mask['bbox']],
                        'stability_score': float(mask['stability_score'])
                    })
                
                segmentation_metadata.append({
                    'image_path': image_path,
                    'filename': filename,
                    'image_resolution': list(image_rgb.shape[:2]),
                    'num_segments': num_segments,
                    'segments': segment_files
                })
                print(f"   Generated and saved {num_segments} segments")
            else:
                print(f"   Failed to segment")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Unload SAM and free GPU memory
    unload_sam_model(sam_model, mask_generator)
    
    # Load CLIP model
    print(f"\n{'='*80}")
    print("PHASE 2: CLIP FEATURE EXTRACTION ON GPU")
    print(f"{'='*80}")
    
    clip_model, clip_preprocess, clip_device = load_clip_model()
    
    # Process all segmentations with CLIP (read from disk)
    success_count = 0
    for i, seg_metadata in enumerate(segmentation_metadata, 1):
        image_path = seg_metadata['image_path']
        filename = seg_metadata['filename']
        num_segments = seg_metadata['num_segments']
        
        print(f"\n[{i}/{len(segmentation_metadata)}] Processing features for: {filename}")
        
        try:
            segment_data = []
            
            for seg_info in seg_metadata['segments']:
                segment_num = seg_info['segment_id']
                segment_path = seg_info['path']
                
                print(f"  Segment {segment_num}/{num_segments}...", end=" ")
                
                # Load segment image from disk
                segment_image = np.array(Image.open(segment_path))
                
                # Extract features and tags
                features = extract_clip_features(segment_image, clip_model, clip_preprocess, clip_device)
                tags = generate_tags(segment_image, clip_model, clip_preprocess, clip_device, tag_list, top_k=5)
                
                print("✓")
                
                segment_info = {
                    'segment_id': segment_num,
                    'area': seg_info['area'],
                    'bbox': seg_info['bbox'],
                    'stability_score': seg_info['stability_score'],
                    'segment_image': seg_info['filename'],
                    'features': features,
                    'tags': tags
                }
                segment_data.append(segment_info)
            
            # Save features
            features_filename = f"{filename}.json"
            features_path = os.path.join(args.features_dir, features_filename)
            features_output = {
                'original_image': str(image_path),
                'filename': filename,
                'image_resolution': seg_metadata['image_resolution'],
                'num_segments': num_segments,
                'segments': segment_data
            }
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(features_output, f, indent=2, ensure_ascii=False)
            
            # Save tags
            tags_filename = f"{filename}_tags.json"
            tags_path = os.path.join(args.tags_dir, tags_filename)
            tags_output = {
                'original_image': str(image_path),
                'filename': filename,
                'num_segments': num_segments,
                'segment_tags': [
                    {
                        'segment_id': seg['segment_id'],
                        'segment_image': seg['segment_image'],
                        'area': seg['area'],
                        'tags': seg['tags']
                    }
                    for seg in segment_data
                ]
            }
            with open(tags_path, 'w', encoding='utf-8') as f:
                json.dump(tags_output, f, indent=2, ensure_ascii=False)
            
            print(f"   Successfully processed {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {len(selected_images)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(selected_images) - success_count}")
    print(f"\nOutput directories:")
    print(f"  Features: {args.features_dir}")
    print(f"  Tags: {args.tags_dir}")
    print(f"  Segments: {args.segments_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
