#!/usr/bin/env python3
"""
Advanced image search using segmented features and tags
Supports multiple search modes:
1. Whole image similarity (baseline)
2. Segment-level similarity (find images with similar segments)
3. Tag-based filtering (filter by semantic tags)
4. Hybrid search (combine whole image + segment similarity)
5. Text-to-image search (search by tag names)
"""

import os
import sys
import json
import argparse
import torch
import clip
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from elasticsearch import Elasticsearch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Elasticsearch connection
ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = os.getenv('ES_PORT', '9201')
es = Elasticsearch([f'http://{ES_HOST}:{ES_PORT}'])
INDEX_NAME = 'foto_atlas'

# SAM configuration
SAM_CHECKPOINT = os.getenv('SAM_CHECKPOINT', '/app/sam_models/sam_vit_b_01ec64.pth')
SAM_MODEL_TYPE = 'vit_b'


def segment_query_image(image_path, max_segments=10):
    """Segment query image using SAM and extract features for each segment."""
    print(f"\nSegmenting query image with SAM...")
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
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
    
    # Load and resize image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for processing
    max_size = 800
    h, w = image_rgb.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
    
    # Generate masks
    print(f"  Generating segments...")
    masks = mask_generator.generate(image_rgb)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    print(f"   Generated {len(masks)} segments")
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Extract features for each segment
    segment_features = []
    num_segments = min(len(masks), max_segments)
    
    print(f"  Extracting CLIP features for top {num_segments} segments...")
    
    for i, mask in enumerate(masks[:num_segments]):
        segmentation = mask['segmentation']
        bbox = mask['bbox']
        
        # Extract segment image
        masked = image_rgb.copy()
        masked[~segmentation] = 255  # White background
        
        x, y, w, h = [int(v) for v in bbox]
        cropped = masked[y:y+h, x:x+w]
        
        # Extract CLIP features
        pil_image = Image.fromarray(cropped)
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        
        segment_features.append({
            'segment_id': i + 1,
            'features': features.cpu().numpy().flatten().tolist(),
            'area': int(mask['area']),
            'bbox': bbox
        })
    
    print(f"   Extracted features for {len(segment_features)} segments")
    
    # Clean up
    del sam
    del mask_generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return segment_features, clip_model, clip_preprocess


def extract_image_features(image_path):
    """Extract CLIP features from a single image."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        features = image_features.cpu().numpy().flatten().tolist()
        print(f" Extracted {len(features)}-dimensional whole-image features")
        return features
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None


def search_whole_image(query_features, top_k=10, tag_filter=None):
    """Traditional whole-image similarity search."""
    query = {
        "size": top_k,
        "_source": {
            "excludes": ["features", "segmented_features.segments.features"]
        },
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'features') + 1.0",
                    "params": {"query_vector": query_features}
                }
            }
        }
    }
    
    # Add tag filter if specified
    if tag_filter:
        query["query"]["script_score"]["query"] = {
            "nested": {
                "path": "segmented_tags.segment_tags",
                "query": {
                    "nested": {
                        "path": "segmented_tags.segment_tags.tags",
                        "query": {
                            "terms": {
                                "segmented_tags.segment_tags.tags.tag": tag_filter
                            }
                        }
                    }
                }
            }
        }
    
    response = es.search(index=INDEX_NAME, body=query)
    return parse_results(response, "whole_image")


def search_segment_level(query_segment_features, top_k=20, segment_top_k=5, tag_filter=None):
    """Search at segment level - find images with segments similar to query segments."""
    
    # First, get candidates with segment features
    query = {
        "size": top_k * 3,  # Get more candidates to process
        "_source": ["filename", "image_path", "baslik", "etiketler", "segmented_features", "segmented_tags"],
        "query": {
            "exists": {
                "field": "segmented_features"
            }
        }
    }
    
    # Add tag filter if specified
    if tag_filter:
        query["query"] = {
            "bool": {
                "must": [
                    {"exists": {"field": "segmented_features"}},
                    {
                        "nested": {
                            "path": "segmented_tags.segment_tags",
                            "query": {
                                "nested": {
                                    "path": "segmented_tags.segment_tags.tags",
                                    "query": {
                                        "terms": {
                                            "segmented_tags.segment_tags.tags.tag": tag_filter
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }
    
    response = es.search(index=INDEX_NAME, body=query)
    
    # Calculate segment-level similarity for each image
    results = []
    
    # Convert query segments to numpy arrays
    query_vectors = [np.array(seg['features']) for seg in query_segment_features]
    
    for hit in response['hits']['hits']:
        source = hit['_source']
        
        if 'segmented_features' not in source or 'segments' not in source['segmented_features']:
            continue
        
        segments = source['segmented_features']['segments']
        
        # For each query segment, find the best matching database segment
        best_matches = []
        
        for q_idx, query_vector in enumerate(query_vectors):
            segment_scores = []
            
            # Calculate similarity for each database segment
            for segment in segments:
                if 'features' in segment and segment['features']:
                    seg_vector = np.array(segment['features'])
                    # Cosine similarity
                    similarity = np.dot(query_vector, seg_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(seg_vector)
                    )
                    segment_scores.append({
                        'segment_id': segment['segment_id'],
                        'similarity': float(similarity),
                        'area': segment.get('area', 0),
                        'segment_image': segment.get('segment_image', '')
                    })
            
            # Get best match for this query segment
            if segment_scores:
                segment_scores.sort(key=lambda x: x['similarity'], reverse=True)
                best_matches.append(segment_scores[0])
        
        if best_matches:
            # Use average of best matches across all query segments
            avg_similarity = np.mean([m['similarity'] for m in best_matches])
            max_similarity = max([m['similarity'] for m in best_matches])
            
            # Get segment tags for top matching segment
            segment_tags = []
            if 'segmented_tags' in source and 'segment_tags' in source['segmented_tags']:
                for seg_tag in source['segmented_tags']['segment_tags']:
                    if seg_tag['segment_id'] == best_matches[0]['segment_id']:
                        segment_tags = seg_tag.get('tags', [])
                        break
            
            result = {
                'filename': source.get('filename', 'Unknown'),
                'image_path': source.get('image_path', 'Unknown'),
                'similarity_score': avg_similarity,
                'similarity_percentage': ((avg_similarity + 1) / 2) * 100,
                'max_segment_similarity': max_similarity,
                'num_query_segments': len(query_vectors),
                'num_segments': len(segments),
                'top_segments': best_matches[:3],
                'best_matching_segment': best_matches[0],
                'segment_tags': segment_tags,
                'baslik': source.get('baslik'),
                'etiketler': source.get('etiketler'),
                'id': hit['_id']
            }
            results.append(result)
    
    # Sort by max similarity
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]


def search_by_tags(tags, top_k=10, min_confidence=0.1):
    """Search images by semantic tags."""
    query = {
        "size": top_k,
        "_source": {
            "excludes": ["features", "segmented_features.segments.features"]
        },
        "query": {
            "nested": {
                "path": "segmented_tags.segment_tags",
                "query": {
                    "nested": {
                        "path": "segmented_tags.segment_tags.tags",
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "terms": {
                                            "segmented_tags.segment_tags.tags.tag": tags
                                        }
                                    },
                                    {
                                        "range": {
                                            "segmented_tags.segment_tags.tags.confidence": {
                                                "gte": min_confidence
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
    
    response = es.search(index=INDEX_NAME, body=query)
    return parse_results(response, "tag_based")


def search_hybrid(query_features, query_segments, whole_weight=0.4, segment_weight=0.6, top_k=10, tag_filter=None):
    """Hybrid search: first get candidates by whole-image, then re-rank with segments."""
    
    # Step 1: Get candidates using whole-image similarity
    candidate_pool = max(top_k * 20, 100)
    print(f"\nStep 1: Getting {candidate_pool} candidates using whole-image search...")
    
    whole_results = search_whole_image(query_features, top_k=candidate_pool, tag_filter=tag_filter)
    print(f"   Found {len(whole_results)} candidates")
    
    if not whole_results:
        return []
    
    # Step 2: For each candidate, compute segment-level similarity
    print(f"\nStep 2: Re-ranking with segment-level matching...")
    
    # Get candidate documents with their segment features
    candidate_ids = [r['id'] for r in whole_results]
    
    # Build query to fetch segment features for candidates
    query = {
        "size": len(candidate_ids),
        "_source": ["filename", "image_path", "baslik", "etiketler", "segmented_features", "segmented_tags"],
        "query": {
            "bool": {
                "must": [
                    {"ids": {"values": candidate_ids}},
                    {"exists": {"field": "segmented_features"}}
                ]
            }
        }
    }
    
    response = es.search(index=INDEX_NAME, body=query)
    
    # Convert query segments to numpy arrays
    query_vectors = [np.array(seg['features']) for seg in query_segments]
    
    # Compute segment similarity for each candidate
    segment_scores_map = {}
    
    for hit in response['hits']['hits']:
        doc_id = hit['_id']
        source = hit['_source']
        
        if 'segmented_features' not in source or 'segments' not in source['segmented_features']:
            continue
        
        segments = source['segmented_features']['segments']
        
        # For each query segment, find best matching database segment
        best_matches = []
        
        for query_vector in query_vectors:
            segment_scores = []
            
            for segment in segments:
                if 'features' in segment and segment['features']:
                    seg_vector = np.array(segment['features'])
                    similarity = np.dot(query_vector, seg_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(seg_vector)
                    )
                    segment_scores.append(float(similarity))
            
            if segment_scores:
                best_matches.append(max(segment_scores))
        
        if best_matches:
            # Average of best matches across all query segments
            avg_similarity = np.mean(best_matches)
            segment_scores_map[doc_id] = avg_similarity
    
    print(f"   Computed segment scores for {len(segment_scores_map)} candidates")
    
    # Step 3: Combine whole-image and segment scores
    print(f"\nStep 3: Combining scores (whole={whole_weight}, segment={segment_weight})...")
    
    hybrid_results = []
    
    for result in whole_results:
        doc_id = result['id']
        whole_score = result['similarity_score']
        segment_score = segment_scores_map.get(doc_id, 0.0)
        
        # Weighted combination
        hybrid_score = (whole_weight * whole_score) + (segment_weight * segment_score)
        
        result['hybrid_score'] = hybrid_score
        result['whole_image_score'] = whole_score
        result['segment_score'] = segment_score
        
        hybrid_results.append(result)
    
    # Sort by hybrid score
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Show statistics
    with_segments = sum(1 for r in hybrid_results if r['segment_score'] > 0)
    print(f"   Results with segment scores: {with_segments}/{len(hybrid_results)}")
    
    return hybrid_results[:top_k]


def parse_results(response, search_type):
    """Parse Elasticsearch response into results."""
    results = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        
        if search_type == "tag_based":
            score = hit['_score']
            percentage = None
        else:
            score = hit['_score']
            percentage = (score / 2.0) * 100
        
        result = {
            'filename': source.get('filename', 'Unknown'),
            'image_path': source.get('image_path', 'Unknown'),
            'similarity_score': score,
            'similarity_percentage': percentage,
            'baslik': source.get('baslik'),
            'etiketler': source.get('etiketler'),
            'id': hit['_id']
        }
        results.append(result)
    
    return results


def visualize_results(query_image_path, results, output_name, workspace_root="/workspace"):
    """
    Create a visualization with query image on the left and found results below.
    
    Args:
        query_image_path: Path to the query image
        results: List of result dictionaries
        output_name: Output filename (e.g., 'sam_hybrid_dikilitaş_test.png')
        workspace_root: Root directory to resolve relative paths
    """
    print(f"\n{'='*80}")
    print("Creating visualization...")
    print(f"{'='*80}")
    
    # Limit to 10 results
    results = results[:10]
    num_results = len(results)
    
    if num_results == 0:
        print("No results to visualize")
        return
    
    # Load query image
    try:
        query_img = Image.open(query_image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading query image: {e}")
        return
    
    # Calculate layout: query on top, results in grid below
    # Grid: 5 columns x 2 rows for up to 10 results
    cols = 5
    rows = (num_results + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 + rows * 4))
    
    # Main grid: 1 row for query, remaining rows for results
    gs = fig.add_gridspec(1 + rows, 1, height_ratios=[3] + [4] * rows, hspace=0.3)
    
    # Display query image
    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_title(f"Query Image: {Path(query_image_path).name}", 
                       fontsize=16, fontweight='bold', pad=20)
    ax_query.axis('off')
    
    # Display results in grid
    for idx, result in enumerate(results):
        row = idx // cols
        col = idx % cols
        
        # Create subplot for this result
        gs_row = fig.add_gridspec(1 + rows, cols, 
                                  height_ratios=[3] + [4] * rows,
                                  hspace=0.3, wspace=0.2)
        ax = fig.add_subplot(gs_row[1 + row, col])
        
        # Load result image
        image_path = result['image_path']
        if image_path.startswith('/dataset'):
            full_path = workspace_root + image_path
        else:
            full_path = image_path
        
        try:
            result_img = Image.open(full_path).convert('RGB')
            ax.imshow(result_img)
            
            # Build title with rank and scores
            title_parts = [f"#{idx + 1}: {Path(result['filename']).stem[:30]}"]
            
            if 'hybrid_score' in result:
                title_parts.append(f"Hybrid: {result['hybrid_score']:.3f}")
                title_parts.append(f"(W:{result['whole_image_score']:.2f}, S:{result['segment_score']:.2f})")
            elif result.get('similarity_percentage'):
                title_parts.append(f"{result['similarity_percentage']:.1f}%")
            
            # Add başlık if available
            if result.get('baslik'):
                title_parts.append(f"\n{result['baslik'][:50]}")
            
            ax.set_title("\n".join(title_parts), fontsize=10, pad=10)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image:\n{Path(image_path).name}", 
                   ha='center', va='center', fontsize=8)
            ax.set_title(f"#{idx + 1}: Error", fontsize=10, color='red')
        
        ax.axis('off')
    
    # Hide empty subplots if results < 10
    for idx in range(num_results, cols * rows):
        row = idx // cols
        col = idx % cols
        gs_row = fig.add_gridspec(1 + rows, cols,
                                  height_ratios=[3] + [4] * rows,
                                  hspace=0.3, wspace=0.2)
        ax = fig.add_subplot(gs_row[1 + row, col])
        ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {output_name}")


def display_results(results, search_mode):
    """Display search results."""
    if not results:
        print("No similar images found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Top {len(results)} Results - Search Mode: {search_mode.upper()}")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Rank #{i}")
        print(f"{'─'*80}")
        
        print(f" Filename: {result['filename']}")
        print(f" Path: {result['image_path']}")
        
        if result.get('similarity_percentage') is not None:
            print(f" Similarity: {result['similarity_percentage']:.2f}%")
        
        if 'hybrid_score' in result:
            print(f"  Hybrid Score: {result['hybrid_score']:.4f}")
            print(f"  - Whole Image: {result['whole_image_score']:.4f}")
            print(f"  - Segment: {result['segment_score']:.4f}")
        
        if 'best_matching_segment' in result:
            seg = result['best_matching_segment']
            print(f" Best Matching Segment: #{seg['segment_id']} (similarity: {seg['similarity']:.4f})")
            
            if result.get('segment_tags'):
                top_tags = sorted(result['segment_tags'], key=lambda x: x['confidence'], reverse=True)[:3]
                tags_str = ", ".join([f"{t['tag']} ({t['confidence']:.2f})" for t in top_tags])
                print(f"  Segment Tags: {tags_str}")
        
        if result.get('baslik'):
            print(f" Başlık: {result['baslik']}")
        
        if result.get('etiketler'):
            print(f"  Etiketler: {result['etiketler']}")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced image search with segmented features and tags'
    )
    parser.add_argument('--image', type=str, help='Query image path')
    parser.add_argument('--mode', type=str, 
                       choices=['whole', 'segment', 'hybrid', 'tags'],
                       default='hybrid',
                       help='Search mode (default: hybrid)')
    parser.add_argument('--tags', type=str, nargs='+',
                       help='Tags to search for (for tag mode) or filter by')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--whole-weight', type=float, default=0.4,
                       help='Weight for whole image similarity in hybrid mode')
    parser.add_argument('--segment-weight', type=float, default=0.6,
                       help='Weight for segment similarity in hybrid mode')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADVANCED IMAGE SEARCH WITH SEGMENTS")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Elasticsearch: {ES_HOST}:{ES_PORT}")
    print(f"Index: {INDEX_NAME}")
    print("="*80)
    
    # Tag-only search
    if args.mode == 'tags':
        if not args.tags:
            print("Error: --tags required for tag mode")
            return 1
        
        print(f"\nSearching by tags: {', '.join(args.tags)}")
        results = search_by_tags(args.tags, top_k=args.top_k)
        display_results(results, args.mode)
        return 0
    
    # Image-based search modes
    if not args.image:
        print("Error: --image required for whole, segment, or hybrid modes")
        return 1
    
    print(f"\nQuery image: {args.image}")
    
    # For segment or hybrid mode, extract segments
    query_segments = None
    if args.mode in ['segment', 'hybrid']:
        query_segments, _, _ = segment_query_image(args.image, max_segments=10)
    
    # Extract whole image features for whole and hybrid modes
    query_features = None
    if args.mode in ['whole', 'hybrid']:
        query_features = extract_image_features(args.image)
        if not query_features:
            return 1
    
    # Perform search based on mode
    tag_filter = args.tags if args.tags else None
    
    if args.mode == 'whole':
        results = search_whole_image(query_features, top_k=args.top_k, tag_filter=tag_filter)
    elif args.mode == 'segment':
        if not query_segments:
            print("Error: Failed to segment query image")
            return 1
        results = search_segment_level(query_segments, top_k=args.top_k, tag_filter=tag_filter)
    elif args.mode == 'hybrid':
        if not query_segments:
            print("Error: Failed to segment query image")
            return 1
        results = search_hybrid(
            query_features,
            query_segments,
            whole_weight=args.whole_weight,
            segment_weight=args.segment_weight,
            top_k=args.top_k,
            tag_filter=tag_filter
        )
    
    display_results(results, args.mode)
    
    # Create visualization
    query_name = Path(args.image).stem
    output_name = f"sam_hybrid_{query_name}.png"
    visualize_results(args.image, results, output_name, workspace_root="/workspace")
    
    print(f"\n✓ Search complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
