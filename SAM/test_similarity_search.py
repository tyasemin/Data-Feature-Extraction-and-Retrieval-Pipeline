#!/usr/bin/env python3
"""
Test similarity search using SAM segments
Supports optional tag filtering and multi-level search
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import clip
from elasticsearch import Elasticsearch

# ElasticSearch connection
ES_HOST = "http://localhost:9200"
INDEX_NAME = "foto_atlas_sam"


def load_clip_model():
    """Load CLIP model for encoding query image"""
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✓ CLIP loaded on {device}")
    return model, preprocess, device


def extract_image_features(image_path, model, preprocess, device):
    """Extract CLIP features from image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()[0]
        
        return features.tolist()
    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        return None


def search_similar_segments(es, query_features, tags=None, top_k=10, search_type="segment"):
    """
    Search for similar segments in ElasticSearch
    
    Args:
        es: ElasticSearch client
        query_features: CLIP features of query image (512-dim)
        tags: Optional list of tags to filter by
        top_k: Number of results to return
        search_type: "segment" (segment-level) or "whole" (whole-image level) or "both"
    """
    
    # Determine which feature field to use
    if search_type == "whole":
        feature_field = "whole_image_features"
    else:
        feature_field = "clip_features"
    
    # Build query
    query = {
        "script_score": {
            "query": {"match_all": {}}
        }
    }
    
    # Add tag filter if specified
    if tags:
        must_clauses = []
        for tag in tags:
            must_clauses.append({"term": {"tag_list": tag}})
        
        query["script_score"]["query"] = {
            "bool": {
                "must": must_clauses
            }
        }
    
    # Add similarity scoring
    query["script_score"]["script"] = {
        "source": f"cosineSimilarity(params.query_vector, '{feature_field}') + 1.0",
        "params": {"query_vector": query_features}
    }
    
    # Execute search
    response = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": query,
            "_source": [
                "original_image", "filename", "segment_id", "segment_image",
                "segment_area", "stability_score", "tags", "tag_list"
            ]
        }
    )
    
    return response['hits']['hits']


def print_results(results, query_image):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print("SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Query image: {query_image}")
    print(f"Found {len(results)} similar segments\n")
    
    for i, hit in enumerate(results, 1):
        source = hit['_source']
        score = hit['_score']
        
        print(f"[{i}] Score: {score:.4f}")
        print(f"    Original: {source['filename']}")
        print(f"    Segment: {source['segment_image']}")
        print(f"    Segment ID: {source['segment_id']}")
        print(f"    Area: {source['segment_area']}")
        
        # Print top tags
        if source.get('tags'):
            top_tags = source['tags'][:3]
            tag_str = ", ".join([f"{t['tag']}({t['confidence']:.2f})" for t in top_tags])
            print(f"    Top tags: {tag_str}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Test SAM segment similarity search with optional tag filtering'
    )
    parser.add_argument('--query-image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--tags', type=str, nargs='*',
                       help='Optional tags to filter results (e.g., mosque minaret dome)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return (default: 10)')
    parser.add_argument('--search-type', type=str, default='segment',
                       choices=['segment', 'whole', 'both'],
                       help='Search at segment-level, whole-image level, or both')
    parser.add_argument('--output', type=str,
                       help='Optional: Save results to JSON file')
    
    args = parser.parse_args()
    
    # Verify query image exists
    if not Path(args.query_image).exists():
        print(f"✗ Query image not found: {args.query_image}")
        return 1
    
    print(f"{'='*80}")
    print("SAM SEGMENT SIMILARITY SEARCH")
    print(f"{'='*80}")
    print(f"Query image: {args.query_image}")
    print(f"Search type: {args.search_type}")
    if args.tags:
        print(f"Tag filters: {', '.join(args.tags)}")
    else:
        print("Tag filters: None (searching all segments)")
    print(f"Top-K: {args.top_k}")
    print(f"{'='*80}\n")
    
    # Load CLIP model
    model, preprocess, device = load_clip_model()
    
    # Extract query image features
    print("Extracting query image features...")
    query_features = extract_image_features(args.query_image, model, preprocess, device)
    
    if query_features is None:
        print("✗ Failed to extract features from query image")
        return 1
    
    print("✓ Features extracted\n")
    
    # Connect to ElasticSearch
    print("Connecting to ElasticSearch...")
    es = Elasticsearch([ES_HOST])
    
    if not es.indices.exists(index=INDEX_NAME):
        print(f"✗ Index {INDEX_NAME} does not exist!")
        print("Please run setup_elasticsearch_sam.py and upload_segments_to_elasticsearch.py first")
        return 1
    
    print(f"✓ Connected to {INDEX_NAME}\n")
    
    # Search
    print("Searching for similar segments...")
    results = search_similar_segments(
        es, query_features, 
        tags=args.tags, 
        top_k=args.top_k,
        search_type=args.search_type
    )
    
    # Print results
    print_results(results, args.query_image)
    
    # Save to file if requested
    if args.output:
        output_data = {
            'query_image': args.query_image,
            'search_type': args.search_type,
            'tag_filters': args.tags,
            'top_k': args.top_k,
            'results': [
                {
                    'rank': i,
                    'score': hit['_score'],
                    'filename': hit['_source']['filename'],
                    'segment_image': hit['_source']['segment_image'],
                    'segment_id': hit['_source']['segment_id'],
                    'tags': hit['_source'].get('tags', [])
                }
                for i, hit in enumerate(results, 1)
            ]
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {args.output}")
    
    es.close()
    print("\n✅ Search completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
