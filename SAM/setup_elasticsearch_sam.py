#!/usr/bin/env python3
"""
Setup ElasticSearch index for SAM segmented features
Creates foto_atlas_SAM index with dense vector support for segment features
"""

from elasticsearch import Elasticsearch
import json

# ElasticSearch connection
ES_HOST = "http://localhost:9200"
INDEX_NAME = "foto_atlas_sam"

def create_index():
    """Create foto_atlas_SAM index with proper mappings"""
    
    # Connect to ElasticSearch
    es = Elasticsearch([ES_HOST])
    
    print(f"{'='*80}")
    print(f"Creating ElasticSearch Index: {INDEX_NAME}")
    print(f"{'='*80}")
    
    # Delete existing index if it exists
    if es.indices.exists(index=INDEX_NAME):
        print(f"⚠ Index {INDEX_NAME} already exists. Deleting...")
        es.indices.delete(index=INDEX_NAME)
        print(f"✓ Deleted existing index")
    
    # Index mapping
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # Original image info
                "original_image": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "image_resolution": {"type": "integer"},
                
                # Segment info
                "segment_id": {"type": "integer"},
                "segment_image": {"type": "keyword"},
                "segment_area": {"type": "integer"},
                "segment_bbox": {"type": "float"},
                "stability_score": {"type": "float"},
                
                # CLIP features (512-dimensional dense vector)
                "clip_features": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine"
                },
                
                # Whole image CLIP features (for multi-level search)
                "whole_image_features": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine"
                },
                
                # Tags with confidence scores
                "tags": {
                    "type": "nested",
                    "properties": {
                        "tag": {"type": "keyword"},
                        "confidence": {"type": "float"},
                        "category": {"type": "keyword"}  # architecture, nature, objects
                    }
                },
                
                # Flattened tags for simple search
                "tag_list": {"type": "keyword"},
                
                # Tag categories summary
                "has_architecture": {"type": "boolean"},
                "has_nature": {"type": "boolean"},
                "has_objects": {"type": "boolean"},
                
                # Metadata
                "indexed_at": {"type": "date"}
            }
        }
    }
    
    # Create index
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✓ Created index: {INDEX_NAME}")
    
    # Print mapping info
    print(f"\n{'='*80}")
    print("INDEX STRUCTURE:")
    print(f"{'='*80}")
    print("✓ Segment-level CLIP features (512-dim)")
    print("✓ Whole-image CLIP features (512-dim)")
    print("✓ Tags with categories (Architecture, Nature, Objects)")
    print("✓ Confidence scores for each tag")
    print("✓ Multi-level search support (segments + whole image)")
    print(f"{'='*80}\n")
    
    es.close()
    return True


if __name__ == "__main__":
    try:
        create_index()
        print("✅ ElasticSearch index setup complete!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
