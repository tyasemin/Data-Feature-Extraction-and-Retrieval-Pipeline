#!/usr/bin/env python3
"""
Upload SAM segment features to ElasticSearch
Indexes both segment-level and whole-image features with tags
"""

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import os
from pathlib import Path
from datetime import datetime
import argparse

# ElasticSearch connection
ES_HOST = "http://localhost:9200"
INDEX_NAME = "foto_atlas_sam"

# Tag categories
ARCHITECTURE_TAGS = [
    "mosque", "church", "tower", "minaret", "dome", "palace", "castle", "bridge",
    "gate", "arch", "column", "fortress", "monument", "building", "wall",
    "roof", "window", "balcony", "courtyard", "fountain"
]

NATURE_TAGS = [
    "water", "sea", "sky", "mountain", "tree", "cloud", "river", "hill",
    "garden", "vegetation"
]

OBJECT_TAGS = [
    "people", "person", "crowd", "boat", "ship", "vehicle", "flag", "statue",
    "sculpture", "decoration", "ornament", "painting", "sign", "text",
    "street", "square", "market", "harbor", "panorama", "cityscape"
]


def get_tag_category(tag):
    """Determine which category a tag belongs to"""
    if tag in ARCHITECTURE_TAGS:
        return "architecture"
    elif tag in NATURE_TAGS:
        return "nature"
    elif tag in OBJECT_TAGS:
        return "objects"
    return "unknown"


def load_segment_data(features_dir):
    """Load all segment features from JSON files"""
    features_files = list(Path(features_dir).glob("*.json"))
    print(f"Found {len(features_files)} feature files")
    
    all_segments = []
    
    for features_file in features_files:
        try:
            with open(features_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract whole image features (average of all segment features)
            whole_image_features = None
            if data.get('segments'):
                # Average all segment features for whole image representation
                all_features = [seg['features'] for seg in data['segments'] if seg.get('features')]
                if all_features:
                    import numpy as np
                    whole_image_features = np.mean(all_features, axis=0).tolist()
            
            # Process each segment
            for segment in data.get('segments', []):
                # Categorize tags
                tags_with_category = []
                tag_list = []
                has_architecture = False
                has_nature = False
                has_objects = False
                
                for tag_info in segment.get('tags', []):
                    tag_name = tag_info['tag']
                    category = get_tag_category(tag_name)
                    
                    tags_with_category.append({
                        'tag': tag_name,
                        'confidence': tag_info['confidence'],
                        'category': category
                    })
                    tag_list.append(tag_name)
                    
                    # Set category flags
                    if category == 'architecture':
                        has_architecture = True
                    elif category == 'nature':
                        has_nature = True
                    elif category == 'objects':
                        has_objects = True
                
                # Create document
                doc = {
                    'original_image': data['original_image'],
                    'filename': data['filename'],
                    'image_resolution': data['image_resolution'],
                    'segment_id': segment['segment_id'],
                    'segment_image': segment['segment_image'],
                    'segment_area': segment['area'],
                    'segment_bbox': segment['bbox'],
                    'stability_score': segment['stability_score'],
                    'clip_features': segment['features'],
                    'whole_image_features': whole_image_features,
                    'tags': tags_with_category,
                    'tag_list': tag_list,
                    'has_architecture': has_architecture,
                    'has_nature': has_nature,
                    'has_objects': has_objects,
                    'indexed_at': datetime.utcnow().isoformat()
                }
                
                all_segments.append(doc)
                
        except Exception as e:
            print(f"✗ Error processing {features_file}: {e}")
    
    return all_segments


def upload_to_elasticsearch(segments, batch_size=100):
    """Upload segments to ElasticSearch using bulk API"""
    
    es = Elasticsearch([ES_HOST])
    
    print(f"\n{'='*80}")
    print(f"UPLOADING TO ELASTICSEARCH: {INDEX_NAME}")
    print(f"{'='*80}")
    print(f"Total segments to upload: {len(segments)}")
    
    # Prepare bulk actions
    actions = []
    for segment in segments:
        action = {
            '_index': INDEX_NAME,
            '_source': segment
        }
        actions.append(action)
    
    # Upload in batches
    success_count = 0
    error_count = 0
    
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i+batch_size]
        try:
            success, errors = bulk(es, batch, raise_on_error=False)
            success_count += success
            if errors:
                error_count += len(errors)
                print(f"  ⚠ Batch {i//batch_size + 1}: {success} success, {len(errors)} errors")
            else:
                print(f"  ✓ Batch {i//batch_size + 1}: {success} documents indexed")
        except Exception as e:
            print(f"  ✗ Batch {i//batch_size + 1} failed: {e}")
            error_count += len(batch)
    
    print(f"\n{'='*80}")
    print("UPLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Successfully indexed: {success_count}")
    if error_count > 0:
        print(f"✗ Errors: {error_count}")
    print(f"{'='*80}\n")
    
    # Refresh index
    es.indices.refresh(index=INDEX_NAME)
    
    # Get index stats
    count = es.count(index=INDEX_NAME)['count']
    print(f"✓ Index {INDEX_NAME} now contains {count} documents")
    
    es.close()
    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description='Upload SAM segment features to ElasticSearch'
    )
    parser.add_argument('--features-dir', type=str,
                       default='/workspace/SAM/segmented_features',
                       help='Directory containing feature JSON files')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for bulk upload')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("SAM SEGMENTS → ELASTICSEARCH UPLOAD")
    print(f"{'='*80}")
    print(f"Features directory: {args.features_dir}")
    print(f"Target index: {INDEX_NAME}")
    print(f"ElasticSearch host: {ES_HOST}")
    print(f"{'='*80}\n")
    
    # Load segment data
    print("Loading segment data...")
    segments = load_segment_data(args.features_dir)
    
    if not segments:
        print("✗ No segments found to upload!")
        return 1
    
    # Upload to ElasticSearch
    success, errors = upload_to_elasticsearch(segments, args.batch_size)
    
    if errors == 0:
        print("\n✅ Upload completed successfully!")
        return 0
    else:
        print(f"\n⚠ Upload completed with {errors} errors")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
