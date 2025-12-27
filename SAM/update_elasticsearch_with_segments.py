
import os
import json
import argparse
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np

# Elasticsearch configuration
ES_HOST = 'localhost'
ES_PORT = 9201
INDEX_NAME = 'foto_atlas'


def connect_to_elasticsearch():
    """Connect to Elasticsearch"""
    es = Elasticsearch([f'http://{ES_HOST}:{ES_PORT}'])
    
    if not es.ping():
        raise Exception("Cannot connect to Elasticsearch!")
    
    print(f"✓ Connected to Elasticsearch at {ES_HOST}:{ES_PORT}")
    return es


def update_index_mapping(es):
    """Update the index mapping to include segment fields"""
    
    if not es.indices.exists(index=INDEX_NAME):
        print(f"✗ Index '{INDEX_NAME}' does not exist!")
        return False
    
    print(f"Updating mapping for index '{INDEX_NAME}'...")
    
    # Add new fields for segments
    new_mapping = {
        "properties": {
            "segmented_features": {
                "properties": {
                    "num_segments": {"type": "integer"},
                    "image_resolution": {"type": "integer"},
                    "segments": {
                        "type": "nested",
                        "properties": {
                            "segment_id": {"type": "integer"},
                            "area": {"type": "integer"},
                            "bbox": {"type": "float"},
                            "stability_score": {"type": "float"},
                            "segment_image": {"type": "keyword"},
                            "features": {
                                "type": "dense_vector",
                                "dims": 512,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            },
            "segmented_tags": {
                "properties": {
                    "num_segments": {"type": "integer"},
                    "segment_tags": {
                        "type": "nested",
                        "properties": {
                            "segment_id": {"type": "integer"},
                            "segment_image": {"type": "keyword"},
                            "area": {"type": "integer"},
                            "tags": {
                                "type": "nested",
                                "properties": {
                                    "tag": {"type": "keyword"},
                                    "confidence": {"type": "float"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    try:
        es.indices.put_mapping(index=INDEX_NAME, body=new_mapping)
        print(" Index mapping updated successfully")
        return True
    except Exception as e:
        print(f" Failed to update mapping: {e}")
        return False


def load_segment_features(features_dir):
    """Load all segment features from JSON files"""
    features_map = {}
    
    feature_files = list(Path(features_dir).glob("*.json"))
    print(f"\nLoading segment features from {features_dir}...")
    print(f"Found {len(feature_files)} feature files")
    
    for feature_file in feature_files:
        try:
            with open(feature_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = data['filename']
            features_map[filename] = data
            
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
    
    print(f" Loaded features for {len(features_map)} images")
    return features_map


def load_segment_tags(tags_dir):
    """Load all segment tags from JSON files"""
    tags_map = {}
    
    tag_files = list(Path(tags_dir).glob("*_tags.json"))
    print(f"\nLoading segment tags from {tags_dir}...")
    print(f"Found {len(tag_files)} tag files")
    
    for tag_file in tag_files:
        try:
            with open(tag_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = data['filename']
            tags_map[filename] = data
            
        except Exception as e:
            print(f"Error loading {tag_file}: {e}")
    
    print(f" Loaded tags for {len(tags_map)} images")
    return tags_map


def get_document_by_filename(es, filename):
    """Find document in Elasticsearch by filename (with or without extension)"""
    # Try different extensions: no extension, .jpg, .jpeg
    extensions_to_try = [filename]
    
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        extensions_to_try.extend([f"{filename}.jpg", f"{filename}.jpeg", f"{filename}.png"])
    
    for fname in extensions_to_try:
        query = {
            "query": {
                "match_phrase": {
                    "filename": fname
                }
            }
        }
        
        try:
            result = es.search(index=INDEX_NAME, body=query)
            if result['hits']['total']['value'] > 0:
                return result['hits']['hits'][0]
        except Exception as e:
            continue
    
    return None


def prepare_update_actions(es, features_map, tags_map):
    """Prepare bulk update actions for Elasticsearch"""
    
    actions = []
    updated_count = 0
    not_found_count = 0
    error_count = 0
    
    print(f"\nPreparing updates for {len(features_map)} documents...")
    
    for i, (filename, feature_data) in enumerate(features_map.items(), 1):
        try:
            # Find the document in ES
            doc = get_document_by_filename(es, filename)
            
            if not doc:
                not_found_count += 1
                if not_found_count <= 5:
                    print(f"   Document not found in ES: {filename}")
                continue
            
            doc_id = doc['_id']
            
            # Prepare update with segments
            update_doc = {}
            
            # Add segment features
            if filename in features_map:
                segment_features = features_map[filename]
                update_doc['segmented_features'] = {
                    'num_segments': segment_features['num_segments'],
                    'image_resolution': segment_features['image_resolution'],
                    'segments': segment_features['segments']
                }
            
            # Add segment tags
            if filename in tags_map:
                segment_tags = tags_map[filename]
                update_doc['segmented_tags'] = {
                    'num_segments': segment_tags['num_segments'],
                    'segment_tags': segment_tags['segment_tags']
                }
            
            # Create bulk update action
            action = {
                '_op_type': 'update',
                '_index': INDEX_NAME,
                '_id': doc_id,
                'doc': update_doc
            }
            
            actions.append(action)
            updated_count += 1
            
            if updated_count <= 5:
                print(f"   Prepared update for: {filename}")
            
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(features_map)} processed ({updated_count} updates prepared)")
        
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"   Error preparing {filename}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Update Preparation Summary:")
    print(f"  Total processed: {len(features_map)}")
    print(f"  Updates prepared: {updated_count}")
    print(f"  Not found in ES: {not_found_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*80}\n")
    
    return actions


def execute_bulk_update(es, actions):
    """Execute bulk update to Elasticsearch"""
    
    if not actions:
        print("No actions to execute!")
        return
    
    print(f"Executing bulk update for {len(actions)} documents...")
    
    try:
        success, failed = bulk(es, actions, raise_on_error=False, stats_only=False)
        
        print(f"\n{'='*80}")
        print(f"Bulk Update Results:")
        print(f"  Successfully updated: {success}")
        print(f"  Failed: {len(failed) if isinstance(failed, list) else 0}")
        print(f"{'='*80}\n")
        
        if failed and len(failed) > 0:
            print("Sample failures:")
            for i, fail in enumerate(failed[:5], 1):
                print(f"  {i}. {fail}")
        
        return success
        
    except Exception as e:
        print(f" Bulk update failed: {e}")
        return 0


def verify_updates(es, sample_filenames):
    """Verify that updates were successful by checking a few documents"""
    
    print(f"\nVerifying updates for sample documents...")
    
    for filename in sample_filenames[:3]:
        doc = get_document_by_filename(es, filename)
        if doc:
            source = doc['_source']
            has_features = 'segmented_features' in source
            has_tags = 'segmented_tags' in source
            
            print(f"\n  Document: {filename}")
            print(f"    Has full image features: {'features' in source}")
            print(f"    Has segmented features: {has_features}")
            print(f"    Has segmented tags: {has_tags}")
            
            if has_features:
                print(f"    Num segments: {source['segmented_features']['num_segments']}")
            if has_tags:
                print(f"    Num tag segments: {source['segmented_tags']['num_segments']}")


def main():
    parser = argparse.ArgumentParser(
        description='Update Elasticsearch with SAM segment features and tags'
    )
    parser.add_argument('--features-dir', type=str,
                       default='segmented_features',
                       help='Directory containing segment feature JSON files')
    parser.add_argument('--tags-dir', type=str,
                       default='segmented_tags',
                       help='Directory containing segment tag JSON files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Prepare updates but do not execute them')
    
    args = parser.parse_args()
    
    print("="*80)
    print("UPDATE ELASTICSEARCH WITH SAM SEGMENTS")
    print("="*80)
    print(f"Features directory: {args.features_dir}")
    print(f"Tags directory: {args.tags_dir}")
    print(f"Elasticsearch: {ES_HOST}:{ES_PORT}")
    print(f"Index: {INDEX_NAME}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    # Connect to Elasticsearch
    es = connect_to_elasticsearch()
    
    # Update index mapping
    if not update_index_mapping(es):
        print("Failed to update index mapping. Continuing anyway...")
    
    # Load segment features
    features_map = load_segment_features(args.features_dir)
    
    # Load segment tags
    tags_map = load_segment_tags(args.tags_dir)
    
    if not features_map:
        print("✗ No features loaded!")
        return 1
    
    # Prepare bulk update actions
    actions = prepare_update_actions(es, features_map, tags_map)
    
    if args.dry_run:
        print("\n✓ Dry run complete. No changes made to Elasticsearch.")
        return 0
    
    # Execute bulk update
    success_count = execute_bulk_update(es, actions)
    
    # Verify updates
    if success_count > 0:
        verify_updates(es, list(features_map.keys()))
    
    print("\n✓ Update complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
