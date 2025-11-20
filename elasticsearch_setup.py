from elasticsearch import Elasticsearch
import json
import numpy as np


es = Elasticsearch(['http://localhost:9201'])


INDEX_NAME = 'foto_atlas'

def create_index():

    mapping = {
        "mappings": {
            "properties": {
                "image_path": {
                    "type": "text"
                },
                "filename": {
                    "type": "keyword"
                },
                "landmark": {
                    "type": "keyword"
                },
                "features": {
                    "type": "dense_vector",
                    "dims": 512,
                    "similarity": "cosine"
                },
                "feature_dimension": {
                    "type": "integer"
                }
            }
        }
    }
    

    # Create new index
    response = es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index: {INDEX_NAME}")
    return response

def load_and_index_features(json_file_path):

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
 
    bulk_data = []
    
    for idx, item in enumerate(data):
        
        if len(item['features']) < 512:
            print(f"Skipping {item['filename']} - incomplete features ({len(item['features'])} dimensions)")
            continue
        
        doc = {
            "image_path": item['image_path'],
            "filename": item['filename'],
            "landmark": item['landmark'],
            "features": item['features'],
            "feature_dimension": item['feature_dimension']
        }
        
        bulk_data.append({
            "index": {
                "_index": INDEX_NAME,
                "_id": idx
            }
        })
        bulk_data.append(doc)
    
    
    if bulk_data:
        response = es.bulk(body=bulk_data)
        
        
        if response['errors']:
            print("Some documents failed to index:")
            for item in response['items']:
                if 'index' in item and 'error' in item['index']:
                    print(f"Error: {item['index']['error']}")
        else:
            print(f"Successfully indexed {len(bulk_data)//2} documents")
        
        es.indices.refresh(index=INDEX_NAME)
        print("Index refreshed - documents are now searchable")
    
    return response

def search_similar_images(query_features, top_k=5):

    
    query = {
        "size": top_k,
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
    
    response = es.search(index=INDEX_NAME, body=query)
    
    results = []
    for hit in response['hits']['hits']:
        result = {
            'filename': hit['_source']['filename'],
            'landmark': hit['_source']['landmark'],
            'image_path': hit['_source']['image_path'],
            'similarity_score': hit['_score'],
            'id': hit['_id']
        }
        results.append(result)
    
    return results

def search_by_landmark(landmark_name, top_k=10):

    query = {
        "size": top_k,
        "query": {
            "term": {
                "landmark": landmark_name
            }
        }
    }
    
    response = es.search(index=INDEX_NAME, body=query)
    
    results = []
    for hit in response['hits']['hits']:
        result = {
            'filename': hit['_source']['filename'],
            'landmark': hit['_source']['landmark'],
            'image_path': hit['_source']['image_path'],
            'id': hit['_id']
        }
        results.append(result)
    
    return results

def get_all_landmarks():

    query = {
        "size": 0,
        "aggs": {
            "landmarks": {
                "terms": {
                    "field": "landmark",
                    "size": 100
                }
            }
        }
    }
    
    response = es.search(index=INDEX_NAME, body=query)
    landmarks = [bucket['key'] for bucket in response['aggregations']['landmarks']['buckets']]
    return landmarks

def get_index_stats():
    stats = es.indices.stats(index=INDEX_NAME)
    doc_count = stats['indices'][INDEX_NAME]['total']['docs']['count']
    size = stats['indices'][INDEX_NAME]['total']['store']['size_in_bytes']
    
    return {
        'document_count': doc_count,
        'size_bytes': size,
        'size_mb': round(size / (1024 * 1024), 2)
    }


if __name__ == "__main__":
    # Check Elasticsearch connection
    if not es.ping():
        print("Error: Could not connect to Elasticsearch. Make sure it's running on localhost:9201")
        exit(1)
    
    print("Connected to Elasticsearch successfully!")
    
    create_index()
    
    json_file_path = "istanbul_landmarks_clip_features.json"
    load_and_index_features(json_file_path)
    
    stats = get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"Documents: {stats['document_count']}")
    print(f"Size: {stats['size_mb']} MB")
    
    landmarks = get_all_landmarks()
    print(f"\nAvailable landmarks ({len(landmarks)}):")
    for landmark in landmarks:
        print(f"  - {landmark}")
    
    print(f"\nSetup complete!")