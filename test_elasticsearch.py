from elasticsearch import Elasticsearch
import json
import numpy as np

# Connect to Elasticsearch
es = Elasticsearch(['http://localhost:9201'])
INDEX_NAME = 'foto_atlas'

def test_connection():

    try:
        if es.ping():
            print("Elasticsearch connection successful")
            return True
        else:
            print("Could not connect to Elasticsearch")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_index_exists():

    try:
        if es.indices.exists(index=INDEX_NAME):
            print(" Index 'foto_atlas' exists")
            return True
        else:
            print(" Index 'foto_atlas' does not exist")
            return False
    except Exception as e:
        print(f" Error checking index: {e}")
        return False

def test_document_count():

    try:
        response = es.count(index=INDEX_NAME)
        doc_count = response['count']
        print(f" Index contains {doc_count} documents")
        return doc_count > 0
    except Exception as e:
        print(f" Error counting documents: {e}")
        return False

def test_sample_search():

    try:
        response = es.search(
            index=INDEX_NAME,
            body={"query": {"match_all": {}}, "size": 1}
        )
        
        if response['hits']['total']['value'] > 0:
            sample_doc = response['hits']['hits'][0]['_source']
            print(f"Sample document found: {sample_doc['filename']}")
            return True
        else:
            print(" No documents found in search")
            return False
    except Exception as e:
        print(f" Search error: {e}")
        return False

def test_similarity_search():
    try:
        # Create a random 512-dimensional vector for testing
        random_vector = np.random.random(512).tolist()
        
        query = {
            "size": 3,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'features') + 1.0",
                        "params": {"query_vector": random_vector}
                    }
                }
            }
        }
        
        response = es.search(index=INDEX_NAME, body=query)
        
        if response['hits']['total']['value'] > 0:
            print("✓ Similarity search working")
            print(f"  Found {len(response['hits']['hits'])} similar images")
            for hit in response['hits']['hits']:
                filename = hit['_source']['filename']
                score = hit['_score']
                print(f"    - {filename} (score: {score:.4f})")
            return True
        else:
            print("✗ Similarity search returned no results")
            return False
    except Exception as e:
        print(f"✗ Similarity search error: {e}")
        return False

def test_landmark_search():
    try:
        
        response = es.search(
            index=INDEX_NAME,
            body={"query": {"match_all": {}}, "size": 1}
        )
        
        if response['hits']['total']['value'] > 0:
            landmark = response['hits']['hits'][0]['_source']['landmark']
            
            
            landmark_query = {
                "query": {
                    "term": {
                        "landmark": landmark
                    }
                }
            }
            
            landmark_response = es.search(index=INDEX_NAME, body=landmark_query)
            count = landmark_response['hits']['total']['value']
            
            print(f" Landmark search working")
            print(f"  Found {count} images for landmark: {landmark}")
            return True
        else:
            print(" No documents available for landmark search test")
            return False
    except Exception as e:
        print(f" Landmark search error: {e}")
        return False

def run_all_tests():

    print("Running Elasticsearch tests for foto_atlas index...\n")
    
    tests = [
        ("Connection Test", test_connection),
        ("Index Existence", test_index_exists),
        ("Document Count", test_document_count),
        ("Sample Search", test_sample_search),
        ("Similarity Search", test_similarity_search),
        ("Landmark Search", test_landmark_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f" All tests passed! ({passed}/{total})")
        print("Your Elasticsearch setup is working correctly.")
    else:
        print(f" {passed}/{total} tests passed.")
        print("Some issues were found. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()