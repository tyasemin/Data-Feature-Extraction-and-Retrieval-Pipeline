#!/usr/bin/env python3
"""
Upload extracted features to Elasticsearch with metadata from label_cleaned.csv
"""

import os
import json
import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Elasticsearch configuration
ES_HOST = 'localhost'
ES_PORT = 9201
INDEX_NAME = 'foto_atlas'

def create_elasticsearch_index(es):
    """Create the Elasticsearch index with proper mapping for dense vectors"""
    
    # Delete existing index if it exists
    if es.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists. Deleting...")
        es.indices.delete(index=INDEX_NAME)
        print(f"✓ Index deleted")
    
    mapping = {
        "mappings": {
            "properties": {
                "image_path": {"type": "keyword"},
                "filename": {"type": "text"},
                "features": {
                    "type": "dense_vector",
                    "dims": 512,
                    "similarity": "cosine"
                },
                "feature_dimension": {"type": "integer"},
                # Metadata from CSV
                "galeri": {"type": "text"},
                "baslik": {"type": "text"},
                "yayinlanma_tarihi": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                "editor": {"type": "text"},
                "olusturanlar": {"type": "text"},
                "kaynaklar": {"type": "text"},
                "turler": {"type": "text"},
                "konular": {"type": "text"},
                "idari_bolgeler": {"type": "text"},
                "etiketler": {"type": "text"},
                "tarih_en_erken": {"type": "integer"},
                "tarih_en_gec": {"type": "integer"},
                "lat": {"type": "float"},
                "lon": {"type": "float"},
                "yon": {"type": "integer"},
                "mesafe": {"type": "float"},
                "aci": {"type": "integer"},
                "source_url": {"type": "keyword"},
                "lisans": {"type": "text"},
                "album_adi": {"type": "text"}
            }
        }
    }
        
    # Create new index
    print(f"Creating index '{INDEX_NAME}'...")
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✓ Index created successfully")

def load_csv_metadata(csv_file='label_cleaned.csv'):
    """Load metadata from CSV file and create a mapping by image filename"""
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return {}
    
    print(f"Loading metadata from {csv_file}...")
    
    metadata_map = {}
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f: 
        reader = csv.DictReader(f)
        
        for row in reader:
            image_path = None
            for key in row.keys():
                if 'Kapak' in key:
                    image_path = row[key].strip()
                    break
            
            if not image_path:
                continue
            
            if image_path.startswith('dataset/'):
                
                filename = os.path.basename(image_path)
                
                def safe_int(value):
                    if not value or value == 'NA':
                        return None
                    try:
                        return int(float(value))
                    except:
                        return None
                
                def safe_float(value):
                    if not value or value == 'NA':
                        return None
                    try:
                        return float(value)
                    except:
                        return None
                
                def safe_str(value):
                    if not value or value == 'NA':
                        return None
                    return value
                
                metadata_map[filename] = {
                    'galeri': safe_str(row.get('Galeri')),
                    'baslik': safe_str(row.get('Başlık')),
                    'yayinlanma_tarihi': safe_str(row.get('Yayınlanma Tarihi')),
                    'editor': safe_str(row.get('Editör')),
                    'olusturanlar': safe_str(row.get('Oluşturanlar')),
                    'kaynaklar': safe_str(row.get('Kaynaklar')),
                    'turler': safe_str(row.get('Türler')),
                    'konular': safe_str(row.get('Konular')),
                    'idari_bolgeler': safe_str(row.get('İdarı Bölgeler')),  
                    'etiketler': safe_str(row.get('Etiketler')),
                    'tarih_en_erken': safe_int(row.get('Tarih En Erken')),
                    'tarih_en_gec': safe_int(row.get('Tarih En Geç')),
                    'lat': safe_float(row.get('Lat')),
                    'lon': safe_float(row.get('Lon')),
                    'yon': safe_int(row.get('Yön')),
                    'mesafe': safe_float(row.get('Mesafe')),
                    'aci': safe_int(row.get('Açı')),
                    'source_url': safe_str(row.get('Source URL')),
                    'lisans': safe_str(row.get('Lisans')),
                    'album_adi': safe_str(row.get('Albüm Adı'))
                }
    
    print(f"Loaded metadata for {len(metadata_map)} images")
    return metadata_map

def load_feature_files(features_dir='features'):
    """Load all feature JSON files from the features directory"""
    
    if not os.path.exists(features_dir):
        print(f"Error: Features directory '{features_dir}' not found!")
        return []
    
    print(f"Loading feature files from {features_dir}...")
    
    feature_files = []
    for filename in os.listdir(features_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(features_dir, filename)
            feature_files.append(filepath)
    
    print(f"✓ Found {len(feature_files)} feature files")
    return sorted(feature_files)

def prepare_documents(feature_files, metadata_map):
    """Prepare documents for Elasticsearch by combining features and metadata"""
    
    documents = []
    matched = 0
    not_matched = 0
    
    print(f"Preparing documents for upload...")
    
    for i, feature_file in enumerate(feature_files, 1):
        try:
            with open(feature_file, 'r') as f:
                feature_data = json.load(f)
            
            filename = feature_data['filename']
            
            document = {
                'image_path': feature_data['image_path'],
                'filename': filename,
                'features': feature_data['features'],
                'feature_dimension': feature_data['feature_dimension']
            }

            if filename in metadata_map:
                document.update(metadata_map[filename])
                matched += 1
                if matched <= 5:
                    print(f" Matched: {filename}")
            else:
                not_matched += 1
                if not_matched <= 5:
                    print(f" Not matched: {filename}")
            
            documents.append(document)
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(feature_files)} files... ({matched} matched, {not_matched} not matched)")
        
        except Exception as e:
            print(f"Error processing {feature_file}: {e}")
    
    print(f"Total documents prepared: {len(documents)}")
    print(f"Matched with CSV: {matched}")
    print(f"Not matched: {not_matched}")
    
    return documents

def upload_to_elasticsearch(es, documents):
    """Upload documents to Elasticsearch using bulk API"""
    
    print(f"\nUploading {len(documents)} documents to Elasticsearch...")

    actions = []
    for doc in documents:
        action = {
            "_index": INDEX_NAME,
            "_source": doc
        }
        actions.append(action)
    
    try:
        success, failed = bulk(es, actions, raise_on_error=False, stats_only=True)
        print(f"✓ Successfully uploaded: {success} documents")
        if failed > 0:
            print(f"✗ Failed to upload: {failed} documents")
    except Exception as e:
        print(f"Error during bulk upload: {e}")
        return False
    
    return True

def main():
    """Main function to orchestrate the upload process"""
    
    print("Elasticsearch Upload - Features + Metadata")
    
    features_dir = 'features'
    csv_file = 'label_cleaned.csv'
    
    if not os.path.exists(features_dir):
        print(f"Error: Features directory '{features_dir}' not found!")
        print("Please run extract_clip_features.py first to generate features.")
        return
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        return
    
    # Connect to Elasticsearch
    print(f"\nConnecting to Elasticsearch at {ES_HOST}:{ES_PORT}...")
    es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
    
    if not es.ping():
        print(f"✗ Failed to connect to Elasticsearch at {ES_HOST}:{ES_PORT}")
        print("Please make sure Elasticsearch is running.")
        return
    
    print(" Connected to Elasticsearch")
    
    create_elasticsearch_index(es)
    
    metadata_map = load_csv_metadata(csv_file)

    feature_files = load_feature_files(features_dir)
    
    if not feature_files:
        print("No feature files found!")
        return
    
    documents = prepare_documents(feature_files, metadata_map)
    
    if not documents:
        print("No documents to upload!")
        return
    
    success = upload_to_elasticsearch(es, documents)
    
    if success:

        es.indices.refresh(index=INDEX_NAME)
        count = es.count(index=INDEX_NAME)['count']
        
        print(f"Index name: {INDEX_NAME}")
        print(f"Total documents in index: {count}")
        print(f"Elasticsearch: http://{ES_HOST}:{ES_PORT}")
        print(f"Kibana: http://{ES_HOST}:5602")
        print("\n Upload completed successfully!")
    else:
        print("\n Upload failed!")

if __name__ == "__main__":
    main()
