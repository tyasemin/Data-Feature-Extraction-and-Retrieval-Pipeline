#!/usr/bin/env python3

import os
import sys
import json
import argparse
import torch
import clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from elasticsearch import Elasticsearch

# Elasticsearch connection (support both Docker and local)
ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = os.getenv('ES_PORT', '9201')
es = Elasticsearch([f'http://{ES_HOST}:{ES_PORT}'])
INDEX_NAME = 'foto_atlas'

def extract_single_image_features(image_path):
    """Extract CLIP features from a single image."""
    
    print(f"\nExtracting CLIP features from: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        features = image_features.cpu().numpy().flatten().tolist()
        
        print(f"✓ Successfully extracted {len(features)}-dimensional features")
        return features
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

def search_similar_images(query_features, top_k=10):
    """Search for similar images using cosine similarity."""
    
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
        source = hit['_source']
        similarity_percentage = (hit['_score'] / 2.0) * 100
        
        result = {
            'filename': source.get('filename', 'Unknown'),
            'image_path': source.get('image_path', 'Unknown'),
            'similarity_score': hit['_score'],
            'similarity_percentage': similarity_percentage,
            'id': hit['_id'],
            # CSV metadata
            'galeri': source.get('galeri'),
            'baslik': source.get('baslik'),
            'yayinlanma_tarihi': source.get('yayinlanma_tarihi'),
            'editor': source.get('editor'),
            'olusturanlar': source.get('olusturanlar'),
            'kaynaklar': source.get('kaynaklar'),
            'turler': source.get('turler'),
            'konular': source.get('konular'),
            'idari_bolgeler': source.get('idari_bolgeler'),
            'etiketler': source.get('etiketler'),
            'tarih_en_erken': source.get('tarih_en_erken'),
            'tarih_en_gec': source.get('tarih_en_gec'),
            'lat': source.get('lat'),
            'lon': source.get('lon'),
            'yon': source.get('yon'),
            'mesafe': source.get('mesafe'),
            'aci': source.get('aci'),
            'source_url': source.get('source_url'),
            'lisans': source.get('lisans'),
            'album_adi': source.get('album_adi')
        }
        results.append(result)
    
    return results

def display_results(results):
    """Display search results with CSV metadata."""
    if not results:
        print("No similar images found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Top {len(results)} Most Similar Images")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Rank #{i} - Similarity: {result['similarity_percentage']:.2f}% (Score: {result['similarity_score']:.4f})")
        print(f"{'─'*80}")
        
        print(f"📁 Filename: {result['filename']}")
        print(f"📂 Path: {result['image_path']}")
        
        if result.get('baslik'):
            print(f"\n Başlık: {result['baslik']}")
        if result.get('galeri'):
            print(f"  Galeri: {result['galeri']}")
        if result.get('olusturanlar'):
            print(f" Oluşturanlar: {result['olusturanlar']}")
        if result.get('yayinlanma_tarihi'):
            print(f" Yayınlanma Tarihi: {result['yayinlanma_tarihi']}")
        if result.get('editor'):
            print(f"  Editör: {result['editor']}")
        if result.get('kaynaklar'):
            print(f" Kaynaklar: {result['kaynaklar']}")
        if result.get('turler'):
            print(f"  Türler: {result['turler']}")
        if result.get('konular'):
            print(f" Konular: {result['konular']}")
        if result.get('idari_bolgeler'):
            print(f"  İdari Bölgeler: {result['idari_bolgeler']}")
        if result.get('etiketler'):
            print(f"  Etiketler: {result['etiketler']}")
        
        # Date and location info
        if result.get('tarih_en_erken') or result.get('tarih_en_gec'):
            erken = result.get('tarih_en_erken', '?')
            gec = result.get('tarih_en_gec', '?')
            print(f"  Tarih: {erken} - {gec}")
        
        if result.get('lat') and result.get('lon'):
            print(f" Konum: {result['lat']}, {result['lon']}")
            if result.get('yon'):
                print(f"   Yön: {result['yon']}°")
            if result.get('mesafe'):
                print(f"   Mesafe: {result['mesafe']} km")
            if result.get('aci'):
                print(f"   Açı: {result['aci']}°")
        
        if result.get('source_url'):
            print(f" Kaynak URL: {result['source_url']}")
        if result.get('lisans'):
            print(f"  Lisans: {result['lisans']}")
        if result.get('album_adi'):
            print(f" Albüm Adı: {result['album_adi']}")
    
    print(f"\n{'='*80}")

def create_visualization(query_image_path, results, output_path):
    """Create a visualization with query image and top results with metadata."""
    
    print(f"\n Creating visualization...")
    
    # Configuration
    num_results = min(len(results), 3)
    img_width = 300
    img_height = 300
    text_height = 200
    padding = 20
    
    # Calculate canvas size
    canvas_width = (num_results + 1) * (img_width + padding) + padding
    canvas_height = img_height + text_height + padding * 2
    
    # Create white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add main title
    main_title = "Image Similarity Search Results"
    bbox = draw.textbbox((0, 0), main_title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(((canvas_width - title_width) // 2, 5), main_title, fill='black', font=title_font)
    
    # Process query image
    x_pos = padding
    y_pos = padding + 30
    
    try:
        query_img = Image.open(query_image_path).convert('RGB')
        query_img.thumbnail((img_width, img_height), Image.Resampling.LANCZOS)
        
        # Center the image if it's smaller
        img_x = x_pos + (img_width - query_img.width) // 2
        img_y = y_pos + (img_height - query_img.height) // 2
        canvas.paste(query_img, (img_x, img_y))
        
        # Draw border
        draw.rectangle([x_pos, y_pos, x_pos + img_width, y_pos + img_height], outline='gray', width=2)
        
        # Add label
        label_y = y_pos + img_height + 10
        draw.text((x_pos, label_y), "Query Image", fill='black', font=title_font)
        draw.text((x_pos, label_y + 20), f"({os.path.basename(query_image_path)})", fill='gray', font=small_font)
        
    except Exception as e:
        print(f"Warning: Could not load query image: {e}")
    
    # Process result images
    for i, result in enumerate(results[:num_results]):
        x_pos = padding + (i + 1) * (img_width + padding)
        
        # Load and display result image
        image_path = result['image_path']
        if not os.path.exists(image_path):
            # Try without leading slash
            image_path = image_path.lstrip('/')
        
        try:
            result_img = Image.open(image_path).convert('RGB')
            result_img.thumbnail((img_width, img_height), Image.Resampling.LANCZOS)
            
            # Center the image
            img_x = x_pos + (img_width - result_img.width) // 2
            img_y = y_pos + (img_height - result_img.height) // 2
            canvas.paste(result_img, (img_x, img_y))
            
            # Draw border
            draw.rectangle([x_pos, y_pos, x_pos + img_width, y_pos + img_height], outline='green', width=2)
            
            # Add rank and similarity
            label_y = y_pos + img_height + 10
            rank_text = f"#{i + 1} • {result['similarity_percentage']:.1f}%"
            draw.text((x_pos, label_y), rank_text, fill='green', font=title_font)
            
            # Add metadata text
            text_y = label_y + 25
            line_height = 15
            
            # Title
            if result.get('baslik'):
                title = result['baslik'][:30] + '...' if len(result['baslik']) > 30 else result['baslik']
                draw.text((x_pos, text_y), title, fill='black', font=text_font)
                text_y += line_height
            
            # Filename
            filename = result['filename'][:35] + '...' if len(result['filename']) > 35 else result['filename']
            draw.text((x_pos, text_y), filename, fill='gray', font=small_font)
            text_y += line_height
            
            # Creator
            if result.get('olusturanlar'):
                creator = result['olusturanlar'][:30] + '...' if len(result['olusturanlar']) > 30 else result['olusturanlar']
                draw.text((x_pos, text_y), creator, fill='gray', font=small_font)
                text_y += line_height
            
            # Date range
            if result.get('tarih_en_erken') or result.get('tarih_en_gec'):
                erken = result.get('tarih_en_erken', '?')
                gec = result.get('tarih_en_gec', '?')
                draw.text((x_pos, text_y), f" {erken}-{gec}", fill='gray', font=small_font)
                text_y += line_height
            
            # Location
            if result.get('idari_bolgeler'):
                location = result['idari_bolgeler'].split(',')[0]  # First part only
                location = location[:30] + '...' if len(location) > 30 else location
                draw.text((x_pos, text_y), f" {location}", fill='gray', font=small_font)
                text_y += line_height
            
        except Exception as e:
            print(f"Warning: Could not load result image {image_path}: {e}")
            draw.text((x_pos, y_pos + 50), "Image not available", fill='red', font=text_font)
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    canvas.save(output_path)
    print(f"✓ Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Search for similar images using CLIP features')
    parser.add_argument('image_path', help='Path to the test image')
    parser.add_argument('--top-k', type=int, default=10, help='Number of similar images to return (default: 10)')
    parser.add_argument('--output', default='temp_test/similarity_results.json', help='Output JSON file path')
    parser.add_argument('--visualize', action='store_true', help='Create visualization image')
    parser.add_argument('--viz-output', default='temp_test/similarity_visualization.png', help='Visualization output path')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Image Similarity Search with CLIP Features")
    print("="*80)
    print(f"Test Image: {args.image_path}")
    print(f"Top K Results: {args.top_k}")
    print(f"Output File: {args.output}")
    print("="*80)
    
    # Check Elasticsearch connection
    if not es.ping():
        print(f"\n Error: Cannot connect to Elasticsearch at {ES_HOST}:{ES_PORT}")
        print("Make sure Elasticsearch is running (docker compose up -d)")
        return 1
    
    print("\n✓ Connected to Elasticsearch")
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"\n Error: Image not found at {args.image_path}")
        return 1
    
    # Extract features from test image
    features = extract_single_image_features(args.image_path)
    if features is None:
        print("\n Failed to extract features from test image")
        return 1
    
    # Search for similar images
    print(f"\n Searching for the top {args.top_k} most similar images...")
    results = search_similar_images(features, top_k=args.top_k)
    
    if not results:
        print("\n No similar images found")
        return 1
    
    # Display results
    display_results(results)
    
    # Save results to JSON
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Create visualization if requested
    if args.visualize:
        create_visualization(args.image_path, results, args.viz_output)
    
    print("\n" + "="*80)
    print("Search completed successfully!")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())