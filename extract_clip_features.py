
import os
import json
import csv
import numpy as np
from PIL import Image
import torch
import clip

def load_clip_model(device="cuda"):
    
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def extract_image_features(image_path, model, preprocess, device):
  
    try:
       
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
      
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def read_csv_image_paths(csv_file='label_cleaned.csv'):
    """
    Read image paths from label_cleaned.csv file.
    Returns a set of image filenames to process.
    """
    image_filenames = set()
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found! Processing all images in folder.")
        return None
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  
            
            for row in reader:
                if row and len(row) > 0:
                    
                    image_path = row[0]
                    if image_path.startswith('dataset/'):
                        
                        filename = os.path.basename(image_path)
                        image_filenames.add(filename)
        
        print(f"Loaded {len(image_filenames)} image filenames from {csv_file}")
        return image_filenames
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def process_images_folder(images_folder, allowed_filenames=None):
    """
    Find images in folder, optionally filtered by allowed_filenames.
    """
    image_files = []
    
   
    for root, dirs, files in os.walk(images_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                
                if allowed_filenames is None or filename in allowed_filenames:
                    filepath = os.path.join(root, filename)
                    image_files.append(filepath)
    
    print(f"Found {len(image_files)} images to process in {images_folder}")
    return sorted(image_files)

def extract_features_from_folder(images_folder, output_folder="features", csv_file="label_cleaned.csv"):
   

    model, preprocess, device = load_clip_model()
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    allowed_filenames = read_csv_image_paths(csv_file)
    
    image_files = process_images_folder(images_folder, allowed_filenames)
    
    if not image_files:
        print("No images found!")
        return
    

    results = []
    
    print(f"\nExtracting features from {len(image_files)} images...")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        features = extract_image_features(image_path, model, preprocess, device)
        
        if features is not None:
  
            filename = os.path.basename(image_path)
            
            result = {
                'image_path': image_path,
                'filename': filename,
                'features': features.tolist(),
                'feature_dimension': len(features)
            }
            
            image_name_without_ext = os.path.splitext(filename)[0]
            output_json_path = os.path.join(output_folder, f"{image_name_without_ext}.json")
            
            with open(output_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            print(f"    Extracted {len(features)}-dim features → {output_json_path}")
        else:
            print(f"    Failed to extract features")
    

    print(f"Total images processed: {len(image_files)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Feature dimension: {results[0]['feature_dimension'] if results else 'N/A'}")
    print(f"Individual JSON files saved to: {output_folder}/")
    print(f"Total JSON files created: {len(results)}")
    
    return results

def main():
    import sys
    
    images_folder = "dataset"
    output_folder = "features"

    if len(sys.argv) >= 3:
        images_folder = sys.argv[1]
        output_folder = sys.argv[2]
    
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' not found!")
        return
    
    print("CLIP Feature Extraction")
    print(f"Input folder: {images_folder}")
    print(f"Output folder: {output_folder}")
    

    results = extract_features_from_folder(images_folder, output_folder)
    
    if results:
        print(f"\n Feature extraction completed!")
        print(f"Images folder: {images_folder}")
        print(f"Features saved to: {output_folder}/")
        print(f"Total files: {len(results)}")
    else:
        print(f"\n No features extracted!")

if __name__ == "__main__":
    main()