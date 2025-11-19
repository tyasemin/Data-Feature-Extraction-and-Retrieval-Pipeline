
import os
import json
import numpy as np
from PIL import Image
import torch
import clip

def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    
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

def process_images_folder(images_folder):
    
    image_files = []
    
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filepath = os.path.join(images_folder, filename)
            image_files.append(filepath)
    
    print(f"Found {len(image_files)} images in {images_folder}")
    return sorted(image_files)

def extract_features_from_folder(images_folder, output_file="clip_features.json"):
   

    model, preprocess, device = load_clip_model()
    

    image_files = process_images_folder(images_folder)
    
    if not image_files:
        print("No images found!")
        return
    

    results = []
    
    print(f"\nExtracting features from {len(image_files)} images...")
    print("-" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        

        features = extract_image_features(image_path, model, preprocess, device)
        
        if features is not None:
  
            filename = os.path.basename(image_path)
            landmark_name = "_".join(filename.split("_")[:-1])  # Remove _1.jpg, _2.jpg etc.
            
            result = {
                'image_path': image_path,
                'filename': filename,
                'landmark': landmark_name,
                'features': features.tolist(),
                'feature_dimension': len(features)
            }
            
            results.append(result)
            print(f"    ✓ Extracted {len(features)}-dim features")
        else:
            print(f"    ✗ Failed to extract features")
    

    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Feature dimension: {results[0]['feature_dimension'] if results else 'N/A'}")
    print(f"Results saved to: {output_file}")
    

    landmarks = {}
    for result in results:
        landmark = result['landmark']
        if landmark not in landmarks:
            landmarks[landmark] = 0
        landmarks[landmark] += 1
    
    print(f"\nLandmarks processed:")
    for landmark, count in sorted(landmarks.items()):
        print(f"  {landmark}: {count} images")
    
    return results

def main():
    images_folder = "istanbul_landmarks_images_extended"
    output_file = "istanbul_landmarks_clip_features.json"
    
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' not found!")
        return
    
    print("CLIP Feature Extraction for Istanbul Landmarks")
    print("=" * 60)
    

    results = extract_features_from_folder(images_folder, output_file)
    
    if results:
        print(f"\n Feature extraction completed!")
        print(f"Images folder: {images_folder}")
        print(f"Features saved: {output_file}")
    else:
        print(f"\n No features extracted!")

if __name__ == "__main__":
    main()