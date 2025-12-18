import csv
import os
from pathlib import Path
from urllib.parse import urlparse
import re

def extract_filename_from_url(url):
    """Extract the filename from a URL."""
    if not url or url.strip() == '':
        return None
    
    parsed = urlparse(url)
    path = parsed.path
    
    filename = os.path.basename(path)
    return filename if filename else None

def find_image_in_dataset(filename, dataset_path='./dataset'):
    """
    Search for an image file in the dataset directory structure.
    Returns the relative path if found, None otherwise.
    """
    if not filename:
        return None
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return None
    
    
    for file_path in dataset_path.rglob(filename):
        if file_path.is_file():
            
            return str(file_path)
    
    return None

def update_label_csv(input_csv='label.csv', output_csv='label_updated.csv', dataset_path='./dataset'):
    """
    Update the label.csv file by checking if images exist in dataset
    and updating their paths accordingly.
    """
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return
    
    
    total_rows = 0
    updated_rows = 0
    not_found = 0
    
    
    not_found_urls = []
    
    print(f"Reading {input_csv}...")
    print(f"Searching for images in {dataset_path}...")
    print("=" * 80)
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        
        header = next(reader)
        writer.writerow(header)
        
        
        for row_num, row in enumerate(reader, start=2):  
            total_rows += 1
            
            if not row or len(row) == 0:
                writer.writerow(row)
                continue
            
            
            original_url = row[0] if len(row) > 0 else ''
            filename = extract_filename_from_url(original_url)
            
            if filename:
               
                local_path = find_image_in_dataset(filename, dataset_path)
                
                if local_path:
                    
                    row[0] = local_path
                    updated_rows += 1
                    
                    if updated_rows <= 10:  # Print first 10 updates
                        print(f"✓ Row {row_num}: Updated")
                        print(f"  From: {original_url}")
                        print(f"  To:   {local_path}")
                        print()
                else:
                    not_found += 1
                    if not_found <= 5:  # Store first 5 not found for reporting
                        not_found_urls.append((row_num, filename, original_url))
            
            
            writer.writerow(row)
            
            
            if total_rows % 1000 == 0:
                print(f"Processed {total_rows} rows... ({updated_rows} updated, {not_found} not found)")
    
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rows processed: {total_rows}")
    print(f"Rows updated: {updated_rows}")
    print(f"Images not found in dataset: {not_found}")
    print(f"Success rate: {(updated_rows/total_rows*100):.2f}%")
    print()
    print(f"Updated CSV saved to: {output_csv}")
    
    
    if not_found_urls:
        print("\n" + "=" * 80)
        print("Sample of images NOT FOUND in dataset (first 5):")
        print("=" * 80)
        for row_num, filename, url in not_found_urls:
            print(f"Row {row_num}: {filename}")
            print(f"  URL: {url}")
            print()

def main():
    """Main function to run the update process."""
    print("Label.csv Image Path Updater")
    print("=" * 80)
    
    
    input_csv = 'label.csv'
    output_csv = 'label_updated.csv'
    dataset_path = './dataset'
    
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        return
    
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found!")
        return
    
    
    update_label_csv(input_csv, output_csv, dataset_path)
    
    print(f"  mv {output_csv} {input_csv}")

if __name__ == "__main__":
    main()
