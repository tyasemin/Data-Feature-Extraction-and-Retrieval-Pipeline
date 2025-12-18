import csv
import os

def filter_label_csv(input_csv='label_updated.csv', output_csv='label_processed.csv'):
    """
    Filter CSV file to include only rows where first column starts with 'dataset/'
    """
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return
    
    # Statistics
    total_rows = 0
    filtered_rows = 0
    excluded_rows = 0
    
    print(f"Reading {input_csv}...")
    print(f"Filtering rows where 'Kapak Görseli' starts with 'dataset/'...")
    print("=" * 80)
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        
        header = next(reader)
        writer.writerow(header)
        print(f"Header: {header[0]}")
        print()
        
        
        for row_num, row in enumerate(reader, start=2):  
            total_rows += 1
            
            if not row or len(row) == 0:
                continue
            
            kapak_gorseli = row[0] if len(row) > 0 else ''
        
            if kapak_gorseli.startswith('dataset/'):
                writer.writerow(row)
                filtered_rows += 1
                
                if filtered_rows <= 10:
                    print(f"✓ Row {row_num}: Included")
                    print(f"  Path: {kapak_gorseli}")
                    if len(row) > 2:  
                        print(f"  Title: {row[2]}")
                    print()
            else:
                excluded_rows += 1
            
            
            if total_rows % 1000 == 0:
                print(f"Processed {total_rows} rows... ({filtered_rows} included, {excluded_rows} excluded)")
    
    
    
    print(f"Total rows processed: {total_rows}")
    print(f"Rows included (starting with 'dataset/'): {filtered_rows}")
    print(f"Rows excluded: {excluded_rows}")
    print()
    print(f"Filtered CSV saved to: {output_csv}")

def main():    
    
    input_csv = 'label_updated.csv'
    output_csv = 'label_processed.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease run update_label_paths.py first to generate label_updated.csv")
        return
    
    filter_label_csv(input_csv, output_csv)
    
if __name__ == "__main__":
    main()
