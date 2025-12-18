import csv
import os
import re

def clean_cell_value(value):
    """
    Clean any cell value by removing it completely if it contains the HTML pattern.
    Returns empty string if the pattern is found, otherwise returns the original value.
    """
    if not value or value.strip() == '':
        return ''
    
    if '<span aria-hidden="true">—</span><span class="screen-reader-text">' in value:
        return ''  
    
    return value 

def clean_label_csv(input_csv='label_processed.csv', output_csv='label_processed_2.csv'):
    """
    Clean the label_processed.csv file by removing noisy HTML data from ALL cells
    """
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return
    
    total_rows = 0
    total_cells_cleaned = 0
    rows_affected = 0
    
    print(f"Reading {input_csv}...")
    print(f"Cleaning all cells containing HTML span tags...")
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        writer.writerow(header)

        for row_num, row in enumerate(reader, start=2):
            total_rows += 1
            
            if not row:
                writer.writerow(row)
                continue
            
            cleaned_row = []
            row_changed = False
            cells_changed_in_row = 0
            
            for col_idx, cell in enumerate(row):
                original_cell = cell
                cleaned_cell = clean_cell_value(cell)
                cleaned_row.append(cleaned_cell)
                
                if original_cell != cleaned_cell:
                    row_changed = True
                    cells_changed_in_row += 1
                    total_cells_cleaned += 1
            
            if row_changed:
                rows_affected += 1
                

                if rows_affected <= 5:
                    print(f"✓ Row {row_num}: {cells_changed_in_row} cell(s) cleaned")
                    for col_idx, (orig, clean) in enumerate(zip(row, cleaned_row)):
                        if orig != clean:
                            print(f"  Column {col_idx} ({header[col_idx] if col_idx < len(header) else 'unknown'}):")
                            print(f"    Original: {orig[:60]}{'...' if len(orig) > 60 else ''}")
                            print(f"    Cleaned:  {clean[:60] if clean else '(empty)'}{'...' if len(clean) > 60 else ''}")
                    print()
            
            writer.writerow(cleaned_row)
            
            if total_rows % 1000 == 0:
                print(f"Processed {total_rows} rows... ({rows_affected} rows affected, {total_cells_cleaned} cells cleaned)")

    print(f"Total rows processed: {total_rows}")
    print(f"Rows affected: {rows_affected}")
    print(f"Total cells cleaned: {total_cells_cleaned}")
    print(f"Cleaned CSV saved to: {output_csv}")

def main():
    """Main function to run the cleaning process."""

    input_csv = 'label_processed.csv'
    output_csv = 'label_ready.csv'
    

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease run filter_label_paths.py first to generate label_processed.csv")
        return
    

    clean_label_csv(input_csv, output_csv)
    
    print("\nDone! The ready file has been created with cleaned cells.")

if __name__ == "__main__":
    main()
