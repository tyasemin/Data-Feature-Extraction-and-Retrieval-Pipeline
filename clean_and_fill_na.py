import csv
import os

def clean_cell_value(value):
    """
    Clean cell value by:
    1. Removing cells that contain 'Konum Yok' or 'Konum İşaretlenmedi'
    2. Returning 'NA' for empty cells
    """
    if not value or value.strip() == '':
        return 'NA'
    
    # Check if the cell contains the patterns to remove
    if 'Konum Yok' in value or 'Konum İşaretlenmedi' in value:
        return 'NA'
    
    return value

def clean_and_fill_csv(input_csv='label_processed_2.csv', output_csv='label_cleaned.csv'):
    """
    Clean the CSV file by removing location markers and filling empty cells with 'NA'
    """
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return
    
    total_rows = 0
    total_cells_cleaned = 0
    total_cells_filled = 0
    rows_affected = 0
    
    print(f"Reading {input_csv}...")
    print(f"Removing 'Konum Yok' and 'Konum İşaretlenmedi' from cells...")
    print(f"Filling empty cells with 'NA'...")

    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
 
        header = next(reader)
        writer.writerow(header)
        print(f"Columns: {len(header)}")
        print()
        
        for row_num, row in enumerate(reader, start=2):
            total_rows += 1
            
            if not row:
                continue
            
            cells_changed_in_row = 0
            original_row = row.copy()
            
            for i in range(len(row)):
                original_value = row[i]
                cleaned_value = clean_cell_value(original_value)
                
                if original_value != cleaned_value:
                    row[i] = cleaned_value
                    cells_changed_in_row += 1
                    
                    # Count different types of changes
                    if original_value.strip() == '':
                        total_cells_filled += 1
                    else:
                        total_cells_cleaned += 1
            
            if cells_changed_in_row > 0:
                rows_affected += 1
                
                if rows_affected <= 5:
                    print(f"✓ Row {row_num}: {cells_changed_in_row} cell(s) changed")
                    for i in range(len(row)):
                        if original_row[i] != row[i]:
                            print(f"  Column {i} ({header[i] if i < len(header) else 'N/A'}):")
                            print(f"    '{original_row[i][:50]}...' → '{row[i]}'")
                    print()
            

            writer.writerow(row)
            

            if total_rows % 1000 == 0:
                print(f"Processed {total_rows} rows... ({rows_affected} affected, {total_cells_cleaned} cleaned, {total_cells_filled} filled)")
    
    print(f"Total rows processed: {total_rows}")
    print(f"Rows affected: {rows_affected}")
    print(f"Cells with 'Konum Yok'/'Konum İşaretlenmedi' removed: {total_cells_cleaned}")
    print(f"Empty cells filled with 'NA': {total_cells_filled}")
    print(f"Total cells changed: {total_cells_cleaned + total_cells_filled}")
    print(f"Cleaned CSV saved to: {output_csv}")

def main():
    """Main function to run the cleaning process."""

    input_csv = 'label_processed_2.csv'
    output_csv = 'label_cleaned.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        return
    
    clean_and_fill_csv(input_csv, output_csv)
    
    print("\nDone! The cleaned file has been created.")

if __name__ == "__main__":
    main()
