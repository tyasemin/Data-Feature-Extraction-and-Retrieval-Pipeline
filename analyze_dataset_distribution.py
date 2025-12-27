#!/usr/bin/env python3
"""
Analyze and visualize the distribution of images in label_cleaned.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Common Istanbul location names to extract from filenames
LOCATION_KEYWORDS = [
    'galata', 'eminonu', 'beyoglu', 'sultanahmet', 'taksim', 'besiktas', 
    'kadikoy', 'uskudar', 'fatih', 'sisli', 'sirkeci', 'haydarpasa',
    'dolmabahce', 'ortakoy', 'bebek', 'yenikoy', 'tarabya', 'sariyer',
    'karakoy', 'halic', 'bogaz', 'topkapi', 'eyup', 'balat', 'fener',
    'suleymaniye', 'aksaray', 'laleli', 'kumkapi', 'yenikapi', 'unkapani',
    'kasimpasa', 'bogazici', 'rumelihisari', 'anadoluhisari', 'cengelkoy',
    'kuzguncuk', 'beylerbeyi', 'camlica', 'edirnekapi', 'yedikule',
    'sultantepe', 'vanikoy', 'kandilli', 'cubuklu', 'pasabahce', 'beykoz',
    'tophane', 'findikli', 'kabatas', 'mecidiyekoy', 'levent', 'etiler',
    'bebek', 'arnavutkoy', 'kuruceşme', 'kurucesme', 'istiklal', 'pera',
    'cihangir', 'harbiye', 'nisantasi', 'osmanbey', 'pangalti', 'okmeydani'
]

def extract_location_from_filename(filename):
    """Extract location keywords from filename"""
    if pd.isna(filename):
        return []
    
    filename_lower = filename.lower()
    found_locations = []
    
    for keyword in LOCATION_KEYWORDS:
        if keyword in filename_lower:
            found_locations.append(keyword.capitalize())
    
    return found_locations

# Read CSV
print("Loading label_cleaned.csv...")
df = pd.read_csv('label_cleaned.csv')
print(f"Total images: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Extract year/month from image paths
df['image_path'] = df['﻿"Kapak Görseli"']
df['year'] = df['image_path'].str.extract(r'dataset/(\d{4})/')
df['month'] = df['image_path'].str.extract(r'dataset/\d{4}/(\d{2})/')

# 1. Distribution by Year
print("\n" + "="*80)
print("Distribution by Year")
print("="*80)
year_counts = df['year'].value_counts().sort_index()
print(year_counts)

# 2. Distribution by Month (within years)
print("\n" + "="*80)
print("Distribution by Year-Month")
print("="*80)
df['year_month'] = df['year'] + '-' + df['month']
year_month_counts = df['year_month'].value_counts().sort_index()
print(year_month_counts)

# 3. Distribution by Location (İdarı Bölgeler)
print("\n" + "="*80)
print("Top 20 Locations")
print("="*80)
# Split multiple locations and count
all_locations = []
for locations in df['İdarı Bölgeler'].dropna():
    all_locations.extend([loc.strip() for loc in str(locations).split(',')])
location_counts = Counter(all_locations)
for loc, count in location_counts.most_common(20):
    print(f"{loc}: {count}")

# 4. Distribution by Type (Türler)
print("\n" + "="*80)
print("Distribution by Type")
print("="*80)
type_counts = df['Türler'].value_counts()
print(type_counts)

# 5. Distribution by Date Range
print("\n" + "="*80)
print("Historical Date Range")
print("="*80)
df['Tarih En Erken'] = pd.to_numeric(df['Tarih En Erken'], errors='coerce')
df['Tarih En Geç'] = pd.to_numeric(df['Tarih En Geç'], errors='coerce')
print(f"Earliest date: {df['Tarih En Erken'].min()}")
print(f"Latest date: {df['Tarih En Geç'].max()}")

# Create decade bins
df['decade'] = (df['Tarih En Erken'] // 10 * 10).astype('Int64')
decade_counts = df['decade'].value_counts().sort_index()
print("\nDistribution by Decade:")
print(decade_counts)

# 6. Top Tags
print("\n" + "="*80)
print("Top 30 Tags")
print("="*80)
all_tags = []
for tags in df['Etiketler'].dropna():
    all_tags.extend([tag.strip() for tag in str(tags).split(',')])
tag_counts = Counter(all_tags)
for tag, count in tag_counts.most_common(30):
    print(f"{tag}: {count}")

# 7. Extract locations from filenames
print("\n" + "="*80)
print("Locations Extracted from Filenames")
print("="*80)
filename_locations = []
for path in df['image_path']:
    filename = Path(path).stem  # Get filename without extension
    locations = extract_location_from_filename(filename)
    filename_locations.extend(locations)

filename_location_counts = Counter(filename_locations)
print(f"Total location mentions in filenames: {len(filename_locations)}")
print(f"Unique locations found: {len(filename_location_counts)}")
print("\nTop 30 locations from filenames:")
for loc, count in filename_location_counts.most_common(30):
    print(f"{loc}: {count}")

# Create Visualizations - Save each as separate file
print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

output_dir = Path('distribution_graphs')
output_dir.mkdir(exist_ok=True)

# Plot 1: Images by Year
plt.figure(figsize=(12, 6))
year_counts.plot(kind='bar', color='steelblue')
plt.title('Distribution of Images by Upload Year', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '01_upload_year_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   01_upload_year_distribution.png")

# Plot 2: Images by Year-Month
plt.figure(figsize=(16, 6))
year_month_counts.plot(kind='bar', color='coral')
plt.title('Distribution by Year-Month', fontsize=14, fontweight='bold')
plt.xlabel('Year-Month')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(output_dir / '02_year_month_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   02_year_month_distribution.png")

# Plot 3: Top 15 Locations (from metadata)
plt.figure(figsize=(12, 8))
top_locs = dict(location_counts.most_common(15))
plt.barh(list(top_locs.keys()), list(top_locs.values()), color='lightgreen')
plt.title('Top 15 Locations (from Metadata)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / '03_top_locations_metadata.png', dpi=150, bbox_inches='tight')
plt.close()
print("   03_top_locations_metadata.png")

# Plot 4: Top 20 Locations from Filenames
plt.figure(figsize=(12, 10))
top_filename_locs = dict(filename_location_counts.most_common(20))
if top_filename_locs:
    plt.barh(list(top_filename_locs.keys()), list(top_filename_locs.values()), color='skyblue')
    plt.title('Top 20 Locations (from Filenames)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Images')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / '04_top_locations_filenames.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   04_top_locations_filenames.png")
else:
    plt.text(0.5, 0.5, 'No locations extracted from filenames', 
            ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Top 20 Locations (from Filenames)', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / '04_top_locations_filenames.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   04_top_locations_filenames.png (no data)")

# Plot 5: Image Types (Pie Chart)
plt.figure(figsize=(14, 10))
type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                 textprops={'fontsize': 10}, pctdistance=0.85)
plt.title('Distribution by Image Type (Pie Chart)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('')
plt.legend(type_counts.index, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / '05_image_types_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print("   05_image_types_pie.png")

# Plot 5b: Image Types (Bar Chart)
plt.figure(figsize=(14, 8))
type_counts.plot(kind='barh', color='steelblue', alpha=0.8)
plt.title('Distribution by Image Type (Bar Chart)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images')
plt.ylabel('Image Type')
plt.gca().invert_yaxis()
# Add value labels on bars
for i, v in enumerate(type_counts.values):
    plt.text(v + 50, i, str(v), va='center', fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / '05_image_types_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("   05_image_types_bar.png")

# Plot 6: Historical Decades
plt.figure(figsize=(12, 6))
decade_counts.plot(kind='bar', color='purple', alpha=0.7)
plt.title('Distribution by Historical Decade', fontsize=14, fontweight='bold')
plt.xlabel('Decade')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '06_historical_decades.png', dpi=150, bbox_inches='tight')
plt.close()
print("   06_historical_decades.png")

# Plot 7: Top 20 Tags
plt.figure(figsize=(12, 10))
top_tags = dict(tag_counts.most_common(20))
plt.barh(list(top_tags.keys()), list(top_tags.values()), color='orange')
plt.title('Top 20 Tags', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / '07_top_tags.png', dpi=150, bbox_inches='tight')
plt.close()
print("   07_top_tags.png")

# Plot 8: Geographic Distribution
plt.figure(figsize=(10, 10))
has_coords = df[['Lat', 'Lon']].notna().all(axis=1).value_counts()
labels = ['Has Coordinates', 'No Coordinates']
colors = ['#66b3ff', '#ff9999']
plt.pie(has_coords.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Geographic Data Availability', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '08_geographic_availability.png', dpi=150, bbox_inches='tight')
plt.close()
print("   08_geographic_availability.png")

# Plot 9: License Distribution
plt.figure(figsize=(12, 6))
license_counts = df['Lisans'].fillna('Unknown').value_counts()
license_counts.plot(kind='bar', color='teal')
plt.title('License Distribution', fontsize=14, fontweight='bold')
plt.xlabel('License Type')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '09_license_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   09_license_distribution.png")

# Plot 10: Source Distribution
plt.figure(figsize=(12, 8))
source_counts = df['Kaynaklar'].value_counts().head(15)
source_counts.plot(kind='barh', color='gold')
plt.title('Top 15 Sources', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / '10_top_sources.png', dpi=150, bbox_inches='tight')
plt.close()
print("   10_top_sources.png")

# Plot 11: Photographer Distribution
plt.figure(figsize=(12, 8))
photographer_counts = df['Oluşturanlar'].value_counts().head(15)
photographer_counts.plot(kind='barh', color='pink')
plt.title('Top 15 Photographers', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / '11_top_photographers.png', dpi=150, bbox_inches='tight')
plt.close()
print("   11_top_photographers.png")

# Plot 12: Histogram of Historical Dates
plt.figure(figsize=(12, 6))
df['Tarih En Erken'].dropna().hist(bins=50, color='brown', alpha=0.7, edgecolor='black')
plt.title('Histogram of Historical Dates (Earliest)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(output_dir / '12_historical_dates_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("   12_historical_dates_histogram.png")

# Plot 13: Comparison of location sources
plt.figure(figsize=(10, 6))
location_sources = pd.DataFrame({
    'Metadata Locations': len(location_counts),
    'Filename Locations': len(filename_location_counts),
    'Total Metadata Mentions': sum(location_counts.values()),
    'Total Filename Mentions': sum(filename_location_counts.values())
}, index=[0]).T
location_sources.plot(kind='bar', legend=False, color=['#1f77b4', '#ff7f0e'])
plt.title('Location Data Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '13_location_data_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   13_location_data_comparison.png")

# Plot 14: Summary Statistics
plt.figure(figsize=(12, 10))
plt.axis('off')
summary_text = f"""
DATASET SUMMARY
{'='*40}

Total Images: {len(df):,}

Upload Period:
  - Years: {df['year'].nunique()} unique
  - Months: {df['year_month'].nunique()} unique

Historical Coverage:
  - Earliest: {int(df['Tarih En Erken'].min())}
  - Latest: {int(df['Tarih En Geç'].max())}
  - Span: {int(df['Tarih En Geç'].max() - df['Tarih En Erken'].min())} years

Geographic:
  - With coordinates: {has_coords.get(True, 0):,}
  - Without coordinates: {has_coords.get(False, 0):,}

Location Data:
  - Unique metadata locations: {len(location_counts)}
  - Unique filename locations: {len(filename_location_counts)}
  - Total filename mentions: {sum(filename_location_counts.values())}

Content:
  - Unique tags: {len(tag_counts)}
  - Image types: {len(type_counts)}
  - Sources: {df['Kaynaklar'].nunique()}
  - Photographers: {df['Oluşturanlar'].nunique()}
"""
plt.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top', 
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(output_dir / '14_summary_statistics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   14_summary_statistics.png")

print(f"\n{'='*80}")
print(f" All visualizations saved to: {output_dir}/")
print(f"{'='*80}")
print("\n✓ Analysis complete!")

