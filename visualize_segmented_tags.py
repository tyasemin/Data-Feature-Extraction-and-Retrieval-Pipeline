#!/usr/bin/env python3
"""
Analyze and visualize the distribution of tags in segmented_tags files
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_segmented_tags(tags_dir: str) -> Tuple[List[Dict], Dict]:
    """
    Load all segmented tags JSON files from directory.
    
    Returns:
        - List of all segment data
        - Statistics dictionary
    """
    tags_path = Path(tags_dir)
    all_segments = []
    all_tags = []
    tag_confidences = {}
    image_count = 0
    total_segments = 0
    
    print(f"Loading segmented tags from: {tags_dir}")
    
    for json_file in tags_path.glob('*_tags.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            image_count += 1
            total_segments += data.get('num_segments', 0)
            
            for segment in data.get('segment_tags', []):
                segment_info = {
                    'filename': data['filename'],
                    'segment_id': segment['segment_id'],
                    'area': segment['area'],
                    'tags': segment['tags']
                }
                all_segments.append(segment_info)
                
                # Collect tag information
                for tag_info in segment['tags']:
                    tag = tag_info['tag']
                    confidence = tag_info['confidence']
                    
                    all_tags.append(tag)
                    
                    if tag not in tag_confidences:
                        tag_confidences[tag] = []
                    tag_confidences[tag].append(confidence)
                    
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue
    
    stats = {
        'image_count': image_count,
        'total_segments': total_segments,
        'unique_tags': len(set(all_tags)),
        'total_tag_mentions': len(all_tags),
        'avg_segments_per_image': total_segments / image_count if image_count > 0 else 0
    }
    
    print(f"\n{'='*80}")
    print("Statistics:")
    print(f"  Images processed: {image_count:,}")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Unique tags: {stats['unique_tags']:,}")
    print(f"  Total tag mentions: {stats['total_tag_mentions']:,}")
    print(f"  Avg segments per image: {stats['avg_segments_per_image']:.2f}")
    print(f"{'='*80}\n")
    
    return all_segments, stats, all_tags, tag_confidences


def visualize_tag_distribution(all_segments: List[Dict], all_tags: List[str], 
                               tag_confidences: Dict, stats: Dict, output_dir: str = 'segmented_tags_analysis'):
    """
    Create comprehensive visualizations of tag distributions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating visualizations in: {output_dir}/")
    
    # Count tags
    tag_counts = Counter(all_tags)
    
    # Calculate average confidences
    tag_avg_confidence = {tag: np.mean(confidences) 
                         for tag, confidences in tag_confidences.items()}
    
    # Plot 1: Top 30 Most Common Tags
    plt.figure(figsize=(14, 10))
    top_tags = dict(tag_counts.most_common(30))
    plt.barh(list(top_tags.keys()), list(top_tags.values()), color='steelblue')
    plt.title('Top 30 Most Common Segment Tags', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.ylabel('Tag', fontsize=12)
    plt.gca().invert_yaxis()
    # Add value labels
    for i, (tag, count) in enumerate(top_tags.items()):
        plt.text(count + 50, i, str(count), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / '01_top_30_tags.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_top_30_tags.png")
    
    # Plot 2: Top 50 Tags (for broader view)
    plt.figure(figsize=(14, 14))
    top_50_tags = dict(tag_counts.most_common(50))
    plt.barh(list(top_50_tags.keys()), list(top_50_tags.values()), color='coral')
    plt.title('Top 50 Most Common Segment Tags', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.ylabel('Tag', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / '02_top_50_tags.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_top_50_tags.png")
    
    # Plot 3: Tag Frequency Distribution
    plt.figure(figsize=(14, 8))
    tag_frequency_values = list(tag_counts.values())
    plt.hist(tag_frequency_values, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Tag Frequencies', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.ylabel('Number of Tags', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / '03_tag_frequency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_tag_frequency_distribution.png")
    
    # Plot 4: Average Confidence by Top 30 Tags
    plt.figure(figsize=(14, 10))
    top_30_tag_names = [tag for tag, _ in tag_counts.most_common(30)]
    top_30_confidences = [tag_avg_confidence[tag] for tag in top_30_tag_names]
    
    plt.barh(top_30_tag_names, top_30_confidences, color='purple', alpha=0.7)
    plt.title('Average Confidence Scores for Top 30 Tags', fontsize=16, fontweight='bold')
    plt.xlabel('Average Confidence', fontsize=12)
    plt.ylabel('Tag', fontsize=12)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    # Add value labels
    for i, conf in enumerate(top_30_confidences):
        plt.text(conf + 0.01, i, f'{conf:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / '04_top_30_tags_confidence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_top_30_tags_confidence.png")
    
    # Plot 5: Confidence Distribution Overall
    plt.figure(figsize=(14, 8))
    all_confidences = [conf for confidences in tag_confidences.values() for conf in confidences]
    plt.hist(all_confidences, bins=100, color='orange', alpha=0.7, edgecolor='black')
    plt.title('Distribution of All Tag Confidence Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_confidences):.3f}')
    plt.axvline(np.median(all_confidences), color='blue', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(all_confidences):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / '05_confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_confidence_distribution.png")
    
    # Plot 6: Segment Area Distribution
    plt.figure(figsize=(14, 8))
    segment_areas = [seg['area'] for seg in all_segments]
    plt.hist(segment_areas, bins=100, color='teal', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Segment Areas', fontsize=16, fontweight='bold')
    plt.xlabel('Segment Area (pixels)', fontsize=12)
    plt.ylabel('Number of Segments', fontsize=12)
    plt.axvline(np.mean(segment_areas), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(segment_areas):.0f}')
    plt.axvline(np.median(segment_areas), color='blue', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(segment_areas):.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / '06_segment_area_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_segment_area_distribution.png")
    
    # Plot 7: Tags per Segment Distribution
    plt.figure(figsize=(14, 8))
    tags_per_segment = [len(seg['tags']) for seg in all_segments]
    unique_counts = Counter(tags_per_segment)
    plt.bar(unique_counts.keys(), unique_counts.values(), color='pink', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Tags per Segment', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Tags', fontsize=12)
    plt.ylabel('Number of Segments', fontsize=12)
    plt.xticks(range(min(unique_counts.keys()), max(unique_counts.keys()) + 1))
    plt.tight_layout()
    plt.savefig(output_path / '07_tags_per_segment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_tags_per_segment.png")
    
    # Plot 8: Tag Categories (if we can infer them)
    plt.figure(figsize=(14, 10))
    # Define some broad categories
    categories = {
        'Architecture': ['building', 'wall', 'tower', 'column', 'arch', 'roof', 'dome', 'bridge', 'house'],
        'Nature': ['sky', 'tree', 'water', 'mountain', 'hill', 'cloud', 'grass', 'flower', 'plant'],
        'Objects': ['boat', 'car', 'flag', 'window', 'door', 'furniture', 'lamp', 'bench'],
        'Text/Signs': ['text', 'sign', 'writing', 'inscription'],
        'People': ['person', 'people', 'crowd', 'face'],
        'Art': ['painting', 'sculpture', 'artwork', 'mosaic']
    }
    
    category_counts = {}
    for category, keywords in categories.items():
        count = sum(tag_counts.get(keyword, 0) for keyword in keywords)
        if count > 0:
            category_counts[category] = count
    
    if category_counts:
        plt.bar(category_counts.keys(), category_counts.values(), color='lightgreen', alpha=0.8, edgecolor='black')
        plt.title('Tag Distribution by Category', fontsize=16, fontweight='bold')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Number of Tags', fontsize=12)
        plt.xticks(rotation=45)
        # Add value labels
        for i, (cat, count) in enumerate(category_counts.items()):
            plt.text(i, count + 50, str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / '08_tag_categories.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 08_tag_categories.png")
    
    # Plot 9: Summary Statistics
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    summary_text = f"""
SEGMENTED TAGS ANALYSIS SUMMARY
{'='*50}

Dataset Statistics:
  - Total images: {stats['image_count']:,}
  - Total segments: {stats['total_segments']:,}
  - Avg segments per image: {stats['avg_segments_per_image']:.2f}

Tag Statistics:
  - Unique tags: {stats['unique_tags']:,}
  - Total tag mentions: {stats['total_tag_mentions']:,}
  - Avg tags per segment: {len(all_tags) / len(all_segments):.2f}
  
Top 10 Tags:
{chr(10).join([f"  {i+1}. {tag}: {count:,} occurrences" for i, (tag, count) in enumerate(tag_counts.most_common(10))])}

Confidence Statistics:
  - Mean confidence: {np.mean(all_confidences):.4f}
  - Median confidence: {np.median(all_confidences):.4f}
  - Min confidence: {np.min(all_confidences):.4f}
  - Max confidence: {np.max(all_confidences):.4f}

Segment Area Statistics:
  - Mean area: {np.mean(segment_areas):.0f} pixels
  - Median area: {np.median(segment_areas):.0f} pixels
  - Min area: {np.min(segment_areas):,} pixels
  - Max area: {np.max(segment_areas):,} pixels
"""
    plt.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_path / '09_summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_summary_statistics.png")
    
    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to: {output_path}/")
    print(f"{'='*80}")


def main():
    # Path to segmented tags directory
    tags_dir = 'SAM/segmented_tags'
    
    if not Path(tags_dir).exists():
        print(f"Error: Directory not found: {tags_dir}")
        print("Please provide the correct path to segmented_tags directory")
        return 1
    
    # Load data
    all_segments, stats, all_tags, tag_confidences = load_segmented_tags(tags_dir)
    
    if not all_segments:
        print("No segments found!")
        return 1
    
    # Create visualizations
    visualize_tag_distribution(all_segments, all_tags, tag_confidences, stats)
    
    print("\n✓ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
