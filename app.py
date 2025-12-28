#!/usr/bin/env python3
"""
Flask microservice for SAM+CLIP image search
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import tempfile

# Add search module to path
sys.path.insert(0, os.path.dirname(__file__))

from search_with_segments import (
    segment_query_image,
    extract_image_features,
    search_whole_image,
    search_segment_level,
    search_hybrid,
    search_by_tags,
    visualize_results
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_results(results):
    """Convert search results to JSON-serializable format."""
    formatted = []
    for result in results:
        item = {
            'filename': result.get('filename'),
            'image_path': result.get('image_path'),
            'baslik': result.get('baslik'),
            'etiketler': result.get('etiketler'),
        }
        
        # Add score information
        if 'similarity_percentage' in result and result['similarity_percentage'] is not None:
            item['similarity_percentage'] = float(result['similarity_percentage'])
        
        if 'hybrid_score' in result:
            item['hybrid_score'] = float(result['hybrid_score'])
            item['whole_image_score'] = float(result['whole_image_score'])
            item['segment_score'] = float(result['segment_score'])
        elif 'similarity_score' in result:
            item['similarity_score'] = float(result['similarity_score'])
        
        # Add segment information if available
        if 'best_matching_segment' in result:
            item['best_matching_segment'] = {
                'segment_id': result['best_matching_segment']['segment_id'],
                'similarity': float(result['best_matching_segment']['similarity'])
            }
        
        formatted.append(item)
    
    return formatted


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'SAM+CLIP Image Search',
        'version': '1.0'
    })


@app.route('/search/whole', methods=['POST'])
def search_whole():
    """
    Search using whole image similarity.
    
    Form data:
        - image: Image file (required)
        - top_k: Number of results (default: 10)
        - tags: Comma-separated tags for filtering (optional)
    """
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get parameters
        top_k = int(request.form.get('top_k', 10))
        tags = request.form.get('tags', '').strip()
        tag_filter = [t.strip() for t in tags.split(',') if t.strip()] if tags else None
        
        # Extract features
        query_features = extract_image_features(filepath)
        if not query_features:
            return jsonify({'error': 'Failed to extract image features'}), 500
        
        # Search
        results = search_whole_image(query_features, top_k=top_k, tag_filter=tag_filter)
        
        return jsonify({
            'mode': 'whole',
            'top_k': top_k,
            'results': format_results(results)
        })
        
    finally:
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/search/segment', methods=['POST'])
def search_segment():
    """
    Search using segment-level similarity.
    
    Form data:
        - image: Image file (required)
        - top_k: Number of results (default: 10)
        - tags: Comma-separated tags for filtering (optional)
        - max_segments: Max segments to extract (default: 10)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        top_k = int(request.form.get('top_k', 10))
        max_segments = int(request.form.get('max_segments', 10))
        tags = request.form.get('tags', '').strip()
        tag_filter = [t.strip() for t in tags.split(',') if t.strip()] if tags else None
        
        # Segment and extract features
        query_segments, _, _ = segment_query_image(filepath, max_segments=max_segments)
        if not query_segments:
            return jsonify({'error': 'Failed to segment image'}), 500
        
        # Search
        results = search_segment_level(query_segments, top_k=top_k, tag_filter=tag_filter)
        
        return jsonify({
            'mode': 'segment',
            'top_k': top_k,
            'num_segments': len(query_segments),
            'results': format_results(results)
        })
        
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/search/hybrid', methods=['POST'])
def search_hybrid_endpoint():
    """
    Hybrid search combining whole image and segment similarity.
    
    Form data:
        - image: Image file (required)
        - top_k: Number of results (default: 10)
        - tags: Comma-separated tags for filtering (optional)
        - max_segments: Max segments to extract (default: 10)
        - whole_weight: Weight for whole image (default: 0.4)
        - segment_weight: Weight for segments (default: 0.6)
        - visualize: Generate visualization (default: false)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        top_k = int(request.form.get('top_k', 10))
        max_segments = int(request.form.get('max_segments', 10))
        whole_weight = float(request.form.get('whole_weight', 0.4))
        segment_weight = float(request.form.get('segment_weight', 0.6))
        visualize = request.form.get('visualize', 'false').lower() == 'true'
        tags = request.form.get('tags', '').strip()
        tag_filter = [t.strip() for t in tags.split(',') if t.strip()] if tags else None
        
        # Extract features
        query_features = extract_image_features(filepath)
        query_segments, _, _ = segment_query_image(filepath, max_segments=max_segments)
        
        if not query_features or not query_segments:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        # Search
        results = search_hybrid(
            query_features,
            query_segments,
            whole_weight=whole_weight,
            segment_weight=segment_weight,
            top_k=top_k,
            tag_filter=tag_filter
        )
        
        response = {
            'mode': 'hybrid',
            'top_k': top_k,
            'num_segments': len(query_segments),
            'whole_weight': whole_weight,
            'segment_weight': segment_weight,
            'results': format_results(results)
        }
        
        # Generate visualization if requested
        if visualize and results:
            query_name = Path(filename).stem
            output_name = f"sam_hybrid_{query_name}.png"
            visualize_results(filepath, results, output_name, workspace_root="/workspace")
            response['visualization'] = output_name
        
        return jsonify(response)
        
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/search/tags', methods=['POST'])
def search_tags_endpoint():
    """
    Search by tags only (no image required).
    
    JSON body or form data:
        - tags: List of tags or comma-separated string (required)
        - top_k: Number of results (default: 10)
    """
    # Try JSON first, then form data
    if request.is_json:
        data = request.get_json()
        tags_input = data.get('tags', [])
        top_k = int(data.get('top_k', 10))
    else:
        tags_input = request.form.get('tags', '')
        top_k = int(request.form.get('top_k', 10))
    
    # Parse tags
    if isinstance(tags_input, str):
        tags = [t.strip() for t in tags_input.split(',') if t.strip()]
    elif isinstance(tags_input, list):
        tags = tags_input
    else:
        return jsonify({'error': 'Invalid tags format'}), 400
    
    if not tags:
        return jsonify({'error': 'No tags provided'}), 400
    
    # Search
    results = search_by_tags(tags, top_k=top_k)
    
    return jsonify({
        'mode': 'tags',
        'tags': tags,
        'top_k': top_k,
        'results': format_results(results)
    })


@app.route('/api/search', methods=['POST'])
def unified_search():
    """
    Unified search endpoint that handles all modes.
    
    Form data:
        - mode: Search mode (whole|segment|hybrid|tags) (required)
        - image: Image file (required for whole/segment/hybrid)
        - tags: Tags for filtering or tag search
        - top_k: Number of results (default: 10)
        - Additional mode-specific parameters
    """
    mode = request.form.get('mode', '').lower()
    
    if mode not in ['whole', 'segment', 'hybrid', 'tags']:
        return jsonify({'error': 'Invalid mode. Use: whole, segment, hybrid, or tags'}), 400
    
    # Route to appropriate handler
    if mode == 'whole':
        return search_whole()
    elif mode == 'segment':
        return search_segment()
    elif mode == 'hybrid':
        return search_hybrid_endpoint()
    elif mode == 'tags':
        return search_tags_endpoint()


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size: 16MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == '__main__':
    print("="*80)
    print("SAM+CLIP Image Search Microservice")
    print("="*80)
    print("Endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /search/whole        - Whole image search")
    print("  POST /search/segment      - Segment-level search")
    print("  POST /search/hybrid       - Hybrid search")
    print("  POST /search/tags         - Tag-based search")
    print("  POST /api/search          - Unified search endpoint")
    print("="*80)
    print("\nStarting server on http://0.0.0.0:5000")
    print("Press CTRL+C to quit\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
