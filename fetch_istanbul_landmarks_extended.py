import requests
import json
import os
import time
from urllib.parse import quote


os.makedirs("istanbul_landmarks_images_extended", exist_ok=True)


landmarks = [
    "Sultan Ahmed Mosque",  # Blue Mosque
    "Galata Tower", 
    "Maiden's Tower",       # Kız Kulesi
    "Hagia Sophia",
    "Topkapi Palace",
    "Basilica Cistern",
    "Dolmabahçe Palace",
    "Beylerbeyi Palace",
    "Rumeli Fortress",
    "Bosphorus Bridge"
]

headers = {
    'User-Agent': 'Istanbul Landmarks Fetcher/1.0 (Educational project; contact: user@example.com)'
}

def search_wikidata(title):
    
    search_terms = [title + ' Istanbul', title]
    
    for search_term in search_terms:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(search_term)}&language=en&format=json&limit=50&origin=*"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            continue
        
        try:
            data = response.json()
        except Exception as e:
            continue
        
        if not data.get('search'):
            continue
        
        
        for item in data['search']:
            description = item.get('description', '').lower()
            if 'istanbul' in description or 'turkey' in description or 'ottoman' in description:
                return item
        
        
        if data['search']:
            return data['search'][0]
    
    return None

def get_entity_data(qid):
   
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    
    response = requests.get(url, headers=headers)
    entity_data = response.json()
    entity = entity_data['entities'][qid]
    
    return entity

def get_commons_category_images(category_name, limit=10):
   
    if not category_name:
        return []
    
    
    if not category_name.startswith('Category:'):
        category_name = 'Category:' + category_name
    
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        'action': 'query',
        'generator': 'categorymembers',
        'gcmtitle': category_name,
        'gcmtype': 'file',
        'gcmlimit': limit,
        'prop': 'imageinfo',
        'iiprop': 'url|size|mime',
        'format': 'json',
        'origin': '*'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        images = []
        if 'query' in data and 'pages' in data['query']:
            for page_id, page_data in data['query']['pages'].items():
                if 'imageinfo' in page_data:
                    image_info = page_data['imageinfo'][0]
                   
                    if image_info.get('mime', '').startswith('image/'):
                        images.append({
                            'filename': page_data['title'].replace('File:', ''),
                            'url': image_info.get('url'),
                            'size': image_info.get('size', 0)
                        })
        
        return images[:limit] 
    except Exception as e:
        print(f"    Error fetching category images: {e}")
        return []

def search_commons_images(search_term, limit=5):
    
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': f'filetype:bitmap|drawing {search_term}',
        'gsrlimit': limit,
        'prop': 'imageinfo',
        'iiprop': 'url|size|mime',
        'format': 'json',
        'origin': '*'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        images = []
        if 'query' in data and 'pages' in data['query']:
            for page_id, page_data in data['query']['pages'].items():
                if 'imageinfo' in page_data:
                    image_info = page_data['imageinfo'][0]
                    if image_info.get('mime', '').startswith('image/'):
                        images.append({
                            'filename': page_data['title'].replace('File:', ''),
                            'url': image_info.get('url'),
                            'size': image_info.get('size', 0)
                        })
        
        return images
    except Exception as e:
        print(f"    Error searching Commons: {e}")
        return []

def download_image_from_url(image_url, filename, landmark_name, index):
    
    try:
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            
            safe_name = "".join(c for c in landmark_name if c.isalnum() or c in (' ', '-', '_'))
            safe_name = safe_name.replace(' ', '_')
            
            
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type or filename.lower().endswith('.jpg'):
                ext = '.jpg'
            elif 'png' in content_type or filename.lower().endswith('.png'):
                ext = '.png'
            elif filename.lower().endswith('.gif'):
                ext = '.gif'
            else:
                ext = '.jpg'
            
            filepath = f"istanbul_landmarks_images_extended/{safe_name}_{index}{ext}"
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"     Downloaded: {filepath}")
            return filepath
        
        return None
    except Exception as e:
        print(f"     Failed to download {image_url}: {e}")
        return None

def download_image(image_filename, landmark_name, index):
    
    if not image_filename:
        return None
        
    image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(image_filename)}"
    return download_image_from_url(image_url, image_filename, landmark_name, index)


results = []

print(" Fetching Istanbul landmarks images...")


for i, landmark in enumerate(landmarks, 1):
    print(f"\n[{i}/{len(landmarks)}] Processing: {landmark}")
    

    search_result = search_wikidata(landmark)
    if not search_result:
        print(f" No results found for {landmark}")
        continue
    
    qid = search_result['id']
    label = search_result.get('label', landmark)
    description = search_result.get('description', '')
    
    print(f" Found: {label} ({qid})")
    if description:
        print(f"    Description: {description}")
    
    all_images = []
    downloaded_paths = []
    
    try:
        entity = get_entity_data(qid)
        claims = entity.get('claims', {})
        
        # 1. Get main images (P18 property)
        main_images = []
        if 'P18' in claims:
            for image_claim in claims['P18']:
                try:
                    image_value = image_claim['mainsnak']['datavalue']['value']
                    main_images.append(image_value)
                except:
                    continue
        
        if main_images:
            print(f" Found {len(main_images)} main images from Wikidata")
            for idx, img in enumerate(main_images):
                all_images.append({'filename': img, 'source': 'wikidata_main'})
        
        # 2. Get Commons category (P373 property)
        commons_category = None
        if 'P373' in claims:
            try:
                commons_category = claims['P373'][0]['mainsnak']['datavalue']['value']
                print(f" Found Commons category: {commons_category}")
                
                # Get images from the category
                category_images = get_commons_category_images(commons_category, limit=8)
                if category_images:
                    print(f" Found {len(category_images)} images in category")
                    for img in category_images:
                        all_images.append({
                            'filename': img['filename'],
                            'url': img['url'],
                            'source': 'commons_category'
                        })
            except:
                pass
        
        # 3. Search Commons directly for additional images
        search_images = search_commons_images(f"{label} Istanbul", limit=5)
        if search_images:
            print(f" Found {len(search_images)} additional images from Commons search")
            for img in search_images:
                
                if not any(existing['filename'] == img['filename'] for existing in all_images):
                    all_images.append({
                        'filename': img['filename'],
                        'url': img['url'],
                        'source': 'commons_search'
                    })
        
       
        unique_images = []
        seen_filenames = set()
        for img in all_images:
            if img['filename'] not in seen_filenames:
                unique_images.append(img)
                seen_filenames.add(img['filename'])
                if len(unique_images) >= 15: 
                    break
        
        print(f" Total unique images to download: {len(unique_images)}")
        
        
        for idx, img in enumerate(unique_images, 1):
            if 'url' in img:
                path = download_image_from_url(img['url'], img['filename'], label, idx)
            else:
                path = download_image(img['filename'], label, idx)
            
            if path:
                downloaded_paths.append(path)
            
            
            time.sleep(0.5)
        
       
        properties = {}
        if 'P625' in claims:
            coord_claim = claims['P625'][0]
            coord_value = coord_claim['mainsnak']['datavalue']['value']
            properties['coordinates'] = f"{coord_value['latitude']}, {coord_value['longitude']}"
        
        if 'P571' in claims:
            date_claim = claims['P571'][0]
            date_value = date_claim['mainsnak']['datavalue']['value']
            properties['inception_date'] = date_value['time']
        
        
        landmark_data = {
            'qid': qid,
            'label': label,
            'description': description,
            'commons_category': commons_category,
            'total_images_found': len(unique_images),
            'images_downloaded': len(downloaded_paths),
            'image_paths': downloaded_paths,
            'image_details': unique_images,
            'properties': properties,
            'wikidata_url': f"https://www.wikidata.org/wiki/{qid}"
        }
        
        results.append(landmark_data)
        
    except Exception as e:
        print(f" Error processing {landmark}: {e}")


with open('istanbul_landmarks_extended_data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)




total = len(results)
with_images = sum(1 for r in results if r['images_downloaded'] > 0)
total_images = sum(r['images_downloaded'] for r in results)

print(f"Total landmarks processed: {total}")
print(f"Landmarks with images: {with_images}")
print(f"Total images downloaded: {total_images}")
print(f"Images saved to: istanbul_landmarks_images_extended/")
print(f"Data saved to: istanbul_landmarks_extended_data.json")

print(f"\n Extended collection completed!")