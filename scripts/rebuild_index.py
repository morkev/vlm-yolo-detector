#!/usr/bin/env python3
"""
Rebuild image index with correct PDF name parsing.
"""
import json
from pathlib import Path
from collections import Counter
import re

# Class mapping
CLASSES = {0:'diagram', 1:'component', 2:'warning', 3:'procedure', 
           4:'specification', 5:'table', 6:'schematic', 7:'photo'}

def parse_filename(filename: str) -> dict:
    """
    Parse image filename to extract PDF name and page number.
    
    Expected formats:
    - PDFNAME_pageX_imgY.ext
    - Manual_PDFNAME_pageX_imgY.ext
    """
    stem = Path(filename).stem
    
    # Find the page part using regex
    match = re.search(r'_page(\d+)_img(\d+)$', stem)
    
    if match:
        page_num = int(match.group(1))
        img_num = int(match.group(2))
        
        # Everything before _pageX is the PDF name
        pdf_name = stem[:match.start()]
        
        # Remove "Manual_" prefix if present
        if pdf_name.startswith('Manual_'):
            pdf_name = pdf_name[7:]
        
        return {
            'pdf_name': pdf_name,
            'page': page_num,
            'img_num': img_num
        }
    
    return {
        'pdf_name': 'Unknown',
        'page': 0,
        'img_num': 0
    }


def rebuild_index():
    """Rebuild the image index with correct metadata."""
    image_index = {}
    
    # Process all train and val images
    for split in ['train', 'val']:
        images_dir = Path(f'data/processed/images/{split}')
        labels_dir = Path(f'data/processed/labels/{split}')
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} not found")
            continue
        
        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            
            # Parse filename
            metadata = parse_filename(img_path.name)
            
            # Get label if exists
            label_file = labels_dir / f'{img_path.stem}.txt'
            label = 'photo'  # default
            if label_file.exists():
                content = label_file.read_text().strip()
                if content:
                    class_id = int(content.split()[0])
                    label = CLASSES.get(class_id, 'photo')
            
            image_index[img_path.name] = {
                'pdf_name': metadata['pdf_name'],
                'page': metadata['page'],
                'img_num': metadata['img_num'],
                'label': label,
                'split': split,
                'path': str(img_path.resolve())
            }
    
    return image_index


def main():
    print('=' * 60)
    print('  REBUILDING IMAGE INDEX')
    print('=' * 60)
    
    # Rebuild
    image_index = rebuild_index()
    
    # Save
    index_path = Path('data/processed/image_index.json')
    with open(index_path, 'w') as f:
        json.dump(image_index, f, indent=2)
    
    print(f'\n✓ Created index with {len(image_index)} images')
    
    # Statistics
    pdfs = Counter(v['pdf_name'] for v in image_index.values())
    labels = Counter(v['label'] for v in image_index.values())
    splits = Counter(v['split'] for v in image_index.values())
    
    print(f'\n📊 Statistics:')
    print(f'  Splits: train={splits["train"]}, val={splits["val"]}')
    
    print(f'\n📁 PDFs ({len(pdfs)} total):')
    for pdf, count in pdfs.most_common():
        print(f'  {pdf}: {count} images')
    
    print(f'\n🏷️  Labels:')
    for label, count in labels.most_common():
        pct = count / len(image_index) * 100
        print(f'  {label:15s}: {count:4d} ({pct:5.1f}%)')
    
    # Sample entries
    print(f'\n📋 Sample entries:')
    for name, data in list(image_index.items())[:5]:
        print(f'  {name}:')
        print(f'    PDF: {data["pdf_name"]}, Page: {data["page"]}, Label: {data["label"]}')


if __name__ == '__main__':
    main()
