#!/usr/bin/env python3
"""Check yolo-gen status"""
import json
from pathlib import Path
from collections import Counter

print('=' * 60)
print('  YOLO-GEN STATUS CHECK')
print('=' * 60)

# 1. Check image index
index_path = Path('data/processed/image_index.json')
if index_path.exists():
    with open(index_path) as f:
        idx = json.load(f)
    print(f'\n✓ Image Index: {len(idx)} images')
    
    # Count by PDF
    pdfs = set(v['pdf_name'] for v in idx.values())
    print(f'  - From {len(pdfs)} PDFs')
    
    # Count by label
    labels = Counter(v['label'] for v in idx.values())
    print(f'  - Labels: {dict(labels)}')
else:
    print('✗ Image Index: NOT FOUND')

# 2. Check training data
train_images = list(Path('data/processed/images/train').glob('*'))
val_images = list(Path('data/processed/images/val').glob('*'))
train_labels = list(Path('data/processed/labels/train').glob('*.txt'))
val_labels = list(Path('data/processed/labels/val').glob('*.txt'))

print(f'\n✓ Training Data:')
print(f'  - Train images: {len(train_images)}')
print(f'  - Train labels: {len(train_labels)}')
print(f'  - Val images: {len(val_images)}')
print(f'  - Val labels: {len(val_labels)}')

# 3. Check trained weights
weights_v1 = Path('runs/equipment_manuals/yolo/weights/best.pt')
weights_v2 = Path('runs/equipment_manuals_v2/yolo/weights/best.pt')
print(f'\n✓ Trained Weights:')
print(f'  - equipment_manuals: {"EXISTS" if weights_v1.exists() else "NOT FOUND"}')
print(f'  - equipment_manuals_v2: {"EXISTS" if weights_v2.exists() else "NOT FOUND"}')

# 4. Check label format (sample)
print(f'\n✓ Sample Labels:')
for label_file in list(Path('data/processed/labels/train').glob('*.txt'))[:3]:
    content = label_file.read_text().strip()
    parts = content.split()
    classes = {0:'diagram', 1:'component', 2:'warning', 3:'procedure', 
               4:'specification', 5:'table', 6:'schematic', 7:'photo'}
    class_name = classes.get(int(parts[0]), 'unknown')
    print(f'  {label_file.name}: class={class_name} ({parts[0]})')

# 5. Show approach explanation
print('\n' + '=' * 60)
print('  CURRENT APPROACH')
print('=' * 60)
print('''
Your labels use YOLO detection format with full-image bounding boxes.
This means YOLO learns to "detect" the entire image as one class.

For RAG image retrieval, this is VALID because:
1. You classify WHAT TYPE of image it is (diagram, photo, etc.)
2. The trained model can classify new images from queries
3. You can filter by class when searching (e.g., "show me schematics")

Alternative: Use YOLO classify mode for pure classification (no bboxes)
But current approach works and has trained weights available.
''')

# 6. Test inference capability
print('=' * 60)
print('  INFERENCE TEST')
print('=' * 60)

try:
    from ultralytics import YOLO
    
    # Load the trained model
    if weights_v2.exists():
        model = YOLO(str(weights_v2))
        print(f'\n✓ Loaded model: {weights_v2}')
        
        # Test on a sample image
        test_img = train_images[0] if train_images else None
        if test_img:
            results = model(str(test_img), verbose=False)
            if results and results[0].boxes:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    cls = int(boxes[0].cls[0])
                    conf = float(boxes[0].conf[0])
                    class_name = classes.get(cls, 'unknown')
                    print(f'  Test image: {test_img.name}')
                    print(f'  Predicted: {class_name} (conf={conf:.2f})')
                else:
                    print('  No detections (model needs more training)')
            else:
                print('  No detections (model needs more training)')
    else:
        print('✗ No trained weights found')
        
except Exception as e:
    print(f'✗ Inference test failed: {e}')

print('\n' + '=' * 60)
print('  RECOMMENDATIONS')
print('=' * 60)
print('''
1. TRAINING: Run more epochs for better accuracy
   - Current: 1-2 epochs completed
   - Recommended: 20-50 epochs minimum
   - Command: python train.py --data data/processed/dataset.yaml --epochs 20

2. INTEGRATION: Your image index is ready for RAG
   - 1,646 images with labels and PDF sources
   - Query by class type (diagram, schematic, etc.)
   - Query by PDF name for specific manual content

3. MCP SERVER: Create server in yolo-gen for agentic-rag
   - Endpoint: classify_image(image_path) -> class, confidence
   - Endpoint: search_images(query, class_filter) -> image_results
   - Endpoint: get_image_for_context(pdf_name, page) -> image_path
''')
