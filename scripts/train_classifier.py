#!/usr/bin/env python3
"""
Train YOLO Image Classifier for Equipment Manual Images

This script trains a YOLO classification model (not detection) which is 
more appropriate for classifying entire images into categories.

Usage:
    python scripts/train_classifier.py --epochs 20
    python scripts/train_classifier.py --epochs 50 --model yolov8s-cls.pt
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Classes for equipment manuals
CLASSES = ['diagram', 'component', 'warning', 'procedure', 
           'specification', 'table', 'schematic', 'photo']


def prepare_classification_dataset(processed_dir: Path, output_dir: Path):
    """
    Reorganize data from detection format to classification format.
    
    Classification format:
        dataset/
            train/
                diagram/
                    img1.jpg
                component/
                    img2.jpg
            val/
                diagram/
                    img3.jpg
    """
    print("Preparing classification dataset...")
    
    # Create class directories
    for split in ['train', 'val']:
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Class ID mapping
    class_map = {i: name for i, name in enumerate(CLASSES)}
    
    # Process each split
    for split in ['train', 'val']:
        images_dir = processed_dir / 'images' / split
        labels_dir = processed_dir / 'labels' / split
        
        copied = 0
        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            
            # Get label
            label_file = labels_dir / f'{img_path.stem}.txt'
            if label_file.exists():
                content = label_file.read_text().strip()
                if content:
                    class_id = int(content.split()[0])
                    class_name = class_map.get(class_id, 'photo')
                else:
                    class_name = 'photo'
            else:
                class_name = 'photo'
            
            # Copy image to class directory
            dest = output_dir / split / class_name / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            copied += 1
        
        print(f"  {split}: {copied} images copied")
    
    # Count per class
    print("\nClass distribution:")
    for split in ['train', 'val']:
        print(f"  {split}:")
        for cls in CLASSES:
            count = len(list((output_dir / split / cls).glob('*')))
            print(f"    {cls}: {count}")
    
    return output_dir


def train_classifier(data_dir: Path, epochs: int = 20, model: str = "yolov8n-cls.pt", 
                     imgsz: int = 224, batch: int = 32, name: str = None):
    """
    Train YOLO classification model.
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("  YOLO Classification Training")
    print("=" * 60)
    print(f"Dataset: {data_dir}")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    
    # Load model
    yolo = YOLO(model)
    
    # Train
    if name is None:
        name = datetime.now().strftime("cls_%Y%m%d_%H%M%S")
    
    results = yolo.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs",
        name=name,
        patience=10,
        verbose=True,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", 
                        help="Base model (yolov8n-cls.pt, yolov8s-cls.pt, etc.)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--name", type=str, default="equipment_classifier", 
                        help="Experiment name")
    parser.add_argument("--skip-prep", action="store_true", 
                        help="Skip dataset preparation if already done")
    args = parser.parse_args()
    
    # Paths
    processed_dir = Path("data/processed")
    cls_data_dir = Path("data/classification")
    
    # Prepare classification dataset
    if not args.skip_prep or not cls_data_dir.exists():
        prepare_classification_dataset(processed_dir, cls_data_dir)
    else:
        print("Using existing classification dataset")
    
    # Train
    results = train_classifier(
        data_dir=cls_data_dir,
        epochs=args.epochs,
        model=args.model,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"Results saved to: runs/{args.name}")
    print(f"Best weights: runs/{args.name}/weights/best.pt")
    
    return results


if __name__ == "__main__":
    main()
