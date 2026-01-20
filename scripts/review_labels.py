#!/usr/bin/env python3
"""
Review Labels - Visual inspection of auto-labeled images

Displays sample images with their assigned labels for verification.
"""

import sys
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# Classes
CLASSES = {
    0: "diagram",
    1: "component", 
    2: "warning",
    3: "procedure",
    4: "specification",
    5: "table",
    6: "schematic",
    7: "photo"
}

def get_label_for_image(image_path: Path, labels_dir: Path) -> tuple:
    """Get the label for an image."""
    label_file = labels_dir / f"{image_path.stem}.txt"
    if label_file.exists():
        content = label_file.read_text().strip()
        if content:
            class_id = int(content.split()[0])
            return class_id, CLASSES.get(class_id, "unknown")
    return -1, "no_label"

def display_samples(images_dir: Path, labels_dir: Path, samples_per_class: int = 2):
    """Display sample images for each class."""
    
    # Group images by class
    class_images = {name: [] for name in CLASSES.values()}
    
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            class_id, class_name = get_label_for_image(img_path, labels_dir)
            if class_name in class_images:
                class_images[class_name].append(img_path)
    
    # Create figure
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(4*samples_per_class, 3*n_classes))
    fig.suptitle('Sample Images by Class - Verify Labels', fontsize=14, fontweight='bold')
    
    for row, (class_name, images) in enumerate(class_images.items()):
        # Sample images for this class
        samples = random.sample(images, min(samples_per_class, len(images))) if images else []
        
        for col in range(samples_per_class):
            ax = axes[row, col] if samples_per_class > 1 else axes[row]
            
            if col < len(samples):
                img_path = samples[col]
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.set_title(f"{class_name}\n{img_path.name[:30]}...", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Load Error", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, f"No more {class_name}", ha='center', va='center')
                ax.set_facecolor('#f0f0f0')
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('label_review.png', dpi=150, bbox_inches='tight')
    print(f"Saved: label_review.png")
    plt.show()

def print_samples_text(images_dir: Path, labels_dir: Path, samples_per_class: int = 3):
    """Print text-based sample review."""
    
    # Group images by class
    class_images = {name: [] for name in CLASSES.values()}
    
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            class_id, class_name = get_label_for_image(img_path, labels_dir)
            if class_name in class_images:
                class_images[class_name].append(img_path)
    
    print("=" * 70)
    print("  SAMPLE IMAGES BY CLASS")
    print("=" * 70)
    
    for class_name, images in class_images.items():
        print(f"\n📁 {class_name.upper()} ({len(images)} images)")
        print("-" * 50)
        
        samples = random.sample(images, min(samples_per_class, len(images))) if images else []
        for img_path in samples:
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                print(f"  • {img_path.name} ({w}x{h})")
            else:
                print(f"  • {img_path.name} (could not load)")

def save_sample_grid(images_dir: Path, labels_dir: Path, output_path: str = "label_review.png"):
    """Save a grid of sample images for each class."""
    
    # Group images by class
    class_images = {name: [] for name in CLASSES.values()}
    
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            class_id, class_name = get_label_for_image(img_path, labels_dir)
            if class_name in class_images:
                class_images[class_name].append(img_path)
    
    # Create grid: 8 classes x 3 samples = 24 images
    samples_per_class = 3
    cell_size = 200
    padding = 10
    label_height = 30
    
    grid_width = samples_per_class * (cell_size + padding) + padding
    grid_height = len(CLASSES) * (cell_size + label_height + padding) + padding + 40
    
    # Create blank canvas
    import numpy as np
    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(canvas, "Label Review - Sample Images by Class", (padding, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    y_offset = 50
    
    for class_idx, (class_name, images) in enumerate(class_images.items()):
        # Class label
        cv2.putText(canvas, f"{class_name} ({len(images)})", 
                    (padding, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        
        y_start = y_offset + label_height
        
        # Sample images
        samples = random.sample(images, min(samples_per_class, len(images))) if images else []
        
        for col, img_path in enumerate(samples):
            x_start = padding + col * (cell_size + padding)
            
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize to fit cell
                h, w = img.shape[:2]
                scale = min(cell_size / w, cell_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                
                # Center in cell
                x_off = (cell_size - new_w) // 2
                y_off = (cell_size - new_h) // 2
                
                canvas[y_start + y_off:y_start + y_off + new_h,
                       x_start + x_off:x_start + x_off + new_w] = img_resized
                
                # Draw border
                cv2.rectangle(canvas, (x_start, y_start), 
                             (x_start + cell_size, y_start + cell_size),
                             (200, 200, 200), 1)
        
        y_offset = y_start + cell_size + padding
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Saved review grid: {output_path}")
    return output_path


if __name__ == "__main__":
    images_dir = Path("data/processed/images/train")
    labels_dir = Path("data/processed/labels/train")
    
    print_samples_text(images_dir, labels_dir, samples_per_class=3)
    
    # Save visual grid
    save_sample_grid(images_dir, labels_dir, "label_review.png")
