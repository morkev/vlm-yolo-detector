#!/usr/bin/env python3
"""
Auto-Label Images using Ollama VLM (LLaVA)

Automatically classifies extracted images into categories using vision models.
This eliminates the need for manual labeling tools like LabelImg/CVAT.

Usage:
    python scripts/auto_label_images.py --images-dir data/processed/images/train --output-dir data/processed/labels/train
    
    # Use specific model
    python scripts/auto_label_images.py --images-dir data/processed/images/train --model "hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yologen.models.vlm.ollama import OllamaVLM


# Class definitions for equipment manuals
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

CLASS_DESCRIPTIONS = {
    "diagram": "Technical diagrams, flowcharts, block diagrams, system overviews",
    "component": "Individual parts, machine components, equipment pieces",
    "warning": "Safety warnings, caution signs, hazard notices, danger symbols",
    "procedure": "Step-by-step instructions, process flows, how-to illustrations",
    "specification": "Technical specs, measurements, dimensions, parameters",
    "table": "Data tables, specification tables, reference charts",
    "schematic": "Electrical schematics, circuit diagrams, wiring diagrams, hydraulic circuits",
    "photo": "Photographs of actual equipment, real-world images"
}

CLASSIFICATION_PROMPT = """Look at this image from an equipment manual. What type of content is it?

Options:
- diagram (technical diagrams, flowcharts, system overviews)
- component (individual parts, machine components)
- warning (safety warnings, caution signs, hazard notices)
- procedure (step-by-step instructions, process flows)
- specification (technical specs, measurements, dimensions)
- table (data tables, charts, reference values)
- schematic (electrical/circuit diagrams, wiring diagrams)
- photo (photographs of actual equipment)

Answer with just ONE word from the options above:"""


def classify_image(vlm: OllamaVLM, image_path: Path) -> Tuple[str, float]:
    """
    Classify an image using VLM.
    
    Args:
        vlm: OllamaVLM instance
        image_path: Path to image
    
    Returns:
        Tuple of (class_name, confidence)
    """
    try:
        response = vlm.generate(
            image_path=str(image_path),
            question=CLASSIFICATION_PROMPT,
            max_tokens=20,
            temperature=0.1  # Low temperature for consistent classification
        )
        
        # Parse response - extract the class name
        response_lower = response.lower().strip()
        
        # Map common variations
        class_map = {
            "1": "diagram", "2": "component", "3": "warning", "4": "procedure",
            "5": "specification", "6": "table", "7": "schematic", "8": "photo",
            "spec": "specification", "specs": "specification",
            "chart": "table", "graph": "diagram", "circuit": "schematic",
            "electrical": "schematic", "wiring": "schematic",
            "photograph": "photo", "image": "photo", "picture": "photo",
            "part": "component", "parts": "component",
            "caution": "warning", "danger": "warning", "safety": "warning",
            "step": "procedure", "steps": "procedure", "instruction": "procedure",
        }
        
        # Check for direct class match
        for class_name in CLASS_DESCRIPTIONS.keys():
            if class_name in response_lower:
                return class_name, 0.9
        
        # Check mapped variations
        for key, class_name in class_map.items():
            if key in response_lower:
                return class_name, 0.8
        
        # Default to photo if unclear
        return "photo", 0.5
        
    except Exception as e:
        print(f"  Warning: Classification failed for {image_path.name}: {e}")
        return "photo", 0.3


def create_yolo_label(
    class_id: int,
    image_width: int,
    image_height: int,
    coverage: float = 0.8
) -> str:
    """
    Create YOLO format label for full-image classification.
    
    For manual images, we often want to classify the entire image
    rather than detect specific objects. This creates a large bbox
    covering most of the image.
    
    Args:
        class_id: Class index
        image_width: Image width in pixels
        image_height: Image height in pixels  
        coverage: How much of the image the bbox should cover (0-1)
    
    Returns:
        YOLO format label string: "class_id x_center y_center width height"
    """
    # Create bbox covering center portion of image
    margin = (1 - coverage) / 2
    x_center = 0.5
    y_center = 0.5
    width = coverage
    height = coverage
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions without loading full image."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return 640, 480  # Default
    return img.shape[1], img.shape[0]


def auto_label_images(
    images_dir: Path,
    labels_dir: Path,
    vlm: OllamaVLM,
    batch_size: int = 10
) -> Dict[str, int]:
    """
    Auto-label all images in a directory.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory to save labels
        vlm: OllamaVLM instance
        batch_size: Images to process before saving stats
    
    Returns:
        Dictionary of class counts
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(images)} images to label")
    
    class_counts = {name: 0 for name in CLASS_DESCRIPTIONS.keys()}
    class_to_id = {name: i for i, name in CLASSES.items()}
    
    for img_path in tqdm(images, desc="Auto-labeling"):
        # Skip if label already exists
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            continue
        
        # Classify image
        class_name, confidence = classify_image(vlm, img_path)
        class_id = class_to_id.get(class_name, 7)  # Default to photo (7)
        class_counts[class_name] += 1
        
        # Get image dimensions
        width, height = get_image_dimensions(img_path)
        
        # Create YOLO label
        label_content = create_yolo_label(class_id, width, height)
        
        # Save label
        with open(label_path, "w") as f:
            f.write(label_content + "\n")
    
    return class_counts


def main():
    parser = argparse.ArgumentParser(description="Auto-label images using VLM")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", type=str, help="Output directory for labels (default: parallel to images)")
    parser.add_argument("--model", type=str, default="hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M", 
                        help="Ollama model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama API URL")
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    
    # Default labels directory
    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
    else:
        # Assume structure: images/train -> labels/train
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    
    print("=" * 60)
    print("  Auto-Label Images with VLM")
    print("=" * 60)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Model: {args.model}")
    print()
    
    # Initialize VLM
    print("Initializing VLM...")
    vlm = OllamaVLM(model=args.model, base_url=args.base_url)
    
    # Auto-label
    class_counts = auto_label_images(images_dir, labels_dir, vlm)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Labeling Complete")
    print("=" * 60)
    print("\nClass distribution:")
    total = sum(class_counts.values())
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {class_name:15s}: {count:5d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':15s}: {total:5d}")


if __name__ == "__main__":
    main()
