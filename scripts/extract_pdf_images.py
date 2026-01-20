#!/usr/bin/env python3
"""
PDF Image Extractor for YoloGen

Extracts images from PDF manuals and prepares them for YOLO training.
Also links extracted images to original PDF page references for RAG integration.

Usage:
    python extract_pdf_images.py --pdf-dir data/manuals --output-dir data/processed
    python extract_pdf_images.py --pdf-dir data/manuals --output-dir data/processed --dpi 200
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

import yaml
from tqdm import tqdm


def extract_images_from_pdf_pymupdf(pdf_path: Path, output_dir: Path, min_size: int = 100) -> List[Dict]:
    """
    Extract embedded images from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
        min_size: Minimum image dimension (skip smaller images)
    
    Returns:
        List of dicts with image info: {filename, page, pdf_source, bbox}
    """
    extracted = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Skip small images (likely icons/logos)
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < min_size or height < min_size:
                    continue
                
                # Save image
                filename = f"{pdf_path.stem}_page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                image_path = output_dir / filename
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                extracted.append({
                    "filename": filename,
                    "page": page_num + 1,
                    "pdf_source": pdf_path.name,
                    "width": width,
                    "height": height,
                    "type": "embedded"
                })
            except Exception as e:
                print(f"  Warning: Could not extract image {img_idx} from page {page_num + 1}: {e}")
    
    doc.close()
    return extracted


def render_pdf_pages_as_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> List[Dict]:
    """
    Render entire PDF pages as images using pdf2image.
    Useful when PDFs have diagrams that aren't extractable as embedded images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save rendered images
        dpi: Resolution for rendering
    
    Returns:
        List of dicts with image info
    """
    if not PDF2IMAGE_AVAILABLE:
        print("Warning: pdf2image not available. Install with: pip install pdf2image")
        print("  Also requires poppler: https://github.com/osber/poppler-windows/releases")
        return []
    
    extracted = []
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        for page_num, page_img in enumerate(pages, 1):
            filename = f"{pdf_path.stem}_fullpage{page_num}.png"
            image_path = output_dir / filename
            page_img.save(image_path, "PNG")
            
            extracted.append({
                "filename": filename,
                "page": page_num,
                "pdf_source": pdf_path.name,
                "width": page_img.width,
                "height": page_img.height,
                "type": "fullpage"
            })
    except Exception as e:
        print(f"  Warning: Could not render pages from {pdf_path.name}: {e}")
    
    return extracted


def create_dataset_yaml(output_dir: Path, class_names: List[str] = None):
    """Create dataset.yaml for YOLO training."""
    if class_names is None:
        # Default classes for equipment manual images
        class_names = [
            "diagram", "component", "warning", "procedure",
            "specification", "table", "schematic", "photo"
        ]
    
    dataset_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return yaml_path


def create_image_index(extracted_images: List[Dict], output_path: Path):
    """
    Create JSON index linking images to PDF sources.
    This is used by RAG to find the original PDF context.
    """
    index = {
        "images": extracted_images,
        "by_pdf": {},
        "by_page": {}
    }
    
    for img in extracted_images:
        # Index by PDF
        pdf = img["pdf_source"]
        if pdf not in index["by_pdf"]:
            index["by_pdf"][pdf] = []
        index["by_pdf"][pdf].append(img["filename"])
        
        # Index by PDF:page
        key = f"{pdf}:{img['page']}"
        if key not in index["by_page"]:
            index["by_page"][key] = []
        index["by_page"][key].append(img["filename"])
    
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)
    
    return index


def main():
    parser = argparse.ArgumentParser(description="Extract images from PDF manuals")
    parser.add_argument("--pdf-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for extracted images")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendering full pages (default: 150)")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum image dimension to extract (default: 100)")
    parser.add_argument("--render-pages", action="store_true", help="Also render full pages as images")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    args = parser.parse_args()
    
    if not PYMUPDF_AVAILABLE:
        print("ERROR: PyMuPDF is required. Install with: pip install pymupdf")
        return
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    
    # Create directory structure
    images_dir = output_dir / "images"
    train_images = images_dir / "train"
    val_images = images_dir / "val"
    labels_dir = output_dir / "labels"
    train_labels = labels_dir / "train"
    val_labels = labels_dir / "val"
    
    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 60)
    
    # Extract images from all PDFs
    all_extracted = []
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(exist_ok=True)
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract embedded images
        extracted = extract_images_from_pdf_pymupdf(pdf_path, temp_dir, min_size=args.min_size)
        print(f"  Extracted {len(extracted)} embedded images")
        all_extracted.extend(extracted)
        
        # Optionally render full pages
        if args.render_pages:
            rendered = render_pdf_pages_as_images(pdf_path, temp_dir, dpi=args.dpi)
            print(f"  Rendered {len(rendered)} full pages")
            all_extracted.extend(rendered)
    
    print("\n" + "=" * 60)
    print(f"Total images extracted: {len(all_extracted)}")
    
    # Split into train/val
    import random
    random.shuffle(all_extracted)
    split_idx = int(len(all_extracted) * args.train_split)
    train_set = all_extracted[:split_idx]
    val_set = all_extracted[split_idx:]
    
    print(f"Train set: {len(train_set)} images")
    print(f"Val set: {len(val_set)} images")
    
    # Move images to train/val directories
    for img_info in train_set:
        src = temp_dir / img_info["filename"]
        dst = train_images / img_info["filename"]
        if src.exists():
            shutil.move(str(src), str(dst))
            # Create empty label file (to be annotated later)
            label_path = train_labels / (Path(img_info["filename"]).stem + ".txt")
            label_path.touch()
    
    for img_info in val_set:
        src = temp_dir / img_info["filename"]
        dst = val_images / img_info["filename"]
        if src.exists():
            shutil.move(str(src), str(dst))
            label_path = val_labels / (Path(img_info["filename"]).stem + ".txt")
            label_path.touch()
    
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(output_dir)
    print(f"\nCreated: {yaml_path}")
    
    # Create image index for RAG
    index_path = output_dir / "image_index.json"
    create_image_index(all_extracted, index_path)
    print(f"Created: {index_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"""
1. Label your images using a tool like LabelImg, CVAT, or Roboflow:
   - Images are in: {train_images}
   - Labels go to: {train_labels}

2. Update class names in: {yaml_path}

3. Train YOLO + VLM:
   python train.py --data {output_dir}/dataset.yaml --vlm

4. The image_index.json can be used by your RAG to:
   - Find which PDF a detected image came from
   - Get the exact page number for citations
""")


if __name__ == "__main__":
    main()
