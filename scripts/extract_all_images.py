#!/usr/bin/env python3
"""
Comprehensive PDF Image Extractor for YoloGen

Extracts ALL visual content from PDFs including:
1. Embedded raster images (PNG, JPEG, etc.)
2. Vector graphics rendered as images (diagrams, schematics, flowcharts)
3. Full pages that contain significant graphical content

This ensures no diagrams, schematics, or visual content is missed.

Usage:
    python scripts/extract_all_images.py --pdf-dir data/manuals --output-dir data/processed
    python scripts/extract_all_images.py --pdf-dir data/manuals --output-dir data/processed --render-all
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    exit(1)

from PIL import Image
import io
from tqdm import tqdm


@dataclass
class ExtractedImage:
    """Represents an extracted image with metadata."""
    filename: str
    pdf_source: str
    pdf_name: str  # Cleaned PDF name without extension
    page: int
    img_num: int
    width: int
    height: int
    extraction_type: str  # 'embedded', 'rendered', 'drawing'
    page_text: str  # Text content from the same page for context
    has_drawings: bool  # Whether page has vector drawings
    image_hash: str  # For deduplication


def get_page_text(page: fitz.Page, max_chars: int = 2000) -> str:
    """Extract text from a page for context."""
    try:
        text = page.get_text("text")
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars] if len(text) > max_chars else text
    except:
        return ""


def page_has_drawings(page: fitz.Page) -> bool:
    """Check if a page contains vector drawings (diagrams, schematics)."""
    try:
        # Get drawing commands (paths)
        drawings = page.get_drawings()
        if len(drawings) > 10:  # Significant number of vector paths
            return True
        
        # Check for significant graphical content
        # Look at the page's display list for drawing operations
        dl = page.get_displaylist()
        if dl:
            # Pages with many drawing operations likely have diagrams
            return True
    except:
        pass
    return False


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute hash for deduplication."""
    return hashlib.md5(image_bytes).hexdigest()[:12]


def extract_embedded_images(
    doc: fitz.Document,
    pdf_path: Path,
    output_dir: Path,
    min_size: int = 100,
    seen_hashes: set = None
) -> List[ExtractedImage]:
    """Extract embedded raster images from PDF."""
    if seen_hashes is None:
        seen_hashes = set()
    
    extracted = []
    pdf_name = pdf_path.stem
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = get_page_text(page)
        has_drawings = page_has_drawings(page)
        
        image_list = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                    
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                
                # Skip small images (icons, logos, bullets)
                if width < min_size or height < min_size:
                    continue
                
                # Skip very large images that are likely background/watermarks
                if width > 5000 or height > 5000:
                    continue
                
                # Deduplication
                img_hash = compute_image_hash(image_bytes)
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                
                # Save image
                filename = f"{pdf_name}_page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                image_path = output_dir / filename
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                extracted.append(ExtractedImage(
                    filename=filename,
                    pdf_source=pdf_path.name,
                    pdf_name=pdf_name,
                    page=page_num + 1,
                    img_num=img_idx + 1,
                    width=width,
                    height=height,
                    extraction_type="embedded",
                    page_text=page_text,
                    has_drawings=has_drawings,
                    image_hash=img_hash
                ))
                
            except Exception as e:
                print(f"  Warning: Could not extract image {img_idx} from page {page_num + 1}: {e}")
    
    return extracted


def render_pages_with_graphics(
    doc: fitz.Document,
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 150,
    min_drawings: int = 5,
    seen_hashes: set = None,
    extracted_pages: set = None
) -> List[ExtractedImage]:
    """Render pages that contain vector graphics as images."""
    if seen_hashes is None:
        seen_hashes = set()
    if extracted_pages is None:
        extracted_pages = set()
    
    extracted = []
    pdf_name = pdf_path.stem
    
    # Calculate zoom factor for DPI
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Check if page has significant vector content
        try:
            drawings = page.get_drawings()
            num_drawings = len(drawings)
        except:
            num_drawings = 0
        
        # Skip if not enough drawings or already have embedded images from this page
        if num_drawings < min_drawings:
            continue
        
        # Check if we already extracted embedded images from this page
        page_key = (pdf_name, page_num + 1)
        if page_key in extracted_pages:
            continue
        
        page_text = get_page_text(page)
        
        try:
            # Render the page
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pix.tobytes("png")
            
            # Deduplication
            img_hash = compute_image_hash(image_bytes)
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)
            
            # Save rendered page
            filename = f"{pdf_name}_page{page_num + 1}_rendered.png"
            image_path = output_dir / filename
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            extracted.append(ExtractedImage(
                filename=filename,
                pdf_source=pdf_path.name,
                pdf_name=pdf_name,
                page=page_num + 1,
                img_num=0,  # 0 indicates full page render
                width=pix.width,
                height=pix.height,
                extraction_type="rendered",
                page_text=page_text,
                has_drawings=True,
                image_hash=img_hash
            ))
            
        except Exception as e:
            print(f"  Warning: Could not render page {page_num + 1}: {e}")
    
    return extracted


def render_all_pages(
    doc: fitz.Document,
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 150,
    seen_hashes: set = None
) -> List[ExtractedImage]:
    """Render ALL pages as images (for comprehensive extraction)."""
    if seen_hashes is None:
        seen_hashes = set()
    
    extracted = []
    pdf_name = pdf_path.stem
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = get_page_text(page)
        
        try:
            drawings = page.get_drawings()
            has_drawings = len(drawings) > 5
        except:
            has_drawings = False
        
        try:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pix.tobytes("png")
            
            img_hash = compute_image_hash(image_bytes)
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)
            
            filename = f"{pdf_name}_page{page_num + 1}_full.png"
            image_path = output_dir / filename
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            extracted.append(ExtractedImage(
                filename=filename,
                pdf_source=pdf_path.name,
                pdf_name=pdf_name,
                page=page_num + 1,
                img_num=0,
                width=pix.width,
                height=pix.height,
                extraction_type="fullpage",
                page_text=page_text,
                has_drawings=has_drawings,
                image_hash=img_hash
            ))
            
        except Exception as e:
            print(f"  Warning: Could not render page {page_num + 1}: {e}")
    
    return extracted


def extract_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    render_mode: str = "smart",  # 'smart', 'all', 'none'
    min_size: int = 100,
    dpi: int = 150,
    seen_hashes: set = None
) -> List[ExtractedImage]:
    """
    Extract all images from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        render_mode: 'smart' (render pages with vectors), 'all' (render all pages), 'none' (only embedded)
        min_size: Minimum image dimension
        dpi: DPI for page rendering
        seen_hashes: Set of seen image hashes for deduplication
    """
    if seen_hashes is None:
        seen_hashes = set()
    
    all_extracted = []
    extracted_pages = set()  # Track pages with embedded images
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ERROR: Could not open {pdf_path}: {e}")
        return []
    
    # 1. Extract embedded images first
    embedded = extract_embedded_images(doc, pdf_path, output_dir, min_size, seen_hashes)
    all_extracted.extend(embedded)
    
    # Track which pages have embedded images
    for img in embedded:
        extracted_pages.add((img.pdf_name, img.page))
    
    # 2. Render pages based on mode
    if render_mode == "all":
        rendered = render_all_pages(doc, pdf_path, output_dir, dpi, seen_hashes)
        all_extracted.extend(rendered)
    elif render_mode == "smart":
        # Render pages with vector graphics that don't have embedded images
        rendered = render_pages_with_graphics(
            doc, pdf_path, output_dir, dpi, 
            min_drawings=5, 
            seen_hashes=seen_hashes,
            extracted_pages=extracted_pages
        )
        all_extracted.extend(rendered)
    
    doc.close()
    return all_extracted


def process_all_pdfs(
    pdf_dir: Path,
    output_dir: Path,
    render_mode: str = "smart",
    min_size: int = 100,
    dpi: int = 150,
    val_split: float = 0.1
) -> Dict:
    """Process all PDFs in directory."""
    
    # Create output directories
    train_dir = output_dir / "images" / "train"
    val_dir = output_dir / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return {}
    
    print(f"Found {len(pdf_files)} PDF files")
    
    all_images = []
    seen_hashes = set()
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\n  Processing: {pdf_path.name}")
        
        images = extract_from_pdf(
            pdf_path,
            train_dir,  # Extract to train first, move to val later
            render_mode=render_mode,
            min_size=min_size,
            dpi=dpi,
            seen_hashes=seen_hashes
        )
        
        print(f"    Extracted {len(images)} images")
        all_images.extend(images)
    
    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(all_images)
    
    val_count = int(len(all_images) * val_split)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]
    
    # Move val images
    for img in val_images:
        src = train_dir / img.filename
        dst = val_dir / img.filename
        if src.exists():
            src.rename(dst)
    
    # Create index
    index = {}
    for img in train_images:
        index[img.filename] = {
            **asdict(img),
            "split": "train",
            "path": str(train_dir / img.filename),
            "vlm_description": None,  # To be filled by VLM
            "label": None  # To be filled by classifier
        }
    
    for img in val_images:
        index[img.filename] = {
            **asdict(img),
            "split": "val", 
            "path": str(val_dir / img.filename),
            "vlm_description": None,
            "label": None
        }
    
    # Save index
    index_path = output_dir / "image_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"{'='*60}")
    print(f"Total images: {len(index)}")
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")
    print(f"\nBy extraction type:")
    type_counts = defaultdict(int)
    for img in all_images:
        type_counts[img.extraction_type] += 1
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
    
    print(f"\nBy PDF:")
    pdf_counts = defaultdict(int)
    for img in all_images:
        pdf_counts[img.pdf_name] += 1
    for pdf, c in sorted(pdf_counts.items(), key=lambda x: -x[1]):
        print(f"  {pdf}: {c}")
    
    print(f"\nIndex saved to: {index_path}")
    
    return index


def main():
    parser = argparse.ArgumentParser(description="Extract all images from PDFs")
    parser.add_argument("--pdf-dir", type=str, default="data/manuals",
                        help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for extracted images")
    parser.add_argument("--render-mode", type=str, default="smart",
                        choices=["smart", "all", "none"],
                        help="Page rendering mode: smart (render vector pages), all (render all), none (only embedded)")
    parser.add_argument("--min-size", type=int, default=100,
                        help="Minimum image dimension to extract")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for page rendering")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        return
    
    process_all_pdfs(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        render_mode=args.render_mode,
        min_size=args.min_size,
        dpi=args.dpi,
        val_split=args.val_split
    )


if __name__ == "__main__":
    main()
