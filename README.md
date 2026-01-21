# VLM YOLO Detector

VLM-powered image extraction and semantic search for equipment manuals. This repository provides the image data pipeline for the agentic-rag system.

## Overview

This tool processes PDF manuals to:

1. **Extract images** from PDF documents (embedded images + rendered pages with diagrams)
2. **Generate VLM descriptions** for each image using LLaVA via Ollama
3. **Create semantic embeddings** for intelligent image search
4. **Integrate with agentic-rag** for visual content retrieval

## Data Pipeline Status

| Step | Output |
|------|--------|
| PDF Extraction | 2,827 images from 30 PDFs |
| VLM Descriptions | Contextual descriptions in image_index.json |
| Semantic Embeddings | 384-dim embeddings in image_embeddings.npy |

## Getting Started

### Prerequisites

- **Python 3.10+** with pip
- **Ollama** installed and running (download from ollama.com/download)
- **LLaVA model** for VLM descriptions (pulled automatically by install script)

### Quick Setup (Windows)

For a fully automated installation:

```bash
git clone https://github.com/morkev/vlm-yolo-detector.git
cd vlm-yolo-detector
install.bat
```

This will:
1. Install uv package manager and Python dependencies
2. Install PyMuPDF and sentence-transformers
3. Pull the LLaVA 13B model for VLM descriptions
4. Optionally process any PDFs in data/manuals/

### Manual Setup Steps

#### 1. Clone and Navigate

```bash
git clone https://github.com/morkev/vlm-yolo-detector.git
cd vlm-yolo-detector
```

#### 2. Install Dependencies

```bash
pip install uv
uv sync

# Or using pip directly
pip install -r requirements.txt
pip install pymupdf sentence-transformers
```

#### 3. Pull Ollama VLM Model

```bash
ollama pull llava:13b
```

#### 4. Add PDF Manuals

Place your PDF files in the `data/manuals/` directory:

```bash
cp /path/to/your/manuals/*.pdf data/manuals/
```

#### 5. Run the Processing Pipeline

```bash
# Step 1: Extract images from PDFs
python scripts/extract_all_images.py --pdf-dir data/manuals --output-dir data/processed --min-size 150

# Step 2: Generate VLM descriptions (requires Ollama with llava:13b)
python scripts/describe_images_vlm.py --index data/processed/image_index.json

# Step 3: Build semantic embeddings
python scripts/build_embeddings.py
```

## Directory Structure

```
vlm-yolo-detector/
├── data/
│   ├── manuals/                    # Source PDF files
│   └── processed/
│       ├── images/                 # Extracted images (train/val split)
│       ├── labels/                 # YOLO format labels
│       ├── image_index.json        # Image metadata + VLM descriptions
│       ├── image_embeddings.npy    # Semantic embeddings (384-dim)
│       └── embedding_mapping.json  # Filename to index mapping
├── scripts/
│   ├── extract_all_images.py       # PDF to images
│   ├── describe_images_vlm.py      # Generate VLM descriptions
│   ├── build_embeddings.py         # Create semantic embeddings
│   ├── auto_label_images.py        # YOLO auto-labeling
│   └── train_classifier.py         # Train YOLO classifier
├── runs/                           # Trained model weights
├── configs/                        # YOLO configuration files
├── yologen/                        # Python package
├── install.bat                     # Automated installation script
├── requirements.txt                # Python dependencies
└── pyproject.toml                  # Project configuration
```

## Integration with Agentic RAG

The agentic-rag repository uses this data for semantic image search:

```python
# agentic-rag/app/backend/api/tools/image_search.py
YOLOGEN_DIR = Path("..") / "vlm-yolo-detector"
IMAGE_INDEX_PATH = YOLOGEN_DIR / "data" / "processed" / "image_index.json"
EMBEDDING_NPY_PATH = YOLOGEN_DIR / "data" / "processed" / "image_embeddings.npy"
```

**Setup for integration:**

1. Clone this repository alongside agentic-rag in the same parent directory
2. Run `install.bat` to process PDFs and generate embeddings
3. The agentic-rag system will automatically find the embeddings

Example directory structure:
```
Repositories/
├── agentic-rag/
└── vlm-yolo-detector/
```

## Processing Scripts

### extract_all_images.py

Extracts all visual content from PDFs including embedded images, vector graphics rendered as images, and full pages with diagrams.

```bash
python scripts/extract_all_images.py --pdf-dir data/manuals --output-dir data/processed --min-size 150
```

Options:
- `--pdf-dir`: Directory containing PDF files
- `--output-dir`: Output directory for extracted images
- `--min-size`: Minimum image dimension in pixels (default: 150)
- `--render-all`: Render all pages as images

### describe_images_vlm.py

Uses LLaVA via Ollama to generate contextual descriptions for each image.

```bash
python scripts/describe_images_vlm.py --index data/processed/image_index.json
```

Options:
- `--index`: Path to image_index.json
- `--batch-size`: Number of images to process in parallel (default: 5)
- `--model`: Ollama VLM model to use (default: llava:13b)

### build_embeddings.py

Creates semantic embeddings from VLM descriptions using sentence-transformers.

```bash
python scripts/build_embeddings.py
```

Options:
- `--model`: Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)

## Output Files

### image_index.json

Contains metadata for each extracted image:

```json
{
  "filename": "APSX-PIM_page_5_img_1.png",
  "pdf_source": "APSX-PIM-Manual.pdf",
  "pdf_name": "APSX-PIM-Manual",
  "page": 5,
  "width": 800,
  "height": 600,
  "extraction_type": "embedded",
  "page_text": "...",
  "vlm_description": "This image shows the control panel of the APSX-PIM injection molding machine..."
}
```

### image_embeddings.npy

NumPy array of 384-dimensional embeddings for each image, indexed by filename in embedding_mapping.json.

### embedding_mapping.json

Maps image filenames to their index in the embeddings array:

```json
{
  "APSX-PIM_page_5_img_1.png": 0,
  "APSX-PIM_page_6_img_2.png": 1,
  ...
}
```

## Troubleshooting

### Ollama Connection Error

**Problem**: Failed to connect to Ollama

**Solution**: 
1. Verify Ollama is running: `ollama list`
2. Start Ollama: `ollama serve`
3. Pull the LLaVA model: `ollama pull llava:13b`

### No Images Extracted

**Problem**: extract_all_images.py finds no images

**Solutions**:
1. Verify PDFs are in `data/manuals/`
2. Try with `--render-all` flag to render pages as images
3. Lower the `--min-size` threshold

### Embedding Generation Fails

**Problem**: build_embeddings.py fails

**Solutions**:
1. Verify sentence-transformers is installed: `pip install sentence-transformers`
2. Check that image_index.json exists and has VLM descriptions
3. Ensure sufficient disk space for embeddings

### Missing VLM Descriptions

**Problem**: image_index.json has empty vlm_description fields

**Solutions**:
1. Verify Ollama is running and LLaVA model is pulled
2. Re-run describe_images_vlm.py
3. Check Ollama logs for errors

## Requirements

### Core Dependencies

- Python 3.10+
- PyMuPDF (for PDF processing)
- sentence-transformers (for embeddings)
- Pillow (for image processing)
- NumPy

### Optional Dependencies (for YOLO training)

- ultralytics
- torch
- torchvision

### VLM Requirements

- Ollama with LLaVA model

## Related Repositories

- **agentic-rag**: Main RAG pipeline that uses this repository for image search
  - Repository: https://github.com/Manufacturing-Demonstration-Facility/agentic-rag
  - Uses the embeddings from this repository for semantic image retrieval

## License

MIT License
