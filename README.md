# VLM YOLO Detector

VLM-powered image extraction and semantic search for equipment manuals.

## What This Does

1. **Extracts images** from PDF manuals (embedded images + rendered pages with diagrams)
2. **Generates VLM descriptions** for each image using LLaVA via Ollama
3. **Creates semantic embeddings** for intelligent image search
4. **Integrates with agentic-rag** for visual content retrieval

## Data Pipeline Status

| Step | Output |
|------|--------|
| PDF Extraction |  2,827 images from 30 PDFs |
| VLM Descriptions | Contextual descriptions in image_index.json |
| Semantic Embeddings | 384-dim embeddings in image_embeddings.npy |

## Directory Structure

```
vlm-yolo-detector/
├── data/
│   ├── manuals/                    # Source PDF files
│   └── processed/
│       ├── images/                 # Extracted images (train/val split)
│       ├── labels/                 # YOLO format labels
│       ├── image_index.json        # Image metadata + VLM descriptions
│       ├── image_embeddings.npy    # Semantic embeddings
│       └── embedding_mapping.json  # Filename to index mapping
├── scripts/
│   ├── extract_all_images.py       # PDF to images
│   ├── describe_images_vlm.py      # Generate VLM descriptions
│   └── build_embeddings.py         # Create semantic embeddings
├── runs/                           # Trained classifier weights
└── yologen/                        # Python package
```

## Usage in Agentic RAG

The agentic-rag repository uses this data for image search:

```python
# agentic-rag/app/backend/api/tools/image_search.py
YOLOGEN_DIR = _REPOSITORIES_DIR / "vlm-yolo-detector"
IMAGE_INDEX_PATH = YOLOGEN_DIR / "data" / "processed" / "image_index.json"
EMBEDDING_NPY_PATH = YOLOGEN_DIR / "data" / "processed" / "image_embeddings.npy"
```

## Re-running the Pipeline

If you need to add new PDFs:

```bash
# 1. Place PDFs in data/manuals/

# 2. Extract images
python scripts/extract_all_images.py --pdf-dir data/manuals --output-dir data/processed

# 3. Generate VLM descriptions (requires Ollama with llava:13b)
python scripts/describe_images_vlm.py --index data/processed/image_index.json

# 4. Build embeddings
python scripts/build_embeddings.py
```

## Requirements

- Python 3.10+
- Ollama with LLaVA model for VLM descriptions
- sentence-transformers for embeddings
