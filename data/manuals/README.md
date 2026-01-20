# PDF Manuals Directory

## Overview
Place your 30 PDF manual files in this directory.

```
data/
├── manuals/              <-- PUT YOUR 30 PDF FILES HERE
│   ├── manual_1.pdf
│   ├── manual_2.pdf
│   └── ...
├── processed/            <-- Extracted images will go here
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── dataset.yaml          <-- Auto-generated config
```

## How to Process Your PDFs

After placing your PDFs here, run:

```bash
python scripts/extract_pdf_images.py --pdf-dir data/manuals --output-dir data/processed
```

This will:
1. Extract all images from your PDF manuals
2. Organize them into YOLO training format
3. Generate a `dataset.yaml` file for training

## Note
The extracted images will need to be labeled before YOLO training.
Use a labeling tool like:
- [LabelImg](https://github.com/tzutalin/labelImg)
- [CVAT](https://github.com/opencv/cvat)
- [Roboflow](https://roboflow.com/)
