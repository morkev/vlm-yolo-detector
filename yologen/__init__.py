"""
YoloGen - Image Classification and Search for Equipment Manuals

A framework for extracting, classifying, and searching images from
equipment manuals. Integrates with RAG systems via MCP.

Features:
- PDF image extraction
- Auto-labeling with VLMs (LLaVA via Ollama)
- YOLO classification training
- Image search API for RAG integration

Example:
    # Search for images
    from yologen import create_api
    api = create_api()
    results = api.search("hydraulic diagram", top_k=5)
    
    # Check if query needs images
    from yologen import detect_image_intent
    needs_image = detect_image_intent("Show me the pump diagram")

MIT License
"""

__version__ = "0.1.0"
__author__ = "YoloGen Contributors"

# API exports
from .api import (
    ImageSearchAPI,
    ImageSearchResult,
    ClassificationResult,
    create_api,
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
)


def detect_image_intent(query: str) -> bool:
    """Quick utility to check if query needs images."""
    from .api import IMAGE_INTENT_KEYWORDS
    query_lower = query.lower()
    return any(kw in query_lower for kw in IMAGE_INTENT_KEYWORDS)


__all__ = [
    "ImageSearchAPI",
    "ImageSearchResult", 
    "ClassificationResult",
    "create_api",
    "detect_image_intent",
    "CLASS_NAMES",
    "CLASS_DESCRIPTIONS",
]
__license__ = "MIT"

from yologen.core.trainer import YOLOTrainer, train_yolo
from yologen.core.vlm_trainer import VLMTrainer, train_vlm
from yologen.core.predictor import YOLOPredictor, VLMPredictor, UnifiedPredictor, predict
from yologen.data.vlm_dataset import VLMDatasetGenerator, generate_vlm_dataset
from yologen.rag import YoloGenRAGTool, ImageSemanticSearch, create_rag_tool

__all__ = [
    # Version
    "__version__",
    # Trainers
    "YOLOTrainer",
    "VLMTrainer",
    "train_yolo",
    "train_vlm",
    # Predictors
    "YOLOPredictor",
    "VLMPredictor",
    "UnifiedPredictor",
    "predict",
    # Data
    "VLMDatasetGenerator",
    "generate_vlm_dataset",
    # RAG Integration
    "YoloGenRAGTool",
    "ImageSemanticSearch",
    "create_rag_tool",
]
