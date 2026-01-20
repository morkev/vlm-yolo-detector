#!/usr/bin/env python3
"""
YoloGen Image Search API

This module provides a clean API for searching and classifying images
from equipment manuals. Designed to be used by external systems like
agentic-rag via MCP or HTTP.

Key Features:
1. Classify new images using trained YOLO classifier
2. Search indexed images by class, PDF source, or semantic query
3. Determine if a user query needs images (intent detection)

Usage:
    from yologen.api import ImageSearchAPI
    
    api = ImageSearchAPI()
    
    # Check if query needs images
    needs_image = api.detect_image_intent("Show me the pump diagram")
    
    # Search for images
    results = api.search("hydraulic diagram", class_filter="diagram", top_k=5)
    
    # Classify a new image
    classification = api.classify_image("path/to/image.jpg")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np


# Class definitions
CLASS_NAMES = ['diagram', 'component', 'warning', 'procedure', 
               'specification', 'table', 'schematic', 'photo']

CLASS_DESCRIPTIONS = {
    'diagram': 'Technical diagrams, flowcharts, system overviews',
    'component': 'Individual parts, machine components',
    'warning': 'Safety warnings, caution signs, hazard notices',
    'procedure': 'Step-by-step instructions, how-to guides',
    'specification': 'Technical specs, measurements, dimensions',
    'table': 'Data tables, charts, reference values',
    'schematic': 'Electrical/hydraulic schematics, wiring diagrams',
    'photo': 'Photographs of actual equipment'
}

# Keywords that suggest user wants images
IMAGE_INTENT_KEYWORDS = [
    'show', 'image', 'picture', 'photo', 'diagram', 'figure',
    'schematic', 'chart', 'graph', 'drawing', 'illustration',
    'look', 'see', 'visual', 'display', 'view', 'appearance',
    'what does', 'how does it look', 'where is', 'location',
    'exploded view', 'wiring', 'circuit', 'layout'
]


@dataclass
class ImageSearchResult:
    """Result from image search."""
    image_path: str
    image_name: str
    pdf_source: str
    page_number: int
    class_name: str
    score: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class ClassificationResult:
    """Result from image classification."""
    class_name: str
    confidence: float
    all_scores: Dict[str, float]
    
    def to_dict(self) -> dict:
        return asdict(self)


class ImageSearchAPI:
    """
    API for image search and classification.
    
    This class provides the main interface for:
    - Searching indexed images
    - Classifying new images
    - Detecting if queries need images
    """
    
    def __init__(
        self,
        classifier_weights: str = None,
        image_index_path: str = None,
        images_base_dir: str = None,
    ):
        """
        Initialize the API.
        
        Args:
            classifier_weights: Path to trained YOLO classifier weights
            image_index_path: Path to image_index.json
            images_base_dir: Base directory containing images
        """
        self.classifier = None
        self.image_index = {}
        self.images_base_dir = Path(images_base_dir) if images_base_dir else None
        
        # Auto-detect paths if not provided
        self._setup_paths(classifier_weights, image_index_path, images_base_dir)
        
        # Load components
        self._load_classifier()
        self._load_index()
    
    def _setup_paths(self, weights: str, index: str, base_dir: str):
        """Setup default paths based on yolo-gen directory structure."""
        # Find yolo-gen directory
        yologen_dir = Path(__file__).parent.parent
        
        # Classifier weights
        if weights is None:
            default_weights = yologen_dir / "runs" / "equipment_classifier" / "weights" / "best.pt"
            if default_weights.exists():
                self.classifier_weights = str(default_weights)
            else:
                self.classifier_weights = None
        else:
            self.classifier_weights = weights
        
        # Image index
        if index is None:
            default_index = yologen_dir / "data" / "processed" / "image_index.json"
            if default_index.exists():
                self.image_index_path = str(default_index)
            else:
                self.image_index_path = None
        else:
            self.image_index_path = index
        
        # Images directory
        if base_dir is None:
            default_dir = yologen_dir / "data" / "processed" / "images"
            if default_dir.exists():
                self.images_base_dir = default_dir
    
    def _load_classifier(self):
        """Load YOLO classifier model."""
        if self.classifier_weights is None:
            print("Warning: No classifier weights found")
            return
        
        try:
            from ultralytics import YOLO
            self.classifier = YOLO(self.classifier_weights)
            print(f"Loaded classifier: {self.classifier_weights}")
        except ImportError:
            print("Warning: ultralytics not installed")
        except Exception as e:
            print(f"Warning: Failed to load classifier: {e}")
    
    def _load_index(self):
        """Load image index."""
        if self.image_index_path is None:
            print("Warning: No image index found")
            return
        
        try:
            with open(self.image_index_path) as f:
                self.image_index = json.load(f)
            print(f"Loaded index: {len(self.image_index)} images")
        except Exception as e:
            print(f"Warning: Failed to load index: {e}")
    
    def detect_image_intent(self, query: str) -> bool:
        """
        Determine if a query is asking for images/visuals.
        
        Args:
            query: User's question or request
            
        Returns:
            True if the query suggests they want images
        """
        query_lower = query.lower()
        return any(kw in query_lower for kw in IMAGE_INTENT_KEYWORDS)
    
    def classify_image(self, image_path: str) -> Optional[ClassificationResult]:
        """
        Classify an image using the trained model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ClassificationResult with class name and confidence
        """
        if self.classifier is None:
            return None
        
        try:
            results = self.classifier(image_path, verbose=False)
            
            if results and len(results) > 0:
                probs = results[0].probs
                
                # Get top prediction
                top_class = int(probs.top1)
                top_conf = float(probs.top1conf)
                
                # Get all scores
                all_scores = {}
                for i, score in enumerate(probs.data.cpu().numpy()):
                    all_scores[CLASS_NAMES[i]] = float(score)
                
                return ClassificationResult(
                    class_name=CLASS_NAMES[top_class],
                    confidence=top_conf,
                    all_scores=all_scores
                )
        except Exception as e:
            print(f"Classification error: {e}")
        
        return None
    
    def search(
        self,
        query: str = None,
        class_filter: str = None,
        pdf_filter: str = None,
        page_filter: int = None,
        top_k: int = 10,
    ) -> List[ImageSearchResult]:
        """
        Search for images in the index.
        
        Args:
            query: Text query (used for keyword matching)
            class_filter: Filter by class name
            pdf_filter: Filter by PDF name (partial match)
            page_filter: Filter by page number
            top_k: Maximum results to return
            
        Returns:
            List of ImageSearchResult objects
        """
        results = []
        
        for image_name, metadata in self.image_index.items():
            score = 1.0
            
            # Apply class filter
            if class_filter:
                if metadata.get('label', '').lower() != class_filter.lower():
                    continue
            
            # Apply PDF filter
            if pdf_filter:
                if pdf_filter.lower() not in metadata.get('pdf_name', '').lower():
                    continue
            
            # Apply page filter
            if page_filter is not None:
                if metadata.get('page') != page_filter:
                    continue
            
            # Query matching (simple keyword)
            if query:
                query_lower = query.lower()
                pdf_name = metadata.get('pdf_name', '').lower()
                label = metadata.get('label', '').lower()
                
                # Score based on matches
                score = 0.0
                if any(word in pdf_name for word in query_lower.split()):
                    score += 0.5
                if any(word in label for word in query_lower.split()):
                    score += 0.3
                if any(word in query_lower for word in label.split()):
                    score += 0.2
                
                if score == 0:
                    continue
            
            # Build image path
            split = metadata.get('split', 'train')
            if self.images_base_dir:
                image_path = str(self.images_base_dir / split / image_name)
            else:
                image_path = metadata.get('path', image_name)
            
            results.append(ImageSearchResult(
                image_path=image_path,
                image_name=image_name,
                pdf_source=metadata.get('pdf_name', 'Unknown'),
                page_number=metadata.get('page', 0),
                class_name=metadata.get('label', 'photo'),
                score=score
            ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def search_by_class(self, class_name: str, top_k: int = 10) -> List[ImageSearchResult]:
        """Convenience method to search by class."""
        return self.search(class_filter=class_name, top_k=top_k)
    
    def search_by_pdf(self, pdf_name: str, top_k: int = 50) -> List[ImageSearchResult]:
        """Convenience method to search by PDF."""
        return self.search(pdf_filter=pdf_name, top_k=top_k)
    
    def get_image_for_context(
        self, 
        pdf_name: str, 
        page_number: int
    ) -> List[ImageSearchResult]:
        """
        Get images from a specific PDF page.
        Useful for RAG when you have text context and want related images.
        """
        return self.search(pdf_filter=pdf_name, page_filter=page_number, top_k=10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed images."""
        if not self.image_index:
            return {"total": 0}
        
        from collections import Counter
        
        pdfs = Counter(v['pdf_name'] for v in self.image_index.values())
        labels = Counter(v['label'] for v in self.image_index.values())
        
        return {
            "total_images": len(self.image_index),
            "total_pdfs": len(pdfs),
            "pdfs": dict(pdfs),
            "labels": dict(labels),
            "classifier_loaded": self.classifier is not None,
        }
    
    def get_class_info(self) -> Dict[str, str]:
        """Get information about available classes."""
        return CLASS_DESCRIPTIONS.copy()


def create_api(
    yologen_dir: str = None,
    experiment_name: str = "equipment_classifier"
) -> ImageSearchAPI:
    """
    Factory function to create ImageSearchAPI.
    
    Args:
        yologen_dir: Path to yolo-gen directory
        experiment_name: Name of the classifier experiment
        
    Returns:
        Configured ImageSearchAPI instance
    """
    if yologen_dir is None:
        yologen_dir = Path(__file__).parent.parent
    else:
        yologen_dir = Path(yologen_dir)
    
    weights = yologen_dir / "runs" / experiment_name / "weights" / "best.pt"
    index = yologen_dir / "data" / "processed" / "image_index.json"
    images = yologen_dir / "data" / "processed" / "images"
    
    return ImageSearchAPI(
        classifier_weights=str(weights) if weights.exists() else None,
        image_index_path=str(index) if index.exists() else None,
        images_base_dir=str(images) if images.exists() else None,
    )


# Test
if __name__ == "__main__":
    print("Testing ImageSearchAPI...")
    print("=" * 60)
    
    api = create_api()
    
    # Stats
    stats = api.get_stats()
    print(f"\nStats: {stats['total_images']} images from {stats['total_pdfs']} PDFs")
    print(f"Classifier loaded: {stats['classifier_loaded']}")
    
    # Test intent detection
    test_queries = [
        ("Show me the hydraulic pump diagram", True),
        ("What is the operating temperature?", False),
        ("Where is the oil filter located?", True),
        ("List the maintenance intervals", False),
        ("Display the wiring schematic", True),
    ]
    
    print("\nIntent Detection Tests:")
    for query, expected in test_queries:
        result = api.detect_image_intent(query)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{query[:40]}...' -> {result}")
    
    # Test search
    print("\nSearch Tests:")
    
    # By class
    diagrams = api.search_by_class("diagram", top_k=3)
    print(f"  Diagrams: {len(diagrams)} found")
    for r in diagrams:
        print(f"    - {r.pdf_source} p{r.page_number}")
    
    # By PDF
    milacron = api.search_by_pdf("MILACRON", top_k=3)
    print(f"  MILACRON images: {len(milacron)} found")
    
    # Query search
    hydraulic = api.search("hydraulic diagram", top_k=3)
    print(f"  'hydraulic diagram': {len(hydraulic)} found")
    
    print("\n✓ API tests complete!")
