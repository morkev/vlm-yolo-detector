#!/usr/bin/env python3
"""
Build Semantic Embeddings for Image Search

Creates embeddings from VLM descriptions for semantic image search.
The embeddings are stored alongside the image index for fast retrieval.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("ERROR: sentence-transformers not installed")
    print("Run: pip install sentence-transformers")
    exit(1)


def build_search_text(entry: Dict) -> str:
    """
    Build a rich searchable text from image metadata.
    
    Combines:
    - VLM description (most important)
    - PDF/machine name
    - Keywords
    - Extraction type
    """
    parts = []
    
    # VLM description is primary
    if entry.get("vlm_description"):
        parts.append(entry["vlm_description"])
    
    # Add machine/PDF context
    pdf_name = entry.get("pdf_name", "")
    if pdf_name:
        # Clean up PDF name for searchability
        clean_name = pdf_name.replace("-", " ").replace("_", " ")
        parts.append(f"From {clean_name} manual")
    
    # Add keywords
    keywords = entry.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    
    # Add extraction type context
    extraction_type = entry.get("extraction_type", "")
    if extraction_type == "rendered":
        parts.append("Page rendering with diagrams or schematics")
    elif extraction_type == "embedded":
        parts.append("Embedded image from document")
    
    return " ".join(parts)


def build_embeddings(
    index_path: Path,
    output_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict:
    """Build embeddings for all images in the index."""
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Loading image index from: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    
    print(f"Building search texts for {len(index)} images...")
    
    # Build search texts
    filenames = []
    search_texts = []
    
    for filename, entry in tqdm(index.items(), desc="Preparing texts"):
        filenames.append(filename)
        search_texts.append(build_search_text(entry))
    
    print(f"Generating embeddings...")
    embeddings = model.encode(
        search_texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create embeddings data structure
    embeddings_data = {
        "model": model_name,
        "dimension": embeddings.shape[1],
        "count": len(filenames),
        "filenames": filenames,
        "embeddings": embeddings.tolist()  # Convert to list for JSON
    }
    
    # Save embeddings
    print(f"Saving embeddings to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f)
    
    # Also save as numpy for faster loading
    np_path = output_path.with_suffix(".npy")
    np.save(np_path, embeddings)
    print(f"Saved numpy embeddings to: {np_path}")
    
    # Save filename mapping separately (smaller file for quick loading)
    mapping_path = output_path.parent / "embedding_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "dimension": embeddings.shape[1],
            "filenames": filenames
        }, f, indent=2)
    print(f"Saved filename mapping to: {mapping_path}")
    
    print(f"\n{'='*60}")
    print(f"Embeddings built successfully!")
    print(f"{'='*60}")
    print(f"  Images: {len(filenames)}")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  Model: {model_name}")
    
    return embeddings_data


def test_search(
    query: str,
    index_path: Path,
    embeddings_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5
):
    """Test semantic search."""
    print(f"\nTesting search for: '{query}'")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Load embeddings
    np_path = embeddings_path.with_suffix(".npy")
    if np_path.exists():
        embeddings = np.load(np_path)
    else:
        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embeddings = np.array(data["embeddings"])
    
    # Load mapping
    mapping_path = embeddings_path.parent / "embedding_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    filenames = mapping["filenames"]
    
    # Load index for metadata
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Compute similarities
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nTop {top_k} results:")
    print("-" * 60)
    for i, idx in enumerate(top_indices):
        filename = filenames[idx]
        score = similarities[idx]
        entry = index.get(filename, {})
        
        print(f"\n{i+1}. {filename}")
        print(f"   Score: {score:.4f}")
        print(f"   PDF: {entry.get('pdf_name', 'N/A')}, Page: {entry.get('page', 'N/A')}")
        desc = entry.get("vlm_description", "N/A")
        print(f"   Description: {desc[:150]}..." if len(desc) > 150 else f"   Description: {desc}")


def main():
    parser = argparse.ArgumentParser(description="Build embeddings for image search")
    parser.add_argument("--index", type=str, default="data/processed/image_index.json",
                        help="Path to image index")
    parser.add_argument("--output", type=str, default="data/processed/image_embeddings.json",
                        help="Output path for embeddings")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model")
    parser.add_argument("--test", type=str, default=None,
                        help="Test query after building")
    
    args = parser.parse_args()
    
    index_path = Path(args.index)
    output_path = Path(args.output)
    
    if not index_path.exists():
        print(f"ERROR: Index not found: {index_path}")
        return
    
    # Build embeddings
    build_embeddings(index_path, output_path, args.model)
    
    # Test if requested
    if args.test:
        test_search(args.test, index_path, output_path, args.model)


if __name__ == "__main__":
    main()
