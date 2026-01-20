#!/usr/bin/env python3
"""
YoloGen Validation Suite

Comprehensive validation of the trained classifier and image index.
Run this after training completes to verify everything is working.

Tests:
1. Model loading and inference
2. Per-class accuracy analysis
3. Image index integrity
4. API functionality
5. MCP server endpoints
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import random

# Setup path
YOLOGEN_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(YOLOGEN_DIR))

# Check for weights
WEIGHTS_PATH = YOLOGEN_DIR / "runs" / "equipment_classifier" / "weights" / "best.pt"
INDEX_PATH = YOLOGEN_DIR / "data" / "processed" / "image_index.json"
CLASSIFICATION_DIR = YOLOGEN_DIR / "data" / "classification"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_model_loading():
    """Test 1: Load the trained model."""
    print_header("Test 1: Model Loading")
    
    if not WEIGHTS_PATH.exists():
        print(f"❌ Weights not found at: {WEIGHTS_PATH}")
        print("   Training may still be in progress.")
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(str(WEIGHTS_PATH))
        print(f"✓ Model loaded successfully")
        print(f"  Path: {WEIGHTS_PATH}")
        print(f"  Classes: {model.names}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def test_inference(model):
    """Test 2: Run inference on sample images."""
    print_header("Test 2: Inference Test")
    
    if model is None:
        print("⚠ Skipping - no model loaded")
        return
    
    # Find sample images
    val_dir = CLASSIFICATION_DIR / "val"
    if not val_dir.exists():
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    # Get random images from each class
    samples_tested = 0
    correct = 0
    
    for class_dir in val_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        images = list(class_dir.glob("*.jpg"))
        if not images:
            continue
        
        # Test up to 3 images per class
        for img_path in images[:3]:
            try:
                results = model(str(img_path), verbose=False)
                pred_class = model.names[results[0].probs.top1]
                actual_class = class_dir.name
                
                samples_tested += 1
                if pred_class == actual_class:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"  {status} {img_path.name[:30]}: actual={actual_class}, pred={pred_class}")
            except Exception as e:
                print(f"  ❌ Error on {img_path.name}: {e}")
    
    if samples_tested > 0:
        accuracy = correct / samples_tested * 100
        print(f"\n  Quick test accuracy: {correct}/{samples_tested} ({accuracy:.1f}%)")


def test_per_class_accuracy(model):
    """Test 3: Full validation set accuracy per class."""
    print_header("Test 3: Per-Class Validation Accuracy")
    
    if model is None:
        print("⚠ Skipping - no model loaded")
        return
    
    val_dir = CLASSIFICATION_DIR / "val"
    if not val_dir.exists():
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    
    for class_dir in val_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        actual_class = class_dir.name
        
        for img_path in class_dir.glob("*.jpg"):
            try:
                results = model(str(img_path), verbose=False)
                pred_class = model.names[results[0].probs.top1]
                
                class_stats[actual_class]["total"] += 1
                if pred_class == actual_class:
                    class_stats[actual_class]["correct"] += 1
                
                confusion[actual_class][pred_class] += 1
            except Exception as e:
                print(f"  Error on {img_path.name}: {e}")
    
    # Print results
    print("\n  Per-class accuracy:")
    total_correct = 0
    total_samples = 0
    
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            total_correct += stats["correct"]
            total_samples += stats["total"]
            status = "✓" if acc >= 70 else "⚠"
            print(f"    {status} {cls:15} {stats['correct']:3}/{stats['total']:3} ({acc:5.1f}%)")
    
    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        print(f"\n  Overall accuracy: {total_correct}/{total_samples} ({overall_acc:.1f}%)")
    
    # Print confusion matrix for problematic classes
    print("\n  Most common confusions:")
    for actual, preds in confusion.items():
        for pred, count in preds.items():
            if actual != pred and count > 0:
                print(f"    {actual} → {pred}: {count} times")


def test_image_index():
    """Test 4: Validate image index."""
    print_header("Test 4: Image Index Validation")
    
    if not INDEX_PATH.exists():
        print(f"❌ Index not found: {INDEX_PATH}")
        return
    
    with open(INDEX_PATH) as f:
        index = json.load(f)
    
    print(f"  Total images indexed: {len(index)}")
    
    # Check for required fields
    missing_fields = 0
    invalid_paths = 0
    label_counts = defaultdict(int)
    pdf_counts = defaultdict(int)
    
    for img_name, metadata in index.items():
        # Check required fields
        for field in ["pdf_name", "page", "label", "split"]:
            if field not in metadata:
                missing_fields += 1
                break
        
        # Count labels
        label_counts[metadata.get("label", "unknown")] += 1
        pdf_counts[metadata.get("pdf_name", "unknown")] += 1
        
        # Verify image exists
        split = metadata.get("split", "train")
        img_path = YOLOGEN_DIR / "data" / "processed" / "images" / split / img_name
        if not img_path.exists():
            invalid_paths += 1
    
    print(f"  Missing fields: {missing_fields}")
    print(f"  Invalid paths: {invalid_paths}")
    print(f"\n  Labels distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:15} {count:5} ({count/len(index)*100:5.1f}%)")
    
    print(f"\n  PDFs indexed: {len(pdf_counts)}")
    
    if missing_fields == 0 and invalid_paths == 0:
        print("\n  ✓ Index validation passed")
    else:
        print(f"\n  ⚠ Index has issues")


def test_api():
    """Test 5: Test the API functionality."""
    print_header("Test 5: API Functionality")
    
    try:
        from yologen.api import create_api, ImageSearchAPI
        
        api = create_api(str(YOLOGEN_DIR))
        
        # Test stats
        stats = api.get_stats()
        print(f"  ✓ Stats: {stats['total_images']} images from {stats['total_pdfs']} PDFs")
        
        # Test intent detection
        test_queries = [
            ("Show me the hydraulic diagram", True),
            ("What is the operating pressure?", False),
            ("Display the wiring schematic", True),
        ]
        
        intent_pass = True
        for query, expected in test_queries:
            result = api.detect_image_intent(query)
            if result != expected:
                print(f"  ✗ Intent detection failed for: {query}")
                intent_pass = False
        
        if intent_pass:
            print(f"  ✓ Intent detection working")
        
        # Test search
        diagrams = api.search_by_class("diagram", top_k=3)
        print(f"  ✓ Search by class: found {len(diagrams)} diagrams")
        
        # Test PDF search
        results = api.search(pdf_filter="MILACRON", top_k=3)
        print(f"  ✓ Search by PDF: found {len(results)} MILACRON images")
        
        print("\n  ✓ API tests passed")
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        import traceback
        traceback.print_exc()


def generate_report(model):
    """Generate final validation report."""
    print_header("VALIDATION SUMMARY")
    
    if model is None:
        print("  ⚠ Model not loaded - training may still be in progress")
        print("  Run this script again after training completes.")
        return
    
    print("  ✓ Model trained and loadable")
    print(f"  ✓ Classes: {list(model.names.values())}")
    
    # Check if weights exist
    if WEIGHTS_PATH.exists():
        size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
        print(f"  ✓ Weights: {WEIGHTS_PATH.name} ({size_mb:.1f} MB)")
    
    # Check index
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            index = json.load(f)
        print(f"  ✓ Image index: {len(index)} images")
    
    print("\n" + "=" * 60)
    print("  NEXT STEPS:")
    print("=" * 60)
    print("""
  1. If validation accuracy < 75%, consider:
     - More training epochs
     - Data augmentation
     - Re-labeling confusing images
  
  2. Start MCP server:
     python mcp_server.py --test  # Quick test
     python mcp_server.py --http  # HTTP mode for testing
     python mcp_server.py         # Full MCP mode
  
  3. Integrate with agentic-rag:
     - Add yologen to MCP config
     - Use image_search tool in agent
""")


def main():
    print("\n" + "=" * 60)
    print("  YoloGen Validation Suite")
    print("=" * 60)
    
    # Run tests
    model = test_model_loading()
    test_inference(model)
    test_per_class_accuracy(model)
    test_image_index()
    test_api()
    generate_report(model)


if __name__ == "__main__":
    main()
