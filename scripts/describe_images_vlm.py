#!/usr/bin/env python3
"""
VLM Contextual Image Description Generator

Uses a Vision-Language Model (via Ollama) to generate contextual descriptions
for each extracted image. The descriptions incorporate:
1. The visual content of the image
2. Context from the PDF page text
3. Knowledge about the machine/equipment type

This creates rich, searchable descriptions for RAG image retrieval.

Usage:
    python scripts/describe_images_vlm.py --index data/processed/image_index.json
    python scripts/describe_images_vlm.py --index data/processed/image_index.json --batch-size 5
"""

import argparse
import json
import base64
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_VLM_MODEL = "llava:13b"  # or "llava:7b", "bakllava", etc.


# Equipment/machine context patterns to extract from PDF names
MACHINE_CONTEXTS = {
    "APSX-PIM": "APSX-PIM plastic injection molding machine - a desktop electric injection molding system for small-scale plastic part production",
    "BOY-35": "BOY 35 E VV injection molding machine - an industrial plastic injection molding system with vertical clamping unit",
    "RSA-G2": "TA Instruments RSA-G2 rheometer - a dynamic mechanical analyzer for measuring viscoelastic properties of materials",
    "DMA": "Dynamic Mechanical Analyzer - equipment for measuring mechanical properties of materials under oscillating stress/strain",
    "DSC": "Differential Scanning Calorimeter - thermal analysis equipment for measuring heat flow in materials",
    "TGA": "Thermogravimetric Analyzer - equipment for measuring mass changes in materials as a function of temperature",
    "TMA": "Thermomechanical Analyzer - equipment for measuring dimensional changes in materials with temperature",
    "DIL": "Dilatometer - equipment for measuring thermal expansion of materials",
    "FTIR": "Fourier Transform Infrared Spectrometer - analytical equipment for identifying chemical compounds",
    "TAM": "Thermal Activity Monitor - isothermal microcalorimeter for measuring heat flow",
    "ElectroForce": "TA ElectroForce mechanical testing system - for fatigue testing and dynamic characterization",
    "MAAC": "MAAC Thermoformer - industrial thermoforming equipment for plastic sheet forming",
    "MILACRON": "Milacron injection molding machine - industrial plastic injection molding equipment",
    "FOX": "TA Instruments FOX heat flow meter - for measuring thermal conductivity",
    "HR": "TA Instruments HR rheometer - hybrid rheometer for measuring flow properties",
    "ARES": "TA Instruments ARES rheometer - strain-controlled rheometer",
}


def get_machine_context(pdf_name: str) -> str:
    """Extract machine context from PDF name."""
    pdf_upper = pdf_name.upper()
    for key, context in MACHINE_CONTEXTS.items():
        if key.upper() in pdf_upper:
            return context
    return "industrial/laboratory equipment from a technical manual"


def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode image to base64 for Ollama API."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def describe_image_with_vlm(
    image_path: str,
    page_text: str,
    machine_context: str,
    model: str = DEFAULT_VLM_MODEL,
    max_retries: int = 3
) -> Optional[str]:
    """
    Generate a contextual description for an image using VLM.
    
    Args:
        image_path: Path to the image file
        page_text: Text extracted from the same PDF page
        machine_context: Description of the machine/equipment type
        model: Ollama model to use
        max_retries: Number of retry attempts
    """
    image_b64 = encode_image_base64(image_path)
    if not image_b64:
        return None
    
    # Truncate page text if too long
    page_context = page_text[:1500] if page_text else "No text available from this page."
    
    # Create a detailed prompt
    prompt = f"""You are analyzing an image from a technical equipment manual.

EQUIPMENT CONTEXT:
{machine_context}

PAGE TEXT CONTEXT:
{page_context}

TASK:
Look at this image and provide a detailed, technical description. Consider:
1. What type of visual is this? (schematic, diagram, photo, table, chart, warning sign, procedure illustration, component view, etc.)
2. What specific equipment, component, or concept does it show?
3. What technical information does it convey?
4. How would someone search for this image? What keywords would they use?

Provide a description in 2-3 sentences that:
- Identifies the type of visual content
- Names specific components, connectors, or systems shown
- Uses technical terminology appropriate for the equipment
- Would be useful for someone searching for this specific information

Description:"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 300
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "").strip()
                if description:
                    return description
            else:
                print(f"  API error (attempt {attempt + 1}): {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt + 1})")
        except Exception as e:
            print(f"  Error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def check_ollama_connection(model: str) -> bool:
    """Check if Ollama is running and model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            if model in models or model.split(":")[0] in [m.split(":")[0] for m in models]:
                return True
            print(f"Model '{model}' not found. Available: {models}")
            return False
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return False


def process_images_batch(
    index: Dict,
    model: str,
    batch_size: int = 1,
    skip_existing: bool = True,
    limit: int = None
) -> Dict:
    """Process all images and add VLM descriptions."""
    
    # Filter images that need processing
    to_process = []
    for filename, data in index.items():
        if skip_existing and data.get("vlm_description"):
            continue
        if not Path(data["path"]).exists():
            print(f"Warning: Image not found: {data['path']}")
            continue
        to_process.append((filename, data))
    
    if limit:
        to_process = to_process[:limit]
    
    print(f"Processing {len(to_process)} images...")
    
    processed = 0
    errors = 0
    
    for filename, data in tqdm(to_process, desc="Generating descriptions"):
        machine_context = get_machine_context(data.get("pdf_name", ""))
        page_text = data.get("page_text", "")
        
        description = describe_image_with_vlm(
            image_path=data["path"],
            page_text=page_text,
            machine_context=machine_context,
            model=model
        )
        
        if description:
            index[filename]["vlm_description"] = description
            processed += 1
        else:
            errors += 1
            index[filename]["vlm_description"] = f"[Auto] Image from {data.get('pdf_name', 'unknown')} page {data.get('page', 0)}"
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)
    
    print(f"\nProcessed: {processed}, Errors: {errors}")
    return index


def generate_searchable_keywords(description: str, page_text: str, pdf_name: str) -> List[str]:
    """Extract searchable keywords from description and context."""
    keywords = set()
    
    # Add PDF name parts
    for part in pdf_name.replace("-", " ").replace("_", " ").split():
        if len(part) > 2:
            keywords.add(part.lower())
    
    # Common technical terms to look for
    tech_terms = [
        "schematic", "diagram", "wiring", "circuit", "connector", "component",
        "assembly", "procedure", "step", "instruction", "warning", "caution",
        "specification", "table", "chart", "graph", "photo", "image",
        "hydraulic", "electrical", "mechanical", "pneumatic", "control",
        "panel", "display", "sensor", "motor", "valve", "pump", "heater",
        "injection", "molding", "clamp", "nozzle", "barrel", "screw",
        "temperature", "pressure", "flow", "speed", "force", "torque"
    ]
    
    text_combined = f"{description} {page_text}".lower()
    for term in tech_terms:
        if term in text_combined:
            keywords.add(term)
    
    return list(keywords)


def main():
    parser = argparse.ArgumentParser(description="Generate VLM descriptions for images")
    parser.add_argument("--index", type=str, default="data/processed/image_index.json",
                        help="Path to image index JSON")
    parser.add_argument("--model", type=str, default=DEFAULT_VLM_MODEL,
                        help="Ollama VLM model to use")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip images that already have descriptions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for updated index (default: same as input)")
    
    args = parser.parse_args()
    
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"ERROR: Index file not found: {index_path}")
        return
    
    # Check Ollama connection
    print(f"Checking Ollama connection with model '{args.model}'...")
    if not check_ollama_connection(args.model):
        print("Please ensure Ollama is running and the model is available.")
        print(f"Try: ollama pull {args.model}")
        return
    print("Ollama connection OK!")
    
    # Load index
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    
    print(f"Loaded {len(index)} images from index")
    
    # Process images
    updated_index = process_images_batch(
        index=index,
        model=args.model,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        limit=args.limit
    )
    
    # Add searchable keywords
    print("\nGenerating searchable keywords...")
    for filename, data in updated_index.items():
        keywords = generate_searchable_keywords(
            data.get("vlm_description", ""),
            data.get("page_text", ""),
            data.get("pdf_name", "")
        )
        data["keywords"] = keywords
    
    # Save updated index
    output_path = Path(args.output) if args.output else index_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_index, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated index saved to: {output_path}")
    
    # Show sample descriptions
    print("\n" + "="*60)
    print("Sample Descriptions:")
    print("="*60)
    samples = list(updated_index.items())[:3]
    for filename, data in samples:
        print(f"\n{filename}:")
        print(f"  PDF: {data.get('pdf_name')}, Page: {data.get('page')}")
        print(f"  Type: {data.get('extraction_type')}")
        desc = data.get('vlm_description', 'N/A')
        print(f"  Description: {desc[:200]}..." if len(desc) > 200 else f"  Description: {desc}")
        print(f"  Keywords: {data.get('keywords', [])[:10]}")


if __name__ == "__main__":
    main()
