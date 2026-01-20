"""
Ollama VLM Integration for YoloGen

Replaces Qwen VLM with Ollama-hosted models like LLaVA.
This allows using local models without Chinese dependencies.

Supported models:
- LLaVA 1.6 (llava:latest, llava-llama3, etc.)
- Any Ollama vision model

Usage:
    from yologen.models.vlm.ollama import OllamaVLM
    
    vlm = OllamaVLM(model="hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M")
    response = vlm.generate(image_path="image.jpg", question="What is in this image?")
"""

import base64
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import cv2
import numpy as np


class OllamaVLM:
    """
    Ollama-based Vision-Language Model interface.
    
    Works with LLaVA and other vision models hosted by Ollama.
    """
    
    def __init__(
        self,
        model: str = "hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M",
        base_url: str = "http://localhost:11434",
        system_prompt: str = None,
    ):
        """
        Initialize Ollama VLM.
        
        Args:
            model: Ollama model name (e.g., "llava:latest", "llava-llama3")
            base_url: Ollama API base URL
            system_prompt: Optional system prompt for the model
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt or (
            "You are a technical documentation assistant specialized in equipment manuals. "
            "When shown an image, identify and describe components, diagrams, or information clearly. "
            "Provide concise, technical descriptions."
        )
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if not any(self.model in name or name in self.model for name in model_names):
                    print(f"Warning: Model '{self.model}' may not be available.")
                    print(f"Available models: {model_names}")
            else:
                print(f"Warning: Could not verify Ollama models")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running: ollama serve")
    
    def _encode_image(self, image_path: Union[str, Path, np.ndarray]) -> str:
        """Encode image to base64 for Ollama API."""
        if isinstance(image_path, np.ndarray):
            # Numpy array - encode directly
            _, buffer = cv2.imencode('.jpg', image_path)
            return base64.b64encode(buffer).decode('utf-8')
        else:
            # File path
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    
    def generate(
        self,
        image_path: Union[str, Path, np.ndarray],
        question: str = "Describe what you see in this image.",
        max_tokens: int = 500,
        temperature: float = 0.7,
        retries: int = 3,
    ) -> str:
        """
        Generate response for an image.
        
        Args:
            image_path: Path to image or numpy array
            question: Question to ask about the image
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retries: Number of retry attempts on failure
        
        Returns:
            Model's text response
        """
        image_b64 = self._encode_image(image_path)
        
        payload = {
            "model": self.model,
            "prompt": question,
            "system": self.system_prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120  # Vision models can be slow
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        print(f"Error calling Ollama after {retries} attempts: {last_error}")
        return f"Error: {last_error}"
    
    def generate_with_bbox(
        self,
        image_path: Union[str, Path],
        bbox: List[int],
        question: str = "What is in the red marked area?",
        box_color: tuple = (255, 0, 0),
        box_thickness: int = 3,
    ) -> str:
        """
        Generate response for a specific region marked with a bounding box.
        
        Args:
            image_path: Path to image
            bbox: Bounding box [x1, y1, x2, y2]
            question: Question about the marked region
            box_color: RGB color for the box
            box_thickness: Line thickness for the box
        
        Returns:
            Model's text response
        """
        # Load and draw bbox
        image = cv2.imread(str(image_path))
        if image is None:
            return f"Error: Could not load image {image_path}"
        
        # Convert RGB to BGR for OpenCV
        box_color_bgr = (box_color[2], box_color[1], box_color[0])
        
        # Draw rectangle
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color_bgr, box_thickness)
        
        return self.generate(image, question)
    
    def batch_generate(
        self,
        image_paths: List[Union[str, Path]],
        questions: List[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple images.
        
        Args:
            image_paths: List of image paths
            questions: Optional list of questions (one per image)
            **kwargs: Additional arguments for generate()
        
        Returns:
            List of responses
        """
        if questions is None:
            questions = ["Describe what you see in this image."] * len(image_paths)
        
        responses = []
        for img, q in zip(image_paths, questions):
            resp = self.generate(img, q, **kwargs)
            responses.append(resp)
        
        return responses


class OllamaVLMPredictor:
    """
    VLM Predictor using Ollama for inference.
    
    Drop-in replacement for Qwen-based VLMPredictor.
    """
    
    def __init__(
        self,
        vlm_model: str = "hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M",
        base_url: str = "http://localhost:11434",
        box_color: tuple = (255, 0, 0),
        box_thickness: int = 3,
    ):
        """
        Initialize Ollama VLM Predictor.
        
        Args:
            vlm_model: Ollama model name
            base_url: Ollama API URL
            box_color: RGB color for bounding boxes
            box_thickness: Line thickness for boxes
        """
        self.vlm = OllamaVLM(model=vlm_model, base_url=base_url)
        self.box_color = box_color
        self.box_thickness = box_thickness
    
    def predict(
        self,
        image: Union[str, Path],
        bbox: List[int] = None,
        question: str = "What is in the red marked area?",
    ) -> str:
        """
        Predict/describe content in image.
        
        Args:
            image: Path to image
            bbox: Optional bounding box [x1, y1, x2, y2]
            question: Question about the image
        
        Returns:
            Text description
        """
        if bbox:
            return self.vlm.generate_with_bbox(
                image_path=image,
                bbox=bbox,
                question=question,
                box_color=self.box_color,
                box_thickness=self.box_thickness,
            )
        else:
            return self.vlm.generate(image_path=image, question=question)


# Test function
def test_ollama_vlm():
    """Test Ollama VLM connection."""
    print("Testing Ollama VLM connection...")
    
    vlm = OllamaVLM()
    
    # Just test the connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running with {len(models)} models")
            for m in models:
                name = m.get("name", "unknown")
                size = m.get("size", 0) / (1024**3)
                print(f"  - {name} ({size:.1f} GB)")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    test_ollama_vlm()
