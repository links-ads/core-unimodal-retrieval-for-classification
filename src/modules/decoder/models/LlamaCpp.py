import os
from typing import Union, Dict, Any, Optional
import requests
from pathlib import Path
from llama_cpp import Llama

class LlamaCpp:
    """Class to handle generation with llama.cpp for low memory inference."""
    
    def __init__(self, model_path: str, model_type: str, **kwargs):
        """
        Initialize a model with llama.cpp for low memory inference.
        
        Args:
            model_path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            **kwargs: Additional parameters for the model
        """
        # Initialize the llama.cpp model
        self.model = Llama(
            model_path=model_path,
            model_type=model_type,
            n_gpu_layers=-1,
            n_ctx=4096,  # Context window size
        )
    
    def generate(self, prompt: str) -> Union[Dict[str, Any], str]:
        """
        Generate a response from the model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response as string
        """
        # Generate the response
        response = self.model(
            prompt
        )
        # Extract generated text
        generated_text = response["choices"][0]["text"]
        return generated_text
