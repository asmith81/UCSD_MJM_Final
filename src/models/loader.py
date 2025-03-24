"""
Model loading utilities for vision-language models.

This module provides functionality to:
1. Load models with appropriate configurations for different environments
2. Apply memory optimization techniques (precision, quantization)
3. Monitor and report GPU memory usage
4. Validate model loading and report errors
"""

import os
import gc
import time
import logging
from typing import Dict, Any, Tuple, Optional, Union

import torch
from transformers import AutoProcessor, AutoTokenizer

# Import project utilities
from src.models.registry import get_model_config, get_loading_params
from src.config.environment import get_environment_config

# Set up logging
logger = logging.getLogger(__name__)

def load_model_and_processor(
    model_name: str,
    quantization: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load a model and its processor with appropriate configuration.
    
    Args:
        model_name: Name of the model to load
        quantization: Optional quantization strategy to use
        cache_dir: Optional override for model cache directory
        **kwargs: Additional parameters to override model loading params
        
    Returns:
        Tuple of (model, processor) instances
        
    Raises:
        ImportError: If required libraries are not installed
        RuntimeError: If model loading fails
        ValueError: If model configuration is invalid
    """
    start_time = time.time()
    
    # Get model configuration from registry
    model_config = get_model_config(model_name)
    
    # Get loading parameters
    loading_params = get_loading_params(model_name, quantization=quantization)
    
    # Filter out model and processor keys from kwargs and loading_params
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'processor']}
    loading_params = {k: v for k, v in loading_params.items() if k not in ['model', 'processor']}
    
    # Override with filtered kwargs
    loading_params.update(filtered_kwargs)
    
    # Override cache directory if provided
    if cache_dir is not None:
        loading_params["cache_dir"] = cache_dir
    
    # Ensure we're using the settings that worked in RunPod
    if torch.cuda.is_available():
        loading_params.setdefault("torch_dtype", torch.bfloat16)
        loading_params.setdefault("device_map", "cuda:0")
    
    # Log model loading attempt
    logger.info(f"Loading model {model_name} from {model_config['repo_id']}")
    logger.info(f"Loading parameters: {loading_params}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Initial GPU memory usage: {initial_memory:.2f} GB")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected, model will be loaded on CPU")
    
    # Import the correct model class based on model_type
    try:
        model_type = model_config.get("model_type")
        
        if model_type == "LlavaForConditionalGeneration":
            from transformers import LlavaForConditionalGeneration
            model_class = LlavaForConditionalGeneration
        else:
            # Handle other model types
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
    except ImportError as e:
        logger.error(f"Error importing required libraries: {str(e)}")
        raise
    
    try:
        # Load the model
        repo_id = model_config["repo_id"]
        
        # Try to import flash_attn to check if it's available
        try:
            import flash_attn
            logger.info(f"Flash Attention available (version {flash_attn.__version__})")
            # Still set this to False since we'll use attn_implementation parameter instead
            loading_params.setdefault("use_flash_attention_2", False)
            loading_params.setdefault("attn_implementation", "flash_attention_2")
        except ImportError:
            logger.info("Flash Attention not available, using eager implementation")
            loading_params.setdefault("use_flash_attention_2", False)
            loading_params.setdefault("attn_implementation", "eager")
        
        model = model_class.from_pretrained(repo_id, **loading_params)
        
        # Load the processor
        processor_type = model_config.get("processor_type", "AutoProcessor")
        if processor_type == "AutoProcessor":
            processor = AutoProcessor.from_pretrained(repo_id)
        elif processor_type == "AutoTokenizer":
            processor = AutoTokenizer.from_pretrained(repo_id)
        else:
            logger.error(f"Unsupported processor type: {processor_type}")
            raise ValueError(f"Unsupported processor type: {processor_type}")
        
        # Report memory usage if GPU is available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated() / 1e9
            memory_used = current_memory - initial_memory
            logger.info(f"Model loaded successfully using {memory_used:.2f} GB of GPU memory")
            logger.info(f"Total GPU memory usage: {current_memory:.2f} GB")
        
        # Verify model is on correct device
        if torch.cuda.is_available():
            if hasattr(model, "device"):
                logger.info(f"Model device: {model.device}")
            else:
                logger.info("Model device information not available")
        
        # Report loading time
        loading_time = time.time() - start_time
        logger.info(f"Model loading completed in {loading_time:.2f} seconds")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        # Try to clean up memory before raising exception
        optimize_memory(clear_cache=True)
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")


def optimize_memory(clear_cache: bool = True) -> None:
    """
    Optimize memory usage by clearing caches.
    
    Args:
        clear_cache: Whether to clear CUDA cache if available
    """
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache if available and requested
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get information about GPU memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    device = torch.cuda.current_device()
    
    # Get device properties
    device_props = torch.cuda.get_device_properties(device)
    
    # Calculate memory usage
    allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
    reserved = torch.cuda.memory_reserved(device) / 1e9    # GB
    total = device_props.total_memory / 1e9                # GB
    
    # Get device name
    device_name = device_props.name
    
    return {
        "gpu_available": True,
        "device_name": device_name,
        "total_memory_gb": total,
        "allocated_memory_gb": allocated,
        "reserved_memory_gb": reserved,
        "free_memory_gb": total - allocated,
        "utilization_percent": (allocated / total) * 100
    }


def verify_gpu_compatibility(model_name: str) -> Dict[str, Any]:
    """
    Verify that the current GPU is compatible with the model requirements.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        Dictionary with compatibility information
    """
    # Get model configuration
    model_config = get_model_config(model_name)
    
    # Get hardware requirements
    hardware_reqs = model_config.get("hardware", {})
    required_gpu = hardware_reqs.get("gpu_required", False)
    min_memory = hardware_reqs.get("gpu_memory_min", "0GB")
    
    # Get current GPU info
    gpu_info = get_gpu_memory_info()
    
    # Check if GPU is required but not available
    if required_gpu and not gpu_info.get("gpu_available", False):
        return {
            "compatible": False,
            "reason": "GPU required but not available",
            "model_requirements": hardware_reqs,
            "current_gpu": None
        }
    
    # Parse minimum memory requirement
    if isinstance(min_memory, str):
        if min_memory.endswith("GB"):
            min_memory_gb = float(min_memory[:-2])
        else:
            # Default to GB if no unit specified
            min_memory_gb = float(min_memory)
    else:
        min_memory_gb = float(min_memory) / 1e9  # Convert bytes to GB
    
    # Check if GPU memory is sufficient
    if gpu_info.get("gpu_available", False):
        has_enough_memory = gpu_info.get("total_memory_gb", 0) >= min_memory_gb
        
        if not has_enough_memory:
            return {
                "compatible": False,
                "reason": "Insufficient GPU memory",
                "model_requirements": hardware_reqs,
                "current_gpu": gpu_info
            }
    
    # If all checks pass, the GPU is compatible
    return {
        "compatible": True,
        "model_requirements": hardware_reqs,
        "current_gpu": gpu_info
    }


def extract_field_from_image(image_path, prompt, model=None, processor=None):
    """
    Extract field from an invoice image using the model.
    Replicates the successful approach from the RunPod notebook.
    
    Args:
        image_path: Path to the invoice image
        prompt: Prompt to use for extraction
        model: Loaded model instance
        processor: Loaded processor instance
        
    Returns:
        Tuple of (extracted_text, processing_time)
    """
    import torch
    from PIL import Image
    import time
    
    # Validate inputs
    if model is None or processor is None:
        raise ValueError("Model and processor must be provided")
    
    start_time = time.time()
    
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
        
        # Process using successful approach from RunPod notebook
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )
        
        # Convert inputs to appropriate dtypes
        for key in inputs:
            if key == "pixel_values":
                inputs[key] = inputs[key].to(dtype=torch.bfloat16, device=model.device)
            else:
                inputs[key] = inputs[key].to(device=model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        # Decode the output
        extracted_text = processor.batch_decode(
            outputs, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        processing_time = time.time() - start_time
        
        return extracted_text, processing_time
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return f"ERROR: {str(e)}", time.time() - start_time


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Simple test of GPU compatibility
    from src.models.registry import list_available_models
    
    print("GPU information:")
    gpu_info = get_gpu_memory_info()
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    print("\nModel compatibility:")
    models = list_available_models()
    for model_name in models:
        compatibility = verify_gpu_compatibility(model_name)
        status = "✓ Compatible" if compatibility["compatible"] else "✗ Incompatible"
        print(f"  {model_name}: {status}")
        if not compatibility["compatible"]:
            print(f"    Reason: {compatibility['reason']}")