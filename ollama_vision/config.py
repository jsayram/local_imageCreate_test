"""
Configuration management for Ollama Vision
Loads settings from config.json file
"""

import json
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file"""
    # Look for config.json in the project root (parent directory of ollama_vision)
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    config_path = os.path.abspath(config_path)

    # Default configuration
    default_config = {
        "model_name": "qwen3-vl:latest",
        "inference_steps": 20,
        "random_seed": 42,
        "output_directory": "ollama_vision/generated_images"
    }

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge user config with defaults
            config = {**default_config, **user_config}
            return config
        else:
            print(f"Warning: config.json not found at {config_path}")
            print("Using default configuration")
            return default_config
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}")
        print("Using default configuration")
        return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        return default_config

# Load configuration on import
CONFIG = load_config()

# Convenience variables
MODEL_NAME = CONFIG.get('model_name', 'qwen3-vl:latest')
INFERENCE_STEPS = CONFIG.get('inference_steps', 20)
RANDOM_SEED = CONFIG.get('random_seed', 42)
OUTPUT_DIRECTORY = CONFIG.get('output_directory', 'ollama_vision/generated_images')

# RealVisXL configuration
REALVISXL_CONFIG = CONFIG.get('realvisxl', {
    "model_id": "SG161222/RealVisXL_V5.0",
    "inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, signature"
})

# RealVisXL V4.0 configuration
REALVISXL_V4_CONFIG = CONFIG.get('realvisxl_v4', {
    "model_id": "SG161222/RealVisXL_V4.0",
    "inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, signature"
})

# SD v1.4 configuration
SD_V14_CONFIG = CONFIG.get('sd_v14', {
    "model_id": "CompVis/stable-diffusion-v1-4",
    "inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "negative_prompt": "blurry, low quality, distorted"
})

# Character consistency configuration
CHARACTER_CONSISTENCY_CONFIG = CONFIG.get('character_consistency', {
    "enabled": False,
    "description": "Use the same random_seed + detailed character description to maintain consistency"
})
