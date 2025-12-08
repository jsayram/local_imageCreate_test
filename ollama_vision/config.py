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
