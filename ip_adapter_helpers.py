"""
IP-Adapter helper functions for preset-based generation.

This module provides thin wrappers around IP-Adapter pipelines to use
faceid_presets configurations without modifying existing code.
"""

from typing import Dict
from PIL import Image
import torch
from diffusers.utils import load_image


def generate_with_ip_and_config(
    pipe,
    ref_image_path: str,
    scene_prompt: str,
    cfg: Dict,
    seed: int = 0,
) -> Image.Image:
    """
    Generate image using IP-Adapter pipeline with a preset configuration.
    
    This is a thin wrapper that applies preset configs (from faceid_presets.py)
    to IP-Adapter generation. Does NOT modify existing pipeline behavior.
    
    Args:
        pipe: IP-Adapter SDXL/RealVisXL pipeline (with IP-Adapter loaded)
        ref_image_path: Path to reference portrait image
        scene_prompt: Text description of the scene
        cfg: Configuration dict (e.g., IP_MAX_FACE_LOCK, IP_CINEMATIC_BALANCED)
        seed: Random seed for reproducibility
    
    Returns:
        Generated PIL Image
    
    Example:
        >>> from faceid_presets import IP_MAX_FACE_LOCK
        >>> pipe = build_realvis_ip_adapter_pipeline()
        >>> img = generate_with_ip_and_config(
        ...     pipe=pipe,
        ...     ref_image_path="assets/characters/model_01.png",
        ...     scene_prompt="in a cozy apartment, warm lighting",
        ...     cfg=IP_MAX_FACE_LOCK,
        ...     seed=42,
        ... )
    """
    # Load reference image
    ref_image = load_image(ref_image_path)
    
    # Setup generator
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Apply IP-Adapter scale from config
    pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])
    
    # Build prompts
    prompt = f"same woman as the reference photo, {scene_prompt}"
    
    negative_prompt = (
        "different person, distorted face, deformed eyes, asymmetrical face, "
        "low quality, blurry, painting, illustration, 3d render, anime"
    )
    
    # Generate
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=ref_image,
        num_inference_steps=cfg["steps"],
        guidance_scale=cfg["guidance_scale"],
        width=cfg["width"],
        height=cfg["height"],
        generator=generator,
    )
    
    return result.images[0]
