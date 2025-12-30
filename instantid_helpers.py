"""
InstantID helper functions for preset-based generation.

This module provides thin wrappers around InstantID pipelines to use
faceid_presets configurations without modifying existing code.
"""

from typing import Dict
from PIL import Image
import torch
from instantid_pipeline import extract_face_embeds_and_kps


def generate_instantid_with_config(
    pipe,
    face_app,
    ref_image_path: str,
    scene_prompt: str,
    cfg: Dict,
    seed: int = 0,
) -> Image.Image:
    """
    Generate image using InstantID pipeline with a preset configuration.
    
    This is a thin wrapper that applies preset configs (from faceid_presets.py)
    to InstantID generation. Does NOT modify existing pipeline behavior.
    
    Args:
        pipe: InstantID pipeline (StableDiffusionXLInstantIDPipeline)
        face_app: FaceAnalysis instance for extracting face embeddings
        ref_image_path: Path to reference portrait image
        scene_prompt: Text description of the scene
        cfg: Configuration dict (e.g., INSTANTID_MAX_FACE_LOCK, INSTANTID_CINEMATIC_BALANCED)
        seed: Random seed for reproducibility
    
    Returns:
        Generated PIL Image
    
    Example:
        >>> from faceid_presets import INSTANTID_MAX_FACE_LOCK
        >>> pipe = build_instantid_pipeline()
        >>> face_app = build_face_analyzer()
        >>> img = generate_instantid_with_config(
        ...     pipe=pipe,
        ...     face_app=face_app,
        ...     ref_image_path="assets/characters/model_01.png",
        ...     scene_prompt="in a softly lit studio",
        ...     cfg=INSTANTID_MAX_FACE_LOCK,
        ...     seed=100,
        ... )
    """
    # Extract face embeddings and keypoints
    face_emb, face_kps = extract_face_embeds_and_kps(face_app, ref_image_path)
    
    # Setup generator
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Build prompts - keep simple
    prompt = scene_prompt
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
    # Generate with InstantID
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=cfg["controlnet_conditioning_scale"],
        num_inference_steps=cfg["steps"],
        guidance_scale=cfg["guidance_scale"],
        width=cfg["width"],
        height=cfg["height"],
        generator=generator,
    )
    
    return result.images[0]
