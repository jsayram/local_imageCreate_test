"""
FLUX.2 refiner module for photorealistic img2img refinement.

This module provides a minimal interface to use FLUX.2-dev as a second-stage
refiner for identity-locked images from InstantID.

Dependencies:
- diffusers (version with Flux2Pipeline support)
- torch (with bfloat16 support)
- Pillow (PIL)

Do NOT modify existing pipelines or main.py behavior.
"""

from __future__ import annotations

import torch
from PIL import Image
from diffusers import FluxPipeline

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.bfloat16


def build_flux_img2img_pipeline(
    repo_id: str = "black-forest-labs/FLUX.1-dev",
) -> FluxPipeline:
    """
    Build and return a FLUX pipeline for img2img-style refinement.
    This must NOT modify any existing pipelines in the project.
    
    Args:
        repo_id: HuggingFace model ID (FLUX.1-dev or FLUX.1-schnell)
    
    Returns:
        Configured FluxPipeline ready for img2img refinement
    
    Note:
        Uses FluxPipeline with image input for img2img workflow.
        FLUX.2 may not be publicly available yet; using FLUX.1-dev.
    """
    print(f"[FLUX] Loading {repo_id}...")
    
    pipe = FluxPipeline.from_pretrained(
        repo_id,
        torch_dtype=DTYPE,
    )
    
    if DEVICE == "mps":
        pipe.to(DEVICE)
        print("[FLUX] Using MPS device")
    elif DEVICE == "cuda":
        pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        print("[FLUX] Using CUDA with CPU offload")
    else:
        pipe.to(DEVICE)
        print("[FLUX] Using CPU")
    
    print("[FLUX] ✓ Pipeline ready!")
    return pipe


def refine_photo_with_flux(
    pipe: FluxPipeline,
    base_image: Image.Image,
    prompt: str,
    seed: int = 0,
    steps: int = 28,
    strength: float = 0.35,
    guidance_scale: float = 3.5,
) -> Image.Image:
    """
    Given an identity-locked base image (e.g. from InstantID) and a
    photographic prompt, run FLUX to refine it.
    
    This is a verification helper, not a production API.
    
    Args:
        pipe: FluxPipeline instance
        base_image: Input image to refine (identity-locked from InstantID)
        prompt: Photographic description for refinement
        seed: Random seed for reproducibility
        steps: Inference steps (20-50 recommended for FLUX)
        strength: Denoising strength (0.3-0.5 for subtle refinement, 0.6-0.8 for more change)
        guidance_scale: CFG scale (3.0-4.5 works well for FLUX)
    
    Returns:
        Refined PIL Image
    
    Note:
        FLUX uses img2img by passing an image parameter.
        Lower strength preserves identity better.
    """
    width, height = base_image.size
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    print(f"[FLUX] Refining with strength={strength}, steps={steps}, guidance={guidance_scale}")
    
    # FLUX img2img: pass image + prompt
    # strength controls how much to denoise (lower = more preservation)
    result = pipe(
        prompt=prompt,
        image=base_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        strength=strength,
    )
    
    print("[FLUX] ✓ Refinement complete")
    return result.images[0]
