"""
Character Pipeline Module for IP-Adapter Plus with RealVisXL
Enables consistent character generation across different scenes using a reference portrait.
"""

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from typing import Optional

# Constants
REALVIS_MODEL_ID = "SG161222/RealVisXL_V5.0"  # Default, can be overridden
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32


def build_realvis_ip_adapter_pipeline(
    model_id: str = REALVIS_MODEL_ID,
    ip_adapter_scale: float = 0.8,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
) -> AutoPipelineForText2Image:
    """
    Build a RealVisXL pipeline with IP-Adapter Plus for character consistency.
    
    Args:
        model_id: HuggingFace model ID for RealVisXL (default: V5.0)
        ip_adapter_scale: Strength of identity preservation (0.5-1.0)
                         Higher = stronger identity lock, lower = more variation
        device: Target device (default: auto-detect CUDA/MPS/CPU)
        dtype: Data type (default: float16 for GPU, float32 for CPU)
    
    Returns:
        Configured pipeline ready for character-consistent generation
    """
    device = device or DEVICE
    dtype = dtype or DTYPE
    
    print(f"[Character Pipeline] Loading on {device} with {dtype}")
    
    # Load the image encoder for IP-Adapter Plus
    print("[Character Pipeline] Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=dtype,
    )
    
    # Load RealVisXL SDXL text2img pipeline with image encoder
    print(f"[Character Pipeline] Loading {model_id}...")
    
    # Handle variant based on device
    variant = "fp16" if device in ["cuda", "mps"] and dtype == torch.float16 else None
    
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            image_encoder=image_encoder,
            variant=variant,
        ).to(device)
    except Exception as e:
        # Fallback without variant if it fails
        print(f"[Character Pipeline] Variant loading failed, trying without variant: {e}")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            image_encoder=image_encoder,
        ).to(device)
    
    # Load IP-Adapter Plus SDXL weights
    print("[Character Pipeline] Loading IP-Adapter Plus weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    )
    
    # Set identity strength
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    print(f"[Character Pipeline] IP-Adapter scale set to {ip_adapter_scale}")
    
    # Disable safety checker to save memory
    pipe.safety_checker = None
    
    print("[Character Pipeline] ✓ Pipeline ready!")
    return pipe


def generate_scene_with_reference(
    pipe: AutoPipelineForText2Image,
    ref_image_path: str,
    scene_prompt: str,
    seed: int = 0,
    steps: int = 30,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    negative_prompt: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image with consistent character identity from a reference portrait.
    
    Args:
        pipe: Pre-built IP-Adapter pipeline
        ref_image_path: Path to reference portrait image
        scene_prompt: Description of the desired scene/pose/setting
        seed: Random seed for reproducibility
        steps: Number of inference steps (30-50 recommended)
        guidance_scale: Prompt adherence (3.0-5.0 recommended for IP-Adapter)
        width: Output width (default: 1024)
        height: Output height (default: 1024)
        negative_prompt: What to avoid (optional, uses default if None)
    
    Returns:
        Generated PIL Image with consistent character in new scene
    """
    # Load reference image
    print(f"[Character Pipeline] Loading reference: {ref_image_path}")
    ref_image = load_image(ref_image_path)
    
    # Create generator for reproducibility
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Use scene prompt directly - IP-Adapter handles identity
    full_prompt = scene_prompt
    
    # Default negative prompt if not provided
    if negative_prompt is None:
        negative_prompt = (
            "same background as reference, duplicate scene, copied composition, "
            "identical pose, same lighting as reference, same outfit as reference, "
            "different person, wrong face, distorted face, deformed eyes, extra limbs, "
            "low quality, blurry, out of frame, bad anatomy, mutation, "
            "multiple people, clone, twin, disfigured, ugly, poorly drawn face"
        )
    
    print(f"[Character Pipeline] Generating: {scene_prompt}")
    print(f"[Character Pipeline] Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}")
    
    # Generate image
    result = pipe(
        prompt=full_prompt,
        ip_adapter_image=ref_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    
    print("[Character Pipeline] ✓ Generation complete")
    return result.images[0]


def adjust_ip_adapter_scale(pipe: AutoPipelineForText2Image, scale: float) -> None:
    """
    Adjust the IP-Adapter identity strength on an existing pipeline.
    
    Args:
        pipe: IP-Adapter enabled pipeline
        scale: New identity strength (0.5-1.0)
               Higher = stronger identity lock, less scene variation
               Lower = more creative variation, slightly weaker identity
    """
    pipe.set_ip_adapter_scale(scale)
    print(f"[Character Pipeline] IP-Adapter scale adjusted to {scale}")
