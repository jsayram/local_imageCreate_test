"""
InstantID Pipeline Wrapper - Face Lock: Hard Mode
Provides stronger face/body consistency than IP-Adapter Plus using InsightFace embeddings.
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional

# Add third_party to path for ip_adapter imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'instantid'))

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis

# Import the vendored InstantID pipeline
from third_party.instantid.pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

# Device/dtype globals
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32


def ensure_instantid_checkpoints(local_root: str = "./checkpoints") -> dict:
    """
    Ensure that InstantID checkpoints are downloaded under local_root.
    Returns a dict with keys: controlnet_path, face_adapter_path.
    Does not re-download if files already exist.
    
    Args:
        local_root: Local directory to store checkpoints
        
    Returns:
        Dictionary with checkpoint paths
    """
    print(f"[InstantID] Checking checkpoints in {local_root}...")
    
    # Create directories
    controlnet_dir = os.path.join(local_root, "ControlNetModel")
    os.makedirs(controlnet_dir, exist_ok=True)
    
    # Checkpoint paths
    controlnet_config = os.path.join(controlnet_dir, "config.json")
    controlnet_weights = os.path.join(controlnet_dir, "diffusion_pytorch_model.safetensors")
    face_adapter = os.path.join(local_root, "ip-adapter.bin")
    
    # Download ControlNet config if missing
    if not os.path.exists(controlnet_config):
        print("[InstantID] Downloading ControlNet config...")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ControlNetModel/config.json",
            local_dir=local_root,
        )
    
    # Download ControlNet weights if missing
    if not os.path.exists(controlnet_weights):
        print("[InstantID] Downloading ControlNet weights (~2.5GB, this may take a while)...")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ControlNetModel/diffusion_pytorch_model.safetensors",
            local_dir=local_root,
        )
    
    # Download face adapter if missing
    if not os.path.exists(face_adapter):
        print("[InstantID] Downloading IP-Adapter weights...")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir=local_root,
        )
    
    print("[InstantID] ✓ All checkpoints ready")
    
    return {
        "controlnet_path": controlnet_dir,
        "face_adapter_path": face_adapter,
    }


def build_face_analyzer(model_root: str = ".") -> FaceAnalysis:
    """
    Initialize and return an insightface FaceAnalysis app using the 'antelopev2' model.
    
    Args:
        model_root: Root directory for InsightFace models (InsightFace adds 'models/' automatically)
        
    Returns:
        Initialized FaceAnalysis app
    """
    print("[InstantID] Initializing InsightFace face analyzer...")
    
    # InsightFace automatically appends "models/" to the root path
    # So passing "." results in models being stored at "./models/antelopev2"
    app = FaceAnalysis(
        name="antelopev2",
        root=model_root,
        providers=["CPUExecutionProvider"]  # MPS doesn't support InsightFace, use CPU
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("[InstantID] ✓ Face analyzer ready")
    return app


def extract_face_embeds_and_kps(
    face_app: FaceAnalysis,
    image_path: str,
) -> tuple:
    """
    Given a path to a reference image with a face, return:
      - face_emb: torch.Tensor of shape [512] (face embedding)
      - face_kps_image: PIL.Image with drawn keypoints suitable for InstantID
      
    Args:
        face_app: Initialized FaceAnalysis app
        image_path: Path to reference portrait
        
    Returns:
        Tuple of (face_embedding, keypoints_image)
        
    Raises:
        ValueError: If no face is found in the image
    """
    # Load image with OpenCV (BGR format for InsightFace)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Detect faces
    faces = face_app.get(image)
    
    if len(faces) == 0:
        raise ValueError(f"No face detected in {image_path}")
    
    # Take the largest face (by bbox area)
    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    face_info = faces[0]
    
    # Extract face embedding
    face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0)
    face_emb = face_emb.to(device=DEVICE, dtype=DTYPE)
    
    # Draw keypoints using InstantID's draw_kps function
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    face_kps = draw_kps(image_pil, face_info['kps'])
    
    return face_emb, face_kps


def build_instantid_pipeline(base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0") -> StableDiffusionXLInstantIDPipeline:
    """
    Build and return a StableDiffusionXLInstantIDPipeline ready for inference.
    Uses SDXL base 1.0 (or custom path) and the InstantID ControlNet + adapter.
    
    Args:
        base_model_path: HuggingFace ID or local path to SDXL base model
    
    Returns:
        Configured InstantID pipeline
    """
    print("[InstantID] Building pipeline...")
    
    # Ensure checkpoints are downloaded
    paths = ensure_instantid_checkpoints()
    
    # Load ControlNet
    print("[InstantID] Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        paths["controlnet_path"],
        torch_dtype=DTYPE,
    )
    
    # Build pipeline
    print(f"[InstantID] Loading SDXL base model from {base_model_path}...")
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    # Load the IP-Adapter / InstantID adapter
    print("[InstantID] Loading InstantID adapter...")
    pipe.load_ip_adapter_instantid(paths["face_adapter_path"])
    # Default to 0.8, but this should be overridden per-generation
    pipe.set_ip_adapter_scale(0.8)
    
    print("[InstantID] ✓ Pipeline ready!")
    return pipe


def generate_with_instantid(
    pipe: StableDiffusionXLInstantIDPipeline,
    face_app: FaceAnalysis,
    ref_image_path: str,
    scene_prompt: str,
    seed: int = 0,
    steps: int = 30,
    guidance_scale: float = 5.0,
    controlnet_conditioning_scale: float = 0.8,
    ip_adapter_scale: float = 0.8,
    return_debug: bool = False,
) -> Image.Image:
    """
    Convenience function:
      - Extract face embedding + keypoints from ref_image_path
      - Run the InstantID SDXL pipeline with that identity + scene prompt
      - Return a PIL.Image
      
    Args:
        pipe: InstantID pipeline from build_instantid_pipeline()
        face_app: FaceAnalysis app from build_face_analyzer()
        ref_image_path: Path to reference portrait
        scene_prompt: Scene description (e.g., "in a kitchen", "at night in city")
        seed: Random seed for reproducibility
        steps: Inference steps (20-50 recommended)
        guidance_scale: CFG scale (4.0-6.0 works well with InstantID)
        controlnet_conditioning_scale: ControlNet strength (1.0 = maximum face lock)
        ip_adapter_scale: Face embedding strength (0.8-1.0, higher = stricter)
        return_debug: If True, returns (image, face_kps_image) tuple
        
    Returns:
        Generated PIL Image (or tuple if return_debug=True)
    """
    # Set the IP-Adapter scale for this generation
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # Extract face features
    print(f"[InstantID] Extracting face from {ref_image_path}...")
    face_emb, face_kps = extract_face_embeds_and_kps(face_app, ref_image_path)
    
    # Setup generator
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Build prompts - keep it simple and within token limits
    full_prompt = scene_prompt
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality"
    )
    
    print(f"[InstantID] Generating: {scene_prompt[:80]}...")
    print(f"[InstantID] Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}")
    print(f"[InstantID] Face Lock: ControlNet={controlnet_conditioning_scale}, IP-Adapter={ip_adapter_scale}")
    
    # Generate with maximum face preservation
    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    print("[InstantID] ✓ Generation complete")
    
    if return_debug:
        return result.images[0], face_kps
        
    return result.images[0]


# ============================================================================
# Example integration (commented out - do not run here)
# ============================================================================
# from instantid_pipeline import build_instantid_pipeline, build_face_analyzer, generate_with_instantid
#
# instantid_pipe = build_instantid_pipeline()
# face_app = build_face_analyzer()
#
# def generate_instantid_character(prompt: str, seed: int = 0):
#     ref = "assets/characters/model_01.png"
#     return generate_with_instantid(instantid_pipe, face_app, ref, prompt, seed)
