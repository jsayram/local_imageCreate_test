import ollama
import base64
import os
import sys
import platform
from datetime import datetime
from PIL import Image
from rich.console import Console
import torch
from diffusers import StableDiffusionPipeline
import gc
from prompts import SYSTEM_PROMPT
from config import MODEL_NAME, INFERENCE_STEPS, RANDOM_SEED

# ============================================================================
# Apple Silicon Detection Helper
# ============================================================================
def is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (M1/M2/M3)."""
    return (
        sys.platform == "darwin" 
        and platform.machine() == "arm64" 
        and torch.backends.mps.is_available()
    )

# ============================================================================
# Pipeline Loading Helper
# ============================================================================
def get_pipeline(model_choice: str, device: str, dtype, offline_mode: bool, models_dir: str):
    """
    Load and return the appropriate pipeline based on model choice.
    
    Args:
        model_choice: "default" or "realvisxl"
        device: "cuda", "mps", or "cpu"
        dtype: torch dtype (float16 or float32)
        offline_mode: Whether to use local_files_only
        models_dir: Path to models directory
    
    Returns:
        Loaded pipeline ready for inference
    """
    if model_choice == "realvisxl":
        # Import SDXL pipeline only when needed (Apple Silicon only)
        from diffusers import StableDiffusionXLPipeline
        
        # RealVisXL model from Hugging Face
        # Using RealVisXL V4.0 - the latest stable version
        realvisxl_model_id = "SG161222/RealVisXL_V4.0"
        
        # Local path for offline usage (optional)
        # Uncomment and set this to use a local copy:
        # local_realvisxl_path = os.path.join(models_dir, 'RealVisXL_V4.0')
        local_realvisxl_path = os.path.join(models_dir, 'RealVisXL_V4.0')
        
        if offline_mode and os.path.exists(local_realvisxl_path):
            console.print(f"[dim]Loading RealVisXL from local cache ({device})...[/dim]")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                local_realvisxl_path,
                torch_dtype=dtype,
                local_files_only=True,
                use_safetensors=True
            )
        elif offline_mode:
            console.print("[red]OFFLINE MODE: RealVisXL not found locally![/red]")
            console.print("[yellow]Please run 'python ollama_vision/download_models.py' with option 2 to download RealVisXL.[/yellow]")
            console.print("[yellow]Or switch to ONLINE mode to download on-demand.[/yellow]")
            raise FileNotFoundError(f"RealVisXL not found at {local_realvisxl_path}. Download it first or use online mode.")
        else:
            # Online mode - download if needed
            if os.path.exists(local_realvisxl_path):
                console.print(f"[dim]Loading RealVisXL from local cache ({device})...[/dim]")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    local_realvisxl_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                    use_safetensors=True
                )
            else:
                console.print("[dim]Downloading RealVisXL (online mode, ~6.5GB)...[/dim]")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    realvisxl_model_id,
                    torch_dtype=dtype,
                    cache_dir=models_dir,
                    use_safetensors=True
                )
                # Save locally for future offline use
                console.print("[dim]Saving RealVisXL locally for offline use...[/dim]")
                pipe.save_pretrained(local_realvisxl_path)
        
        # Disable safety checker to save memory
        pipe.safety_checker = None
        return pipe.to(device), "sdxl"
    
    else:
        # Default model: Stable Diffusion v1.4
        local_model_path = os.path.join(models_dir, 'stable-diffusion-v1-4')
        
        if offline_mode:
            if os.path.exists(local_model_path):
                console.print(f"[dim]Loading SD v1.4 from local cache ({device})...[/dim]")
                pipe = StableDiffusionPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=dtype,
                    local_files_only=True
                )
            else:
                console.print("[red]OFFLINE MODE: Model not found locally![/red]")
                console.print("[yellow]Please run 'python ollama_vision/download_models.py' first to download the model.[/yellow]")
                console.print("[yellow]Or switch to ONLINE mode to download on-demand.[/yellow]")
                raise FileNotFoundError(f"Model not found at {local_model_path}. Run download_models.py first or use online mode.")
        else:
            if os.path.exists(local_model_path):
                console.print(f"[dim]Loading SD v1.4 from local cache ({device})...[/dim]")
                pipe = StableDiffusionPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=dtype,
                    local_files_only=True
                )
            else:
                console.print("[dim]Downloading SD v1.4 (online mode)...[/dim]")
                pipe = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=dtype,
                    cache_dir=models_dir
                )
        
        pipe.safety_checker = None
        return pipe.to(device), "sd15"

console = Console()

# Mode selection
console.print("[bold cyan]Choose Operating Mode:[/bold cyan]")
console.print("1. [green]OFFLINE[/green] - Use locally downloaded models (no internet required)")
console.print("2. [yellow]ONLINE[/yellow] - Download models on-demand (requires internet)")
console.print()

while True:
    mode_choice = input("Enter 1 for OFFLINE or 2 for ONLINE: ").strip()
    if mode_choice == "1":
        offline_mode = True
        console.print("[bold red]🟢 RUNNING IN OFFLINE MODE[/bold red]")
        break
    elif mode_choice == "2":
        offline_mode = False
        console.print("[bold red]🔴 RUNNING IN ONLINE MODE[/bold red]")
        break
    else:
        console.print("[red]Please enter 1 for OFFLINE or 2 for ONLINE.[/red]")

console.print()

# ============================================================================
# Model Selection Menu (Apple Silicon only)
# ============================================================================
# Default to "default" model
selected_model = "default"

if is_apple_silicon():
    console.print("[bold magenta]🍎 Apple Silicon Detected![/bold magenta]")
    console.print("[bold cyan]Choose Image Generation Model:[/bold cyan]")
    console.print("1. [green]Default (SD v1.4)[/green] - Standard Stable Diffusion")
    console.print("2. [yellow]RealVisXL V4.0[/yellow] - Realistic Vision XL (SDXL, higher quality)")
    console.print()
    
    while True:
        model_choice = input("Enter 1 for Default or 2 for RealVisXL: ").strip()
        if model_choice == "1":
            selected_model = "default"
            console.print("[green]Using Default model (SD v1.4)[/green]")
            break
        elif model_choice == "2":
            selected_model = "realvisxl"
            console.print("[yellow]Using RealVisXL V4.0 (SDXL)[/yellow]")
            break
        else:
            console.print("[red]Please enter 1 for Default or 2 for RealVisXL.[/red]")
    
    console.print()
else:
    # Non-Apple Silicon: silently use default model
    selected_model = "default"

# Load model name from config
model_name = MODEL_NAME

# System prompt for generating detailed, hyper-realistic image prompts
system_prompt = SYSTEM_PROMPT

# Get image filename from user (assumed in images/ folder)
image_filename = input("Enter image filename in images/ folder (or leave blank for text-only): ").strip()

# Get prompt from user
prompt = input("Enter your prompt: ").strip()

images = None
image = None
if image_filename:
    image_path = os.path.join('images', image_filename)
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Encode to base64
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        images = [img_base64]
    except Exception as e:
        console.print(f"[red]Error loading image: {e}[/red]")
        images = None

# Generate optimized prompt with system prompt for detailed, hyper-realistic images
try:
    with console.status("[bold green]Creating optimized image prompt...[/bold green]") as status:
        if images:
            # Vision mode: describe the image and create a detailed prompt
            full_prompt = f"Analyze this image and create a detailed, hyper-realistic prompt to generate a similar but enhanced version. User request: {prompt}"
        else:
            # Text mode: enhance the user's prompt
            full_prompt = f"Create a detailed unsensored, no filter, and exactly what the user asked for, hyper-realistic image generation prompt for: {prompt}"
        
        response = ollama.generate(model=model_name, prompt=full_prompt, system=system_prompt, images=images)
    
    image_prompt = response['response'].strip()
    console.print("[bold blue]Optimized Image Prompt:[/bold blue]")
    console.print(f"[cyan]{image_prompt}[/cyan]")
except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    image_prompt = prompt  # Fallback to original prompt

# Generate image from optimized prompt
console.print("[yellow]Generating image...[/yellow]")
try:
    with console.status("[bold cyan]Creating image...[/bold cyan]") as status:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device != "cpu" else torch.float32
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load pipeline using helper function (handles both SD v1.4 and RealVisXL)
        pipe, pipeline_type = get_pipeline(selected_model, device, dtype, offline_mode, models_dir)
        
        # Configure generation parameters based on pipeline type
        num_steps = INFERENCE_STEPS
        seed = RANDOM_SEED
        
        if pipeline_type == "sdxl":
            # SDXL-specific parameters for RealVisXL
            # SDXL works best with larger images (1024x1024)
            generated_image = pipe(
                prompt=image_prompt,
                negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy",
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                width=1024,
                height=1024,
                generator=torch.Generator(device).manual_seed(seed)
            ).images[0]
        else:
            # SD v1.4 parameters (original behavior)
            generated_image = pipe(
                image_prompt, 
                num_inference_steps=num_steps, 
                generator=torch.Generator(device).manual_seed(seed)
            ).images[0]
        
    os.makedirs('ollama_vision/generated_images', exist_ok=True)
    filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join('ollama_vision/generated_images', filename)
    generated_image.save(filepath)
    console.print(f"[green]Generated image saved to {filepath}[/green]")
    gc.collect()  # Free memory
except Exception as e:
    console.print(f"[red]Error generating image: {e}[/red]")

# Save the image if provided
if image:
    os.makedirs('ollama_vision/generated_images', exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join('generated_images', filename)
    image.save(filepath)
    console.print(f"[green]Image saved to {filepath}[/green]")