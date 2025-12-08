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
from config import MODEL_NAME, INFERENCE_STEPS, RANDOM_SEED, REALVISXL_CONFIG, SD_V14_CONFIG

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
        
        # RealVisXL model from Hugging Face - use config or default to V5.0
        realvisxl_model_id = REALVISXL_CONFIG.get('model_id', 'SG161222/RealVisXL_V5.0')
        
        # Extract version from model_id for local path (e.g., "RealVisXL_V5.0")
        model_version = realvisxl_model_id.split('/')[-1] if '/' in realvisxl_model_id else 'RealVisXL_V5.0'
        local_realvisxl_path = os.path.join(models_dir, model_version)
        
        if offline_mode and os.path.exists(local_realvisxl_path):
            console.print(f"[dim]Loading {model_version} from local cache ({device})...[/dim]")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                local_realvisxl_path,
                torch_dtype=dtype,
                local_files_only=True,
                use_safetensors=True
            )
        elif offline_mode:
            console.print("[red]OFFLINE MODE: RealVisXL not found locally![/red]")
            console.print(f"[yellow]Expected location: {local_realvisxl_path}[/yellow]")
            console.print()
            
            # Offer to download now
            console.print("[bold cyan]Would you like to download RealVisXL now? (~6.5GB)[/bold cyan]")
            download_choice = input("Enter Y to download or N to cancel: ").strip().lower()
            
            if download_choice == 'y':
                console.print("[dim]Downloading RealVisXL V5.0...[/dim]")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    realvisxl_model_id,
                    torch_dtype=dtype,
                    cache_dir=models_dir,
                    use_safetensors=True
                )
                console.print("[dim]Saving RealVisXL locally for offline use...[/dim]")
                pipe.save_pretrained(local_realvisxl_path)
                console.print("[green]✓ RealVisXL downloaded successfully![/green]")
            else:
                console.print("[yellow]Download cancelled. Please run 'python ollama_vision/download_models.py' to download later.[/yellow]")
                raise FileNotFoundError(f"RealVisXL not found at {local_realvisxl_path}. Download cancelled.")
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
                console.print("[red]OFFLINE MODE: SD v1.4 not found locally![/red]")
                console.print(f"[yellow]Expected location: {local_model_path}[/yellow]")
                console.print()
                
                # Offer to download now
                console.print("[bold cyan]Would you like to download SD v1.4 now? (~4GB)[/bold cyan]")
                download_choice = input("Enter Y to download or N to cancel: ").strip().lower()
                
                if download_choice == 'y':
                    console.print("[dim]Downloading Stable Diffusion v1.4...[/dim]")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        torch_dtype=dtype,
                        cache_dir=models_dir
                    )
                    console.print("[dim]Saving SD v1.4 locally for offline use...[/dim]")
                    pipe.save_pretrained(local_model_path)
                    console.print("[green]✓ SD v1.4 downloaded successfully![/green]")
                else:
                    console.print("[yellow]Download cancelled. Please run 'python ollama_vision/download_models.py' to download later.[/yellow]")
                    raise FileNotFoundError(f"SD v1.4 not found at {local_model_path}. Download cancelled.")
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
    console.print("2. [yellow]RealVisXL V5.0[/yellow] - Realistic Vision XL (SDXL, higher quality)")
    console.print()
    
    while True:
        model_choice = input("Enter 1 for Default or 2 for RealVisXL: ").strip()
        if model_choice == "1":
            selected_model = "default"
            console.print("[green]Using Default model (SD v1.4)[/green]")
            break
        elif model_choice == "2":
            selected_model = "realvisxl"
            console.print("[yellow]Using RealVisXL V5.0 (SDXL)[/yellow]")
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
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Check if model exists BEFORE starting the spinner (so download prompt works)
    pipe, pipeline_type = get_pipeline(selected_model, device, dtype, offline_mode, models_dir)
    
    with console.status("[bold cyan]Creating image...[/bold cyan]") as status:
        
        # Configure generation parameters based on pipeline type
        seed = RANDOM_SEED
        
        if pipeline_type == "sdxl":
            # RealVisXL parameters from config
            num_images = REALVISXL_CONFIG.get('num_images', 1)
            console.print(f"[dim]Using RealVisXL config: {REALVISXL_CONFIG['width']}x{REALVISXL_CONFIG['height']}, {REALVISXL_CONFIG['inference_steps']} steps, {num_images} image(s)[/dim]")
            
            # Parse dual prompts (prompt on line 1, prompt_2 on line 2)
            prompt_lines = image_prompt.strip().split('\n')
            main_prompt = prompt_lines[0].strip() if len(prompt_lines) > 0 else image_prompt
            secondary_prompt = prompt_lines[1].strip() if len(prompt_lines) > 1 else ""
            
            # Truncate prompts to ~75 tokens (roughly 60 words) to avoid CLIP overflow
            def truncate_prompt(prompt, max_words=55):
                words = prompt.split()
                if len(words) > max_words:
                    truncated = ' '.join(words[:max_words])
                    console.print(f"[yellow]⚠ Prompt truncated from {len(words)} to {max_words} words[/yellow]")
                    return truncated
                return prompt
            
            main_prompt = truncate_prompt(main_prompt)
            secondary_prompt = truncate_prompt(secondary_prompt) if secondary_prompt else ""
            
            # Show parsed prompts
            if secondary_prompt:
                console.print(f"[dim]Prompt 1: {main_prompt[:80]}...[/dim]")
                console.print(f"[dim]Prompt 2: {secondary_prompt[:80]}...[/dim]")
            
            # Parse negative prompts (support dual negative prompts too)
            neg_prompt = REALVISXL_CONFIG.get('negative_prompt', '')
            
            # Generate multiple images with different seeds
            generated_images = []
            for i in range(num_images):
                current_seed = seed + i  # Vary seed for each image
                console.print(f"[dim]Generating image {i+1}/{num_images} (seed: {current_seed})...[/dim]")
                
                result = pipe(
                    prompt=main_prompt,
                    prompt_2=secondary_prompt if secondary_prompt else None,
                    negative_prompt=neg_prompt,
                    negative_prompt_2=neg_prompt,
                    num_inference_steps=REALVISXL_CONFIG.get('inference_steps', 30),
                    guidance_scale=REALVISXL_CONFIG.get('guidance_scale', 7.5),
                    width=REALVISXL_CONFIG.get('width', 1024),
                    height=REALVISXL_CONFIG.get('height', 1024),
                    generator=torch.Generator(device).manual_seed(current_seed)
                ).images[0]
                generated_images.append(result)
        else:
            # SD v1.4 parameters from config
            num_images = SD_V14_CONFIG.get('num_images', 1)
            console.print(f"[dim]Using SD v1.4 config: {SD_V14_CONFIG['width']}x{SD_V14_CONFIG['height']}, {SD_V14_CONFIG['inference_steps']} steps, {num_images} image(s)[/dim]")
            
            # Generate multiple images with different seeds
            generated_images = []
            for i in range(num_images):
                current_seed = seed + i
                console.print(f"[dim]Generating image {i+1}/{num_images} (seed: {current_seed})...[/dim]")
                
                result = pipe(
                    prompt=image_prompt,
                    negative_prompt=SD_V14_CONFIG.get('negative_prompt', ''),
                    num_inference_steps=SD_V14_CONFIG.get('inference_steps', 50),
                    guidance_scale=SD_V14_CONFIG.get('guidance_scale', 7.5),
                    width=SD_V14_CONFIG.get('width', 512),
                    height=SD_V14_CONFIG.get('height', 512),
                    generator=torch.Generator(device).manual_seed(current_seed)
                ).images[0]
                generated_images.append(result)
        
    # Get output directory from config based on pipeline type
    if pipeline_type == "sdxl":
        output_dir = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/')
    else:
        output_dir = SD_V14_CONFIG.get('output_directory', 'ollama_vision/generated_images/sd_v14/')
    
    # Save all generated images
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, img in enumerate(generated_images):
        if len(generated_images) > 1:
            filename = f"generated_{timestamp}_{i+1}.png"
        else:
            filename = f"generated_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        console.print(f"[green]Generated image saved to {filepath}[/green]")
    
    console.print(f"[bold green]✓ {len(generated_images)} image(s) generated successfully![/bold green]")
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