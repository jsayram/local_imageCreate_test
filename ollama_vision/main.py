import ollama
import base64
import os
from datetime import datetime
from PIL import Image
from rich.console import Console
import torch
from diffusers import StableDiffusionPipeline
import gc
from prompts import SYSTEM_PROMPT
from config import MODEL_NAME

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
        
        # Model loading logic based on selected mode
        local_model_path = os.path.join(os.path.dirname(__file__), 'models', 'stable-diffusion-v1-4')
        
        if offline_mode:
            # OFFLINE MODE: Require local model
            if os.path.exists(local_model_path):
                console.print(f"[dim]Loading model from local cache ({device})...[/dim]")
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
            # ONLINE MODE: Download if needed
            if os.path.exists(local_model_path):
                console.print(f"[dim]Loading model from local cache ({device})...[/dim]")
                pipe = StableDiffusionPipeline.from_pretrained(
                    local_model_path, 
                    torch_dtype=dtype, 
                    local_files_only=True
                )
            else:
                console.print("[dim]Downloading model (online mode)...[/dim]")
                pipe = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", 
                    torch_dtype=dtype,
                    cache_dir=os.path.join(os.path.dirname(__file__), 'models')
                )
        
        pipe.safety_checker = None  # Disable safety checker to save memory
        pipe = pipe.to(device)
        
        # Generate image with configurable inference steps
        num_steps = 20  # Increase for better quality (10-50 typical range)
        generated_image = pipe(
            image_prompt, 
            num_inference_steps=num_steps, 
            generator=torch.Generator(device).manual_seed(42)
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