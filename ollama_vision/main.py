import ollama
import base64
import os
from datetime import datetime
from PIL import Image
from rich.console import Console
import torch
from diffusers import StableDiffusionPipeline

console = Console()

# Assuming the vision model is 'qwen3-vl:latest', change if different
model_name = 'qwen3-vl:latest'

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

# Generate response with loading indicator
try:
    with console.status("[bold green]Generating response...[/bold green]") as status:
        response = ollama.generate(model=model_name, prompt=prompt, images=images)
    console.print("[bold blue]Response:[/bold blue]", response['response'])
except Exception as e:
    console.print(f"[red]Error: {e}[/red]")

# Generate image from prompt or response
image_prompt = response['response'] if images else prompt
console.print("[yellow]Generating image...[/yellow]")
try:
    with console.status("[bold cyan]Creating image...[/bold cyan]") as status:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        generated_image = pipe(image_prompt, num_inference_steps=20).images[0]  # Reduced steps for speed
    os.makedirs('generated_images', exist_ok=True)
    filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join('generated_images', filename)
    generated_image.save(filepath)
    console.print(f"[green]Generated image saved to {filepath}[/green]")
except Exception as e:
    console.print(f"[red]Error generating image: {e}[/red]")

# Save the image if provided
if image:
    os.makedirs('generated_images', exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join('generated_images', filename)
    image.save(filepath)
    console.print(f"[green]Image saved to {filepath}[/green]")