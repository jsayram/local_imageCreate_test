#!/usr/bin/env python3
"""
Download and cache Stable Diffusion models for offline use.
Run this script once while connected to the internet to download all required models.
"""

import os
import sys
import platform
import torch
from diffusers import StableDiffusionPipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# ============================================================================
# Apple Silicon Detection
# ============================================================================
def is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (M1/M2/M3)."""
    return (
        sys.platform == "darwin" 
        and platform.machine() == "arm64" 
        and torch.backends.mps.is_available()
    )

def download_stable_diffusion():
    """Download Stable Diffusion model to local cache."""
    
    # Define local model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    model_path = os.path.join(models_dir, 'stable-diffusion-v1-4')
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        config_file = os.path.join(model_path, 'model_index.json')
        if os.path.exists(config_file):
            console.print(f"[green]✓ Model already downloaded at:[/green] {model_path}")
            return True
    
    console.print("[bold cyan]Downloading Stable Diffusion v1.4...[/bold cyan]")
    console.print("[yellow]This is a one-time download (~4GB). Please wait...[/yellow]")
    
    try:
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device != "cpu" else torch.float32
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading model files...", total=None)
            
            # Download the model - it will be cached in the models directory
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=dtype,
                cache_dir=models_dir
            )
            
            # Save the pipeline to a dedicated folder for easier loading
            progress.update(task, description="Saving model locally...")
            pipe.save_pretrained(model_path)
            
        console.print(f"[green]✓ Model downloaded successfully![/green]")
        console.print(f"[green]Location:[/green] {model_path}")
        console.print("[green]You can now run the app in OFFLINE mode.[/green]")
        return True
        
    except Exception as e:
        console.print(f"[bold red]✗ Error downloading model:[/bold red] {e}")
        console.print("[yellow]Please check your internet connection and try again.[/yellow]")
        return False

def download_realvisxl():
    """Download RealVisXL V4.0 model to local cache (Apple Silicon only)."""
    from diffusers import StableDiffusionXLPipeline
    
    # Define local model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    model_path = os.path.join(models_dir, 'RealVisXL_V4.0')
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        config_file = os.path.join(model_path, 'model_index.json')
        if os.path.exists(config_file):
            console.print(f"[green]✓ RealVisXL already downloaded at:[/green] {model_path}")
            return True
    
    console.print("[bold cyan]Downloading RealVisXL V4.0 (SDXL)...[/bold cyan]")
    console.print("[yellow]This is a one-time download (~6.5GB). Please wait...[/yellow]")
    
    try:
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Use MPS on Apple Silicon
        device = "mps"
        dtype = torch.float16
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading RealVisXL model files...", total=None)
            
            # Download from Hugging Face
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "SG161222/RealVisXL_V4.0",
                torch_dtype=dtype,
                cache_dir=models_dir,
                use_safetensors=True
            )
            
            # Save the pipeline locally for offline use
            progress.update(task, description="Saving RealVisXL locally...")
            pipe.save_pretrained(model_path)
            
        console.print(f"[green]✓ RealVisXL downloaded successfully![/green]")
        console.print(f"[green]Location:[/green] {model_path}")
        console.print("[green]You can now use RealVisXL in OFFLINE mode.[/green]")
        return True
        
    except Exception as e:
        console.print(f"[bold red]✗ Error downloading RealVisXL:[/bold red] {e}")
        console.print("[yellow]Please check your internet connection and try again.[/yellow]")
        return False

def verify_ollama():
    """Verify Ollama is installed and qwen3-vl model is available."""
    console.print("\n[bold cyan]Checking Ollama setup...[/bold cyan]")
    
    try:
        import ollama
        
        # Try to list models
        result = ollama.list()
        
        # Handle both dict and object response formats
        if hasattr(result, 'models'):
            models = result.models
        elif isinstance(result, dict) and 'models' in result:
            models = result['models']
        else:
            models = []
        
        # Extract model names - use 'model' attribute, not 'name'
        model_names = []
        for m in models:
            if hasattr(m, 'model'):
                model_names.append(m.model)
            elif isinstance(m, dict) and 'model' in m:
                model_names.append(m['model'])
        
        if 'qwen3-vl:latest' in model_names or any('qwen3-vl' in name for name in model_names):
            console.print("[green]✓ Ollama qwen3-vl model found[/green]")
            return True
        else:
            console.print("[yellow]⚠ Ollama is installed but qwen3-vl model not found[/yellow]")
            console.print("[yellow]Run: ollama pull qwen3-vl[/yellow]")
            return False
            
    except ImportError:
        console.print("[red]✗ Ollama Python package not installed[/red]")
        console.print("[yellow]Run: pip install ollama[/yellow]")
        return False
    except Exception as e:
        console.print(f"[yellow]⚠ Could not verify Ollama: {e}[/yellow]")
        console.print("[yellow]Make sure Ollama is installed and running[/yellow]")
        console.print("[yellow]Visit: https://ollama.ai/download[/yellow]")
        return False

def main():
    console.print("[bold]Model Setup for Offline Use[/bold]\n")
    
    # ============================================================================
    # Model Selection Menu
    # ============================================================================
    console.print("[bold cyan]Which model(s) would you like to download?[/bold cyan]")
    console.print("1. [green]SD v1.4 only[/green] - Standard Stable Diffusion (~4GB)")
    
    # Only show RealVisXL option on Apple Silicon
    if is_apple_silicon():
        console.print("2. [yellow]RealVisXL V4.0 only[/yellow] - Realistic Vision XL, SDXL (~6.5GB) [Apple Silicon]")
        console.print("3. [magenta]Both models[/magenta] - Download SD v1.4 and RealVisXL (~10.5GB)")
    else:
        console.print("[dim](RealVisXL option available only on Apple Silicon)[/dim]")
    
    console.print()
    
    while True:
        if is_apple_silicon():
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice in ["1", "2", "3"]:
                break
            console.print("[red]Please enter 1, 2, or 3.[/red]")
        else:
            choice = input("Enter your choice (1): ").strip()
            if choice == "1":
                break
            console.print("[red]Please enter 1.[/red]")
    
    console.print()
    
    # Download selected models
    sd_success = False
    realvisxl_success = False
    
    if choice == "1":
        sd_success = download_stable_diffusion()
    elif choice == "2" and is_apple_silicon():
        realvisxl_success = download_realvisxl()
    elif choice == "3" and is_apple_silicon():
        sd_success = download_stable_diffusion()
        console.print()
        realvisxl_success = download_realvisxl()
    
    # Verify Ollama
    ollama_success = verify_ollama()
    
    # Summary
    console.print("\n" + "="*60)
    
    models_ready = []
    if sd_success:
        models_ready.append("SD v1.4")
    if realvisxl_success:
        models_ready.append("RealVisXL")
    
    if models_ready and ollama_success:
        console.print(f"[bold green]✓ Setup complete! Models ready: {', '.join(models_ready)}[/bold green]")
        console.print("[green]App is ready for offline use.[/green]")
    elif models_ready:
        console.print(f"[bold yellow]⚠ Models ready ({', '.join(models_ready)}), but Ollama needs setup.[/bold yellow]")
    else:
        console.print("[bold red]✗ Setup incomplete. Please resolve errors above.[/bold red]")
    
    console.print("="*60)
    
    return 0 if (sd_success or realvisxl_success) else 1

if __name__ == "__main__":
    sys.exit(main())
