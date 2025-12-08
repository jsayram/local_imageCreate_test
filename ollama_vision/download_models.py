#!/usr/bin/env python3
"""
Download and cache Stable Diffusion models for offline use.
Run this script once while connected to the internet to download all required models.
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys

console = Console()

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
    
    # Download Stable Diffusion
    sd_success = download_stable_diffusion()
    
    # Verify Ollama
    ollama_success = verify_ollama()
    
    # Summary
    console.print("\n" + "="*60)
    if sd_success and ollama_success:
        console.print("[bold green]✓ Setup complete! App is ready for offline use.[/bold green]")
    elif sd_success:
        console.print("[bold yellow]⚠ Stable Diffusion ready, but Ollama needs setup.[/bold yellow]")
    else:
        console.print("[bold red]✗ Setup incomplete. Please resolve errors above.[/bold red]")
    console.print("="*60)
    
    return 0 if sd_success else 1

if __name__ == "__main__":
    sys.exit(main())
