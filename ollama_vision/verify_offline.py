#!/usr/bin/env python3
"""
Quick verification script to test offline capability.
This can be run without internet to verify everything works.
"""

import os
import sys
from rich.console import Console

console = Console()

def check_stable_diffusion():
    """Check if Stable Diffusion model is available locally."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'stable-diffusion-v1-4')
    config_file = os.path.join(model_path, 'model_index.json')
    
    if os.path.exists(config_file):
        console.print("[green]✓ Stable Diffusion model found locally[/green]")
        console.print(f"  Location: {model_path}")
        return True
    else:
        console.print("[red]✗ Stable Diffusion model NOT found[/red]")
        console.print(f"  Expected at: {model_path}")
        console.print("[yellow]  Run: python ollama_vision/download_models.py[/yellow]")
        return False

def check_ollama():
    """Check if Ollama models are available."""
    try:
        import ollama
        result = ollama.list()
        models = result.models if hasattr(result, 'models') else []
        model_names = [m.model if hasattr(m, 'model') else '' for m in models]
        
        if any('qwen3-vl' in name for name in model_names):
            console.print("[green]✓ Ollama qwen3-vl model found[/green]")
            return True
        else:
            console.print("[yellow]⚠ Ollama running but qwen3-vl not found[/yellow]")
            console.print("[yellow]  Run: ollama pull qwen3-vl[/yellow]")
            return False
    except Exception as e:
        console.print(f"[yellow]⚠ Ollama check failed: {e}[/yellow]")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    required = ['torch', 'diffusers', 'PIL', 'ollama', 'rich']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'PIL':
                __import__('PIL')
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        console.print("[green]✓ All Python dependencies installed[/green]")
        return True
    else:
        console.print(f"[red]✗ Missing packages: {', '.join(missing)}[/red]")
        console.print("[yellow]  Run: pip install -r ollama_vision/requirements.txt[/yellow]")
        return False

def check_output_dir():
    """Ensure output directory exists."""
    output_dir = 'ollama_vision/generated_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        console.print(f"[cyan]Created output directory: {output_dir}[/cyan]")
    else:
        console.print(f"[green]✓ Output directory exists: {output_dir}[/green]")
    return True

def main():
    console.print("[bold]Offline Capability Check[/bold]\n")
    
    checks = [
        ("Dependencies", check_dependencies()),
        ("Stable Diffusion Model", check_stable_diffusion()),
        ("Ollama Setup", check_ollama()),
        ("Output Directory", check_output_dir()),
    ]
    
    console.print("\n" + "="*60)
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        console.print("[bold green]✓ ALL CHECKS PASSED - Ready for offline use![/bold green]")
        console.print("\n[cyan]You can now run:[/cyan]")
        console.print("[white]  python ollama_vision/main.py[/white]")
    else:
        console.print("[bold yellow]⚠ SOME CHECKS FAILED[/bold yellow]")
        console.print("[yellow]Please resolve the issues above before running offline.[/yellow]")
    
    console.print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
