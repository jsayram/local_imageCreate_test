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
from config import MODEL_NAME, INFERENCE_STEPS, RANDOM_SEED, REALVISXL_CONFIG, REALVISXL_V4_CONFIG, SD_V14_CONFIG
from character_manager import CharacterManager
import glob

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
    if model_choice == "realvisxl" or model_choice == "realvisxl_v4":
        # Import SDXL pipeline only when needed (Apple Silicon only)
        from diffusers import StableDiffusionXLPipeline
        
        # RealVisXL model from Hugging Face - use config based on model choice
        if model_choice == "realvisxl_v4":
            realvisxl_model_id = REALVISXL_V4_CONFIG.get('model_id', 'SG161222/RealVisXL_V4.0')
        else:
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
    console.print("2. [yellow]RealVisXL V5.0[/yellow] - Realistic Vision XL (SDXL, latest)")
    console.print("3. [cyan]RealVisXL V4.0[/cyan] - Realistic Vision XL V4 (SDXL, stable)")
    console.print()
    
    while True:
        model_choice = input("Enter 1 for Default, 2 for V5.0, or 3 for V4.0: ").strip()
        if model_choice == "1":
            selected_model = "default"
            console.print("[green]Using Default model (SD v1.4)[/green]")
            break
        elif model_choice == "2":
            selected_model = "realvisxl"
            console.print("[yellow]Using RealVisXL V5.0 (SDXL)[/yellow]")
            break
        elif model_choice == "3":
            selected_model = "realvisxl_v4"
            console.print("[cyan]Using RealVisXL V4.0 (SDXL)[/cyan]")
            break
        else:
            console.print("[red]Please enter 1, 2, or 3.[/red]")
    
    console.print()
else:
    # Non-Apple Silicon: silently use default model
    selected_model = "default"

# ============================================================================
# Generation Settings Menu (with defaults from config)
# ============================================================================
console.print("[bold cyan]⚙️  Generation Settings[/bold cyan]")
console.print("[dim]Press Enter to use default values shown in [brackets][/dim]")
console.print()

# Get the appropriate config based on selected model
if selected_model == "realvisxl":
    current_config = REALVISXL_CONFIG
    model_label = "RealVisXL V5.0"
elif selected_model == "realvisxl_v4":
    current_config = REALVISXL_V4_CONFIG
    model_label = "RealVisXL V4.0"
else:
    current_config = SD_V14_CONFIG
    model_label = "SD v1.4"

# Display current config defaults
console.print(f"[dim]Current {model_label} config defaults:[/dim]")
console.print(f"[dim]  Guidance Scale: {current_config.get('guidance_scale', 7.5)} | Steps: {current_config.get('inference_steps', 50)} | Images: {current_config.get('num_images', 1)}[/dim]")
console.print()

# --- Guidance Scale ---
default_guidance = current_config.get('guidance_scale', 7.5)
guidance_input = input(f"🎯 Guidance Scale (1-20, how closely to follow prompt) [{default_guidance}]: ").strip()
if guidance_input:
    try:
        custom_guidance = max(1.0, min(20.0, float(guidance_input)))
    except ValueError:
        console.print(f"[yellow]Invalid value, using default: {default_guidance}[/yellow]")
        custom_guidance = default_guidance
else:
    custom_guidance = default_guidance

# --- Inference Steps ---
default_steps = current_config.get('inference_steps', 50)
steps_input = input(f"🔄 Inference Steps (10-150, more = better quality but slower) [{default_steps}]: ").strip()
if steps_input:
    try:
        custom_steps = max(10, min(150, int(steps_input)))
    except ValueError:
        console.print(f"[yellow]Invalid value, using default: {default_steps}[/yellow]")
        custom_steps = default_steps
else:
    custom_steps = default_steps

# --- Number of Images ---
default_num_images = current_config.get('num_images', 1)
num_images_input = input(f"🖼️  Number of Images (1-10) [{default_num_images}]: ").strip()
if num_images_input:
    try:
        custom_num_images = max(1, min(10, int(num_images_input)))
    except ValueError:
        console.print(f"[yellow]Invalid value, using default: {default_num_images}[/yellow]")
        custom_num_images = default_num_images
else:
    custom_num_images = default_num_images

# --- Character Consistency (fixed seed) ---
from config import CHARACTER_CONSISTENCY_CONFIG
default_consistency = True  # Default to enabled for consistent characters
consistency_default_str = "Y" if default_consistency else "N"
consistency_input = input(f"🎭 Character Consistency (use fixed seed for same character) [{'Y' if default_consistency else 'N'}]: ").strip().upper()
if consistency_input:
    use_character_consistency = consistency_input == 'Y'
else:
    use_character_consistency = default_consistency

# --- IP-Adapter Character Mode (SDXL only) ---
use_ip_adapter_mode = False
ip_adapter_ref_image = None
ip_adapter_scale = 0.8

if selected_model in ["realvisxl", "realvisxl_v4"]:
    console.print()
    console.print("[bold magenta]═══ IP-Adapter Character Mode ═══[/bold magenta]")
    console.print("[dim]Use a reference portrait for consistent character identity across scenes[/dim]")
    console.print("[dim]This is different from img2img - it locks facial features while giving full scene freedom[/dim]")
    ip_mode_input = input("Enable IP-Adapter character mode? (y/N): ").strip().lower()
    
    if ip_mode_input == 'y':
        # Look for reference images
        import glob
        ref_folder = "assets/characters"
        available_refs = glob.glob(os.path.join(ref_folder, "*.png")) + glob.glob(os.path.join(ref_folder, "*.jpg"))
        
        if available_refs:
            console.print("\n[green]Available reference portraits:[/green]")
            for idx, ref_path in enumerate(available_refs, 1):
                console.print(f"{idx}. {os.path.basename(ref_path)}")
            console.print()
            
            ref_choice = input("Select reference image (1-N) or enter custom path: ").strip()
            try:
                ref_idx = int(ref_choice) - 1
                if 0 <= ref_idx < len(available_refs):
                    ip_adapter_ref_image = available_refs[ref_idx]
                    console.print(f"[green]✓ Selected: {os.path.basename(ip_adapter_ref_image)}[/green]")
                    use_ip_adapter_mode = True
            except ValueError:
                if os.path.exists(ref_choice):
                    ip_adapter_ref_image = ref_choice
                    console.print(f"[green]✓ Loaded: {os.path.basename(ip_adapter_ref_image)}[/green]")
                    use_ip_adapter_mode = True
                else:
                    console.print("[yellow]Invalid path, IP-Adapter mode disabled[/yellow]")
        else:
            # Prompt for custom path
            ref_path = input(f"Enter path to reference portrait (or press Enter to skip): ").strip()
            if ref_path and os.path.exists(ref_path):
                ip_adapter_ref_image = ref_path
                console.print(f"[green]✓ Loaded: {os.path.basename(ip_adapter_ref_image)}[/green]")
                use_ip_adapter_mode = True
            else:
                console.print("[yellow]No reference image, IP-Adapter mode disabled[/yellow]")
        
        # If IP-Adapter enabled, ask for strength
        if use_ip_adapter_mode:
            console.print()
            console.print("[bold cyan]IP-Adapter Identity Strength:[/bold cyan]")
            console.print("[dim]Higher (0.8-1.0) = stronger identity lock, less variation[/dim]")
            console.print("[dim]Lower (0.5-0.7) = more creative freedom, weaker identity[/dim]")
            scale_input = input("Identity strength (0.5-1.0) [0.8]: ").strip()
            if scale_input:
                try:
                    ip_adapter_scale = max(0.5, min(1.0, float(scale_input)))
                except ValueError:
                    console.print("[yellow]Invalid value, using default 0.8[/yellow]")
            console.print(f"[dim]Using IP-Adapter scale: {ip_adapter_scale}[/dim]")

# Summary
console.print()
console.print("[bold green]✓ Settings configured:[/bold green]")
console.print(f"  Guidance Scale: {custom_guidance}")
console.print(f"  Inference Steps: {custom_steps}")
console.print(f"  Number of Images: {custom_num_images}")
console.print(f"  Character Consistency: {'Yes (fixed seed)' if use_character_consistency else 'No (random seed)'}")
if use_ip_adapter_mode:
    console.print(f"  [magenta]IP-Adapter Mode: Enabled (strength: {ip_adapter_scale})[/magenta]")
    console.print(f"  [magenta]Reference: {os.path.basename(ip_adapter_ref_image)}[/magenta]")
console.print()

# Initialize character manager
character_manager = CharacterManager()

# Character selection workflow
selected_character = None
character_id = None

console.print("[bold cyan]═══ Character Selection ═══[/bold cyan]")
console.print()

# List available characters
characters = character_manager.list_characters()
if characters:
    console.print("[green]Saved Characters:[/green]")
    for idx, char in enumerate(characters, 1):
        last_used = char.get('last_used', 'Never')
        times_used = char.get('times_used', 0)
        console.print(f"{idx}. {char['name']} - Used {times_used} times (Last: {last_used})")
    console.print()

# Prompt for character selection
console.print("Options:")
console.print("  [1-N] - Select saved character by number")
console.print("  [N]   - Create new character")
console.print("  [Enter] - Skip (no character consistency)")
character_choice = input("Choose option: ").strip()

if character_choice and character_choice.upper() != 'N':
    # Try to select existing character
    try:
        char_idx = int(character_choice) - 1
        if 0 <= char_idx < len(characters):
            selected_character = character_manager.get_character(characters[char_idx]['id'])
            character_id = characters[char_idx]['id']
            console.print(f"[green]✓ Loaded character: {selected_character['name']}[/green]")
            console.print(f"[dim]Description: {selected_character['description'][:80]}...[/dim]")
            console.print(f"[dim]Fixed seed: {selected_character['seed']}[/dim]")
            # Override settings with character's saved settings
            use_character_consistency = True
            custom_steps = selected_character.get('settings', {}).get('num_steps', custom_steps)
            custom_guidance = selected_character.get('settings', {}).get('guidance_scale', custom_guidance)
        else:
            console.print("[yellow]Invalid selection, proceeding without character[/yellow]")
    except ValueError:
        console.print("[yellow]Invalid input, proceeding without character[/yellow]")
elif character_choice.upper() == 'N':
    console.print("[cyan]Creating new character (will prompt after generation)[/cyan]")

console.print()

# Reference Image Selection (for img2img - maintaining facial features)
reference_image = None
img2img_strength = 0.75  # Default: keep 25% of original, change 75%

if selected_character:
    console.print("[bold cyan]═══ Reference Image (Optional) ═══[/bold cyan]")
    console.print("[dim]Use a reference image to maintain facial features while changing the scene[/dim]")
    console.print()
    
    # Check if character has a reference image saved
    char_ref_image = selected_character.get('reference_image')
    
    # Look for character's generation folders (determine based on selected model)
    if selected_model == "realvisxl":
        char_folder_base = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/')
    elif selected_model == "realvisxl_v4":
        char_folder_base = REALVISXL_V4_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl_v4/')
    else:
        char_folder_base = SD_V14_CONFIG.get('output_directory', 'ollama_vision/generated_images/sd_v14/')
    
    safe_char_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in selected_character['name'])
    safe_char_name = safe_char_name.strip().replace(' ', '_')
    char_folder = os.path.join(char_folder_base, safe_char_name)
    
    # Find all PNG images in character's folder
    available_images = []
    if os.path.exists(char_folder):
        available_images = sorted(glob.glob(os.path.join(char_folder, '**', '*.png'), recursive=True))
    
    if available_images:
        console.print("[green]Available reference images:[/green]")
        for idx, img_path in enumerate(available_images[:10], 1):  # Show first 10
            rel_path = os.path.relpath(img_path, char_folder_base)
            console.print(f"{idx}. {rel_path}")
        if len(available_images) > 10:
            console.print(f"[dim]... and {len(available_images) - 10} more[/dim]")
        console.print()
    elif char_ref_image and os.path.exists(char_ref_image):
        console.print(f"[green]Character has saved reference: {os.path.basename(char_ref_image)}[/green]")
        console.print()
    
    console.print("Options:")
    console.print("  [1-N]   - Select image by number")
    console.print("  [path]  - Enter custom image path")
    console.print("  [Enter] - Skip (text-to-image mode)")
    ref_choice = input("Choose reference image: ").strip()
    
    if ref_choice:
        try:
            # Try as number first
            ref_idx = int(ref_choice) - 1
            if 0 <= ref_idx < len(available_images):
                reference_image = Image.open(available_images[ref_idx]).convert('RGB')
                console.print(f"[green]✓ Loaded reference: {os.path.basename(available_images[ref_idx])}[/green]")
        except ValueError:
            # Try as file path
            if os.path.exists(ref_choice):
                reference_image = Image.open(ref_choice).convert('RGB')
                console.print(f"[green]✓ Loaded reference: {os.path.basename(ref_choice)}[/green]")
            else:
                console.print("[yellow]Invalid path, proceeding without reference[/yellow]")
    
    # If reference image selected, ask for strength
    if reference_image:
        console.print()
        console.print("[bold cyan]Image-to-Image Strength:[/bold cyan]")
        console.print("[dim]Lower = keep more of original face, Higher = more variation[/dim]")
        console.print("[dim]Recommended: 0.5-0.7 for same person, 0.7-0.85 for scene changes[/dim]")
        strength_input = input(f"Strength (0.1-0.95) [0.75]: ").strip()
        if strength_input:
            try:
                img2img_strength = max(0.1, min(0.95, float(strength_input)))
            except ValueError:
                console.print("[yellow]Invalid value, using default 0.75[/yellow]")
        console.print(f"[dim]Using strength: {img2img_strength}[/dim]")

console.print()

# Load model name from config
model_name = MODEL_NAME

# System prompt for generating detailed, hyper-realistic image prompts
system_prompt = SYSTEM_PROMPT

# Get image filename from user (assumed in images/ folder)
image_filename = input("Enter image filename in images/ folder (or leave blank for text-only): ").strip()

# Get prompt from user
if selected_character:
    console.print(f"[bold cyan]Character loaded: {selected_character['name']}[/bold cyan]")
    console.print("[dim]Tip: Describe the scene/pose for this character (e.g., 'standing in a forest', 'sitting at a desk')[/dim]")
    scene_prompt = input("Enter scene/pose description: ").strip()
    # Combine character description with scene
    prompt = f"{selected_character['description']}, {scene_prompt}"
    console.print(f"[dim]Full prompt: {prompt}[/dim]")
else:
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
            full_prompt = f"Analyze this image and create a prompt for: {prompt}"
        else:
            # Text mode: enhance the user's prompt - keep it simple
            full_prompt = prompt  # Just use the user's prompt directly
        
        response = ollama.generate(model=model_name, prompt=full_prompt, system=system_prompt, images=images)
    
    image_prompt = response['response'].strip()
    
    # Validate the response - ensure it has 2 lines
    lines = [line.strip() for line in image_prompt.split('\n') if line.strip() and not line.strip().startswith('-') and not line.strip()[0].isdigit()]
    
    if len(lines) < 2:
        console.print("[yellow]⚠ LLM returned invalid format, creating manual prompt...[/yellow]")
        # Fallback: create a proper dual-prompt manually
        image_prompt = f"{prompt}, high quality, detailed\n, RAW photo, shot on iphone pro, 85mm f/1.8, bokeh, film grain"
    else:
        # Use first 2 valid lines only
        image_prompt = f"{lines[0]}\n{lines[1]}"
    
    console.print("[bold blue]Optimized Image Prompt:[/bold blue]")
    console.print(f"[cyan]{image_prompt}[/cyan]")
except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    # Ultimate fallback
    image_prompt = f"{prompt}, high quality\n, RAW photo, shot on iphone pro, 85mm f/1.8, bokeh"

# Generate image from optimized prompt
console.print("[yellow]Generating image...[/yellow]")
try:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # === IP-ADAPTER CHARACTER MODE ===
    if use_ip_adapter_mode and selected_model in ["realvisxl", "realvisxl_v4"]:
        from character_pipeline import build_realvis_ip_adapter_pipeline, generate_scene_with_reference
        
        console.print("[bold magenta]🎭 Using IP-Adapter Character Mode[/bold magenta]")
        console.print("[dim]Building IP-Adapter pipeline...[/dim]")
        
        # Set pipeline type for metadata
        pipeline_type = "sdxl"
        
        # Determine model ID based on selection
        if selected_model == "realvisxl_v4":
            model_id = REALVISXL_V4_CONFIG.get('model_id', 'SG161222/RealVisXL_V4.0')
            config_used = REALVISXL_V4_CONFIG
        else:
            model_id = REALVISXL_CONFIG.get('model_id', 'SG161222/RealVisXL_V5.0')
            config_used = REALVISXL_CONFIG
        
        # Build IP-Adapter pipeline
        ip_pipe = build_realvis_ip_adapter_pipeline(
            model_id=model_id,
            ip_adapter_scale=ip_adapter_scale,
            device=device,
            dtype=dtype
        )
        
        # Use character consistency seed if enabled
        if use_character_consistency:
            if selected_character:
                base_seed = selected_character['seed']
                console.print(f"[dim]🎭 Using character seed: {base_seed}[/dim]")
            else:
                base_seed = CHARACTER_CONSISTENCY_CONFIG.get('fixed_seed', 42)
                console.print(f"[dim]🎭 Character Consistency ON - using fixed seed: {base_seed}[/dim]")
        else:
            base_seed = RANDOM_SEED
        
        # Generate multiple images with IP-Adapter
        generated_images = []
        for i in range(custom_num_images):
            current_seed = base_seed + i
            console.print(f"[dim]Generating image {i+1}/{custom_num_images} (seed: {current_seed})...[/dim]")
            
            # Use the scene prompt (without character description, IP-Adapter handles identity)
            scene_description = prompt if not selected_character else prompt.split(',', 1)[-1].strip() if ',' in prompt else prompt
            
            img = generate_scene_with_reference(
                pipe=ip_pipe,
                ref_image_path=ip_adapter_ref_image,
                scene_prompt=scene_description,
                seed=current_seed,
                steps=custom_steps,
                guidance_scale=custom_guidance,
                width=config_used.get('width', 1024),
                height=config_used.get('height', 1024),
                negative_prompt=config_used.get('negative_prompt', '')
            )
            generated_images.append(img)
        
        # Clean up
        del ip_pipe
        gc.collect()
        
    # === STANDARD TEXT2IMG / IMG2IMG MODE ===
    else:
        # Check if model exists BEFORE starting the spinner (so download prompt works)
        pipe, pipeline_type = get_pipeline(selected_model, device, dtype, offline_mode, models_dir)
    
        # If reference image provided, convert pipeline to img2img
        if reference_image and pipeline_type == "sdxl":
            from diffusers import StableDiffusionXLImg2ImgPipeline
            console.print("[dim]Switching to img2img mode for reference-based generation...[/dim]")
            
            # Convert text2img pipeline to img2img pipeline (reuses loaded weights)
            pipe = StableDiffusionXLImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
            ).to(device)
            
            # Resize reference image to match target resolution
            if selected_model == "realvisxl_v4":
                target_width = REALVISXL_V4_CONFIG.get('width', 1024)
                target_height = REALVISXL_V4_CONFIG.get('height', 1024)
            else:
                target_width = REALVISXL_CONFIG.get('width', 1024)
                target_height = REALVISXL_CONFIG.get('height', 1024)
            reference_image = reference_image.resize((target_width, target_height), Image.LANCZOS)
            console.print(f"[dim]Reference image resized to {target_width}x{target_height}[/dim]")
        
        with console.status("[bold cyan]Creating image...[/bold cyan]") as status:
            
            if pipeline_type == "sdxl":
                # RealVisXL parameters - use custom settings
                if selected_model == "realvisxl_v4":
                    console.print(f"[dim]Using RealVisXL V4.0: {REALVISXL_V4_CONFIG['width']}x{REALVISXL_V4_CONFIG['height']}, {custom_steps} steps, guidance {custom_guidance}, {custom_num_images} image(s)[/dim]")
                    config_used = REALVISXL_V4_CONFIG
                else:
                    console.print(f"[dim]Using RealVisXL V5.0: {REALVISXL_CONFIG['width']}x{REALVISXL_CONFIG['height']}, {custom_steps} steps, guidance {custom_guidance}, {custom_num_images} image(s)[/dim]")
                    config_used = REALVISXL_CONFIG
                
                # Parse dual prompts (prompt on line 1, prompt_2 on line 2)
                prompt_lines = image_prompt.strip().split('\\n')
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
                
                # Show parsed prompts (full text)
                console.print(f"[bold cyan]Parsed Dual Prompts:[/bold cyan]")
                console.print(f"[cyan]Line 1:[/cyan] {main_prompt}")
                if secondary_prompt:
                    console.print(f"[cyan]Line 2:[/cyan] {secondary_prompt}")
                else:
                    console.print(f"[yellow]⚠ No Line 2 found - using single prompt mode[/yellow]")
                
                # Parse negative prompts (support dual negative prompts too)
                neg_prompt = config_used.get('negative_prompt', '')
                
                # Use character consistency seed if enabled
                if use_character_consistency:
                    if selected_character:
                        # Use character's saved seed
                        base_seed = selected_character['seed']
                        console.print(f"[dim]🎭 Using character seed: {base_seed}[/dim]")
                    else:
                        base_seed = CHARACTER_CONSISTENCY_CONFIG.get('fixed_seed', 42)
                        console.print(f"[dim]🎭 Character Consistency ON - using fixed seed: {base_seed}[/dim]")
                else:
                    base_seed = RANDOM_SEED
                
                # Generate multiple images with different seeds
                generated_images = []
                for i in range(custom_num_images):
                    current_seed = base_seed + i  # Vary seed for each image
                    console.print(f"[dim]Generating image {i+1}/{custom_num_images} (seed: {current_seed})...[/dim]")
                    
                    if reference_image:
                        # img2img mode - use reference image
                        console.print(f"[dim]Using img2img with strength {img2img_strength} (keeping {int((1-img2img_strength)*100)}% of reference)[/dim]")
                        result = pipe(
                            prompt=main_prompt,
                            prompt_2=secondary_prompt if secondary_prompt else None,
                            image=reference_image,
                            strength=img2img_strength,
                            negative_prompt=neg_prompt,
                            negative_prompt_2=neg_prompt,
                            num_inference_steps=custom_steps,
                            guidance_scale=custom_guidance,
                            generator=torch.Generator(device).manual_seed(current_seed)
                        ).images[0]
                    else:
                        # text2img mode - generate from scratch
                        result = pipe(
                            prompt=main_prompt,
                            prompt_2=secondary_prompt if secondary_prompt else None,
                            negative_prompt=neg_prompt,
                            negative_prompt_2=neg_prompt,
                            num_inference_steps=custom_steps,
                            guidance_scale=custom_guidance,
                            width=config_used.get('width', 1024),
                            height=config_used.get('height', 1024),
                            generator=torch.Generator(device).manual_seed(current_seed)
                        ).images[0]
                    generated_images.append(result)
            else:
                # SD v1.4 parameters - use custom settings
                console.print(f"[dim]Using SD v1.4: {SD_V14_CONFIG['width']}x{SD_V14_CONFIG['height']}, {custom_steps} steps, guidance {custom_guidance}, {custom_num_images} image(s)[/dim]")
                
                # Use character consistency seed if enabled
                if use_character_consistency:
                    if selected_character:
                        # Use character's saved seed
                        base_seed = selected_character['seed']
                        console.print(f"[dim]🎭 Using character seed: {base_seed}[/dim]")
                    else:
                        base_seed = CHARACTER_CONSISTENCY_CONFIG.get('fixed_seed', 42)
                        console.print(f"[dim]🎭 Character Consistency ON - using fixed seed: {base_seed}[/dim]")
                else:
                    base_seed = RANDOM_SEED
                
                # Generate multiple images with different seeds
                generated_images = []
                for i in range(custom_num_images):
                    current_seed = base_seed + i
                    console.print(f"[dim]Generating image {i+1}/{custom_num_images} (seed: {current_seed})...[/dim]")
                    
                    result = pipe(
                        prompt=image_prompt,
                        negative_prompt=SD_V14_CONFIG.get('negative_prompt', ''),
                        num_inference_steps=custom_steps,
                        guidance_scale=custom_guidance,
                        width=SD_V14_CONFIG.get('width', 512),
                        height=SD_V14_CONFIG.get('height', 512),
                        generator=torch.Generator(device).manual_seed(current_seed)
                    ).images[0]
                    generated_images.append(result)
        
    # Get output directory from config based on pipeline type
    if pipeline_type == "sdxl":
        if selected_model == "realvisxl_v4":
            base_output_dir = REALVISXL_V4_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl_v4/')
            config_used = REALVISXL_V4_CONFIG
        else:
            base_output_dir = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/')
            config_used = REALVISXL_CONFIG
    else:
        base_output_dir = SD_V14_CONFIG.get('output_directory', 'ollama_vision/generated_images/sd_v14/')
        config_used = SD_V14_CONFIG
    
    # If using a saved character, organize into character-specific folder
    if selected_character:
        # Sanitize character name for folder name (remove special chars)
        safe_char_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in selected_character['name'])
        safe_char_name = safe_char_name.strip().replace(' ', '_')
        base_output_dir = os.path.join(base_output_dir, safe_char_name)
    
    # Save all generated images with prompt files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, img in enumerate(generated_images):
        if len(generated_images) > 1:
            image_name = f"generated_{timestamp}_{i+1}"
        else:
            image_name = f"generated_{timestamp}"
        
        # Create individual folder for this image
        image_folder = os.path.join(base_output_dir, image_name)
        os.makedirs(image_folder, exist_ok=True)
        
        # Save image
        filename = f"{image_name}.png"
        filepath = os.path.join(image_folder, filename)
        img.save(filepath)
        
        # Calculate seed used for this image
        if use_character_consistency:
            if selected_character:
                img_seed = selected_character['seed'] + i
            else:
                img_seed = CHARACTER_CONSISTENCY_CONFIG.get('fixed_seed', 42) + i
        else:
            img_seed = RANDOM_SEED + i
        
        # Save prompt info to .txt file
        prompt_filepath = os.path.join(image_folder, f"{image_name}.txt")
        with open(prompt_filepath, 'w') as f:
            f.write(f"=== Image Generation Details ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {pipeline_type.upper()}\n")
            if selected_character:
                f.write(f"Character: {selected_character['name']} (ID: {character_id})\n")
            if use_ip_adapter_mode and selected_model in ["realvisxl", "realvisxl_v4"]:
                f.write(f"IP-Adapter Mode: ENABLED (scale: {ip_adapter_scale})\n")
                f.write(f"Reference Image: {os.path.basename(ip_adapter_ref_image)}\n")
            f.write(f"\n=== Original Prompt ===\n")
            f.write(f"{prompt}\n\n")
            f.write(f"=== Optimized Prompt ===\n")
            f.write(f"{image_prompt}\n\n")
            if pipeline_type == "sdxl" and not (use_ip_adapter_mode and selected_model in ["realvisxl", "realvisxl_v4"]):
                # Only write dual prompts for standard SDXL mode (not IP-Adapter)
                f.write(f"=== SDXL Dual Prompts (Actual) ===\n")
                f.write(f"Line 1: {main_prompt}\n")
                if secondary_prompt:
                    f.write(f"Line 2: {secondary_prompt}\n")
            f.write(f"\n=== Generation Settings ===\n")
            f.write(f"Seed: {img_seed}\n")
            f.write(f"Inference Steps: {custom_steps}\n")
            f.write(f"Guidance Scale: {custom_guidance}\n")
            f.write(f"Width: {config_used.get('width', 512)}\n")
            f.write(f"Height: {config_used.get('height', 512)}\n")
            f.write(f"Character Consistency: {use_character_consistency}\n")
            if reference_image:
                f.write(f"Mode: Image-to-Image (Strength: {img2img_strength})\n")
                f.write(f"Reference: Used facial reference image\n")
            else:
                f.write(f"Mode: Text-to-Image\n")
            f.write(f"Negative Prompt: {config_used.get('negative_prompt', 'N/A')[:200]}...\n")
        
        console.print(f"[green]Generated image saved to {filepath}[/green]")
        console.print(f"[dim]Prompt saved to {prompt_filepath}[/dim]")
    
    console.print(f"[bold green]✓ {len(generated_images)} image(s) generated successfully![/bold green]")
    
    # Character save workflow (only if not already using a saved character)
    if not selected_character and character_choice and character_choice.upper() == 'N':
        console.print()
        console.print("[bold cyan]═══ Save as Character? ═══[/bold cyan]")
        save_char = input("Save this as a character for future use? (y/N): ").strip().lower()
        
        if save_char == 'y':
            char_name = input("Character name: ").strip()
            if char_name:
                # Use the seed from the first generated image
                char_seed = base_seed
                
                # Ask for reference image (optional)
                ref_image = None
                use_generated = input("Use first generated image as reference? (Y/n): ").strip().lower()
                if use_generated != 'n':
                    ref_image = filepath  # Use the last saved filepath
                
                # Save character
                try:
                    new_char_id = character_manager.save_character(
                        name=char_name,
                        description=image_prompt,  # Use the optimized prompt as description
                        seed=char_seed,
                        settings={
                            'num_steps': custom_steps,
                            'guidance_scale': custom_guidance,
                            'model': selected_model
                        },
                        reference_image=ref_image
                    )
                    console.print(f"[green]✓ Character '{char_name}' saved! (ID: {new_char_id})[/green]")
                    console.print(f"[dim]Seed: {char_seed}, Settings: {custom_steps} steps, guidance {custom_guidance}[/dim]")
                except Exception as char_err:
                    console.print(f"[red]Error saving character: {char_err}[/red]")
    
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