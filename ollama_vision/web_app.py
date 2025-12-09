"""
Simple Flask Web App for Image Generation
Run with: python ollama_vision/web_app.py
Open browser to: http://localhost:3300

Supports up to 5 concurrent jobs with queue system.
"""

import os
import sys
import platform
import uuid
import threading
from datetime import datetime
from collections import OrderedDict
from flask import Flask, render_template_string, request, send_file, jsonify
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, DDIMScheduler
from diffusers import AutoencoderKL
import ollama
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT
from config import MODEL_NAME, RANDOM_SEED, REALVISXL_CONFIG, CHARACTER_CONSISTENCY_CONFIG
from character_manager import CharacterManager

app = Flask(__name__)

# Initialize character manager
character_manager = CharacterManager()

# Global pipeline (loaded once)
pipe = None
device = None
is_offline_mode = False
pipeline_lock = threading.Lock()

# ============================================================================
# Job Queue System
# ============================================================================
MAX_CONCURRENT_JOBS = 5
jobs = OrderedDict()  # job_id -> job_info
jobs_lock = threading.Lock()
active_jobs_count = 0
active_jobs_lock = threading.Lock()

def create_job(prompt, optimize_prompt=True, character_consistency=False, settings=None):
    """Create a new job and return its ID.
    
    Args:
        prompt: The user's prompt text
        optimize_prompt: Whether to use Ollama to enhance the prompt
        character_consistency: Whether to use fixed seed for consistent characters
        settings: Optional dict with customizable generation settings:
            - guidance_scale: How closely to follow the prompt (1-20)
            - inference_steps: Number of denoising steps (10-150)
    """
    global active_jobs_count
    job_id = str(uuid.uuid4())[:8]
    
    # Merge custom settings with defaults from config
    job_settings = {
        'guidance_scale': REALVISXL_CONFIG.get('guidance_scale', 6.0),
        'inference_steps': REALVISXL_CONFIG.get('inference_steps', 45),
    }
    if settings:
        if 'guidance_scale' in settings:
            job_settings['guidance_scale'] = max(1.0, min(20.0, float(settings['guidance_scale'])))
        if 'inference_steps' in settings:
            job_settings['inference_steps'] = max(10, min(150, int(settings['inference_steps'])))
    
    with jobs_lock:
        # Calculate queue position
        pending_jobs = sum(1 for j in jobs.values() if j['status'] in ['queued', 'processing'])
        
        job_info = {
            'id': job_id,
            'prompt': prompt,
            'optimize_prompt': optimize_prompt,
            'character_consistency': character_consistency,
            'settings': job_settings,  # Custom generation settings
            'status': 'queued',  # queued, processing, completed, error
            'step': 0,
            'total': job_settings['inference_steps'],
            'stage': 'Waiting in queue',
            'queue_position': pending_jobs,
            'created_at': datetime.now().isoformat(),
            'filename': None,
            'optimized_prompt': None,
            'error': None
        }
        jobs[job_id] = job_info
        
        # Clean up old completed jobs (keep last 20)
        completed = [jid for jid, j in jobs.items() if j['status'] in ['completed', 'error']]
        if len(completed) > 20:
            for jid in completed[:-20]:
                del jobs[jid]
    
    return job_id

def update_job_progress(job_id, step, stage):
    """Update job progress."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]['step'] = step
            jobs[job_id]['stage'] = stage

def get_job_info(job_id):
    """Get job info by ID."""
    with jobs_lock:
        return jobs.get(job_id, None)

def get_all_jobs():
    """Get all jobs."""
    with jobs_lock:
        return list(jobs.values())

def process_job(job_id):
    """Process a single job in a background thread."""
    global pipe, device, active_jobs_count
    
    job = get_job_info(job_id)
    if not job:
        return
    
    try:
        with jobs_lock:
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['stage'] = 'Optimizing prompt'
            # Update queue positions for waiting jobs
            pos = 0
            for jid, j in jobs.items():
                if j['status'] == 'queued':
                    j['queue_position'] = pos
                    pos += 1
        
        user_prompt = job['prompt']
        should_optimize = job.get('optimize_prompt', True)
        print(f"[Job {job_id}] Processing: {user_prompt[:50]}... (optimize={should_optimize})")
        
        # Optimize prompt with Ollama (if enabled)
        if should_optimize:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=user_prompt,
                system=SYSTEM_PROMPT
            )
            optimized_prompt = response['response'].strip()
            
            # Validate response - filter out invalid lines (bullets, numbers, etc.)
            lines = [line.strip() for line in optimized_prompt.split('\n') 
                    if line.strip() and not line.strip().startswith('-') 
                    and not line.strip().startswith('*')
                    and not (line.strip()[0].isdigit() and line.strip()[1:3] in ['. ', ') '])]
            
            if len(lines) < 2:
                print(f"[Job {job_id}] Warning: Invalid LLM response, using fallback prompt")
                optimized_prompt = f"{user_prompt}, professional portrait, high quality, detailed\nstudio lighting, RAW photo, shot on Canon EOS R5, 85mm f/1.8, bokeh, film grain"
            else:
                # Use first 2 valid lines only
                optimized_prompt = f"{lines[0]}\n{lines[1]}"
        else:
            # Use the raw prompt as-is
            optimized_prompt = user_prompt
        
        with jobs_lock:
            jobs[job_id]['optimized_prompt'] = optimized_prompt
            jobs[job_id]['stage'] = 'Processing prompt'
        
        # Parse dual prompts
        prompt_lines = optimized_prompt.split('\n')
        main_prompt = prompt_lines[0].strip() if len(prompt_lines) > 0 else optimized_prompt
        secondary_prompt = prompt_lines[1].strip() if len(prompt_lines) > 1 else ""
        
        # Truncate if needed
        def truncate(p, max_words=55):
            words = p.split()
            return ' '.join(words[:max_words]) if len(words) > max_words else p
        
        main_prompt = truncate(main_prompt)
        secondary_prompt = truncate(secondary_prompt)
        
        # Create job-specific progress callback
        def job_progress_callback(pipe_obj, step, timestep, callback_kwargs):
            update_job_progress(job_id, step + 1, 'Generating')
            return callback_kwargs
        
        # Generate image (use lock since pipeline isn't thread-safe for MPS)
        with jobs_lock:
            jobs[job_id]['stage'] = 'Generating'
            jobs[job_id]['step'] = 0
        
        # Get job-specific settings (with fallback to config defaults)
        job_settings = job.get('settings', {})
        guidance_scale = job_settings.get('guidance_scale', REALVISXL_CONFIG.get('guidance_scale', 6.0))
        inference_steps = job_settings.get('inference_steps', REALVISXL_CONFIG.get('inference_steps', 45))
        
        with pipeline_lock:
            # Check if using a character
            character_info = job.get('character')
            if character_info:
                # Use character's saved seed
                seed = character_info['seed']
                print(f"[Job {job_id}] Using character '{character_info['name']}' with seed {seed}")
            elif job.get('character_consistency', False):
                # Use fixed seed for character consistency
                seed = CHARACTER_CONSISTENCY_CONFIG.get('fixed_seed', 42)
            else:
                seed = RANDOM_SEED + hash(job_id) % 10000  # Vary seed per job
            
            print(f"[Job {job_id}] Generating with: steps={inference_steps}, guidance={guidance_scale}, seed={seed}")
            
            result = pipe(
                prompt=main_prompt,
                prompt_2=secondary_prompt if secondary_prompt else None,
                negative_prompt=REALVISXL_CONFIG.get('negative_prompt', ''),
                negative_prompt_2=REALVISXL_CONFIG.get('negative_prompt', ''),
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                width=REALVISXL_CONFIG.get('width', 832),
                height=REALVISXL_CONFIG.get('height', 1216),
                generator=torch.Generator(device).manual_seed(seed),
                callback_on_step_end=job_progress_callback
            ).images[0]
        
        with jobs_lock:
            jobs[job_id]['stage'] = 'Saving'
        
        # Save image and prompt to individual folder
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_output_dir = os.path.join(workspace_root, REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/web_images/'))
        
        # If using a saved character, organize into character-specific folder
        if job.get('character'):
            # Sanitize character name for folder name (remove special chars)
            safe_char_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in job['character']['name'])
            safe_char_name = safe_char_name.strip().replace(' ', '_')
            base_output_dir = os.path.join(base_output_dir, safe_char_name)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_name = f"web_{job_id}_{timestamp}"
        
        # Create individual folder for this image
        image_folder = os.path.join(base_output_dir, image_name)
        os.makedirs(image_folder, exist_ok=True)
        
        # Save image
        filename = f"{image_name}.png"
        filepath = os.path.join(image_folder, filename)
        result.save(filepath)
        
        # Save prompt info to .txt file
        prompt_filepath = os.path.join(image_folder, f"{image_name}.txt")
        with open(prompt_filepath, 'w') as f:
            f.write(f"=== Image Generation Details ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Job ID: {job_id}\n")
            if job.get('character'):
                f.write(f"Character: {job['character']['name']} (ID: {job['character']['id']})\n")
            f.write(f"\n=== Original Prompt ===\n")
            f.write(f"{user_prompt}\n\n")
            f.write(f"=== Optimized Prompt ===\n")
            f.write(f"{optimized_prompt}\n\n")
            f.write(f"=== SDXL Dual Prompts (Actual) ===\n")
            f.write(f"Line 1: {main_prompt}\n")
            if secondary_prompt:
                f.write(f"Line 2: {secondary_prompt}\n")
            f.write(f"\n=== Generation Settings ===\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Inference Steps: {inference_steps}\n")
            f.write(f"Guidance Scale: {guidance_scale}\n")
            f.write(f"Width: {REALVISXL_CONFIG.get('width', 832)}\n")
            f.write(f"Height: {REALVISXL_CONFIG.get('height', 1216)}\n")
            f.write(f"Character Consistency: {job.get('character_consistency', False)}\n")
            f.write(f"Prompt Optimization: {job.get('optimize_prompt', True)}\n")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['stage'] = 'Complete'
            jobs[job_id]['filename'] = filename
            jobs[job_id]['step'] = jobs[job_id]['total']
        
        print(f"[Job {job_id}] Completed: {filename}")
        gc.collect()
        
    except Exception as e:
        print(f"[Job {job_id}] Error: {e}")
        with jobs_lock:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['stage'] = 'Error'
            jobs[job_id]['error'] = str(e)
    
    finally:
        with active_jobs_lock:
            active_jobs_count -= 1
        # Try to start next queued job
        start_next_job()

def start_next_job():
    """Start the next queued job if capacity allows."""
    global active_jobs_count
    
    with active_jobs_lock:
        if active_jobs_count >= MAX_CONCURRENT_JOBS:
            return
    
    with jobs_lock:
        for job_id, job in jobs.items():
            if job['status'] == 'queued':
                with active_jobs_lock:
                    if active_jobs_count >= MAX_CONCURRENT_JOBS:
                        return
                    active_jobs_count += 1
                
                # Start job in background thread
                thread = threading.Thread(target=process_job, args=(job_id,), daemon=True)
                thread.start()
                return

# Legacy progress tracking for backwards compatibility
progress_info = {
    'step': 0,
    'total': 45,
    'stage': 'Idle'
}

def progress_callback(pipe, step, timestep, callback_kwargs):
    """Callback to track generation progress."""
    global progress_info
    progress_info['step'] = step + 1
    progress_info['total'] = REALVISXL_CONFIG.get('inference_steps', 45)
    progress_info['stage'] = 'Generating'
    return callback_kwargs

# ============================================================================
# HTML Template
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Generator For the Warzone Squad</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1, h2 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #a0a0a0;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #333;
            border-radius: 10px;
            background: #0f0f23;
            color: #fff;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
        }
        textarea:focus {
            outline: none;
            border-color: #48dbfb;
        }
        button {
            width: 100%;
            padding: 15px 30px;
            font-size: 18px;
            font-weight: 600;
            color: #fff;
            background: linear-gradient(90deg, #ff6b6b, #ee5a24);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(238, 90, 36, 0.4);
        }
        button:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: #0f0f23;
            display: none;
        }
        .status.show { display: block; }
        .status.loading { color: #48dbfb; }
        .status.error { color: #ff6b6b; }
        .status.success { color: #2ecc71; }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #48dbfb;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .progress-container {
            margin-top: 15px;
            display: none;
        }
        .progress-container.show { display: block; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #0f0f23;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #333;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48dbfb, #2ecc71);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        .progress-text {
            text-align: center;
            margin-top: 8px;
            font-size: 14px;
            color: #888;
        }
        .step-info {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .result {
            margin-top: 30px;
            text-align: center;
            display: none;
        }
        .result.show { display: block; }
        .result img {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }
        .download-btn {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(90deg, #2ecc71, #27ae60);
            color: #fff;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        .download-btn:hover {
            transform: translateY(-2px);
        }
        .reset-btn {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(90deg, #6c757d, #495057);
            color: #fff;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            margin-left: 10px;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .reset-btn:hover {
            transform: translateY(-2px);
            background: linear-gradient(90deg, #5a6268, #3d4246);
        }
        .button-group {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .prompt-display {
            margin-top: 15px;
            padding: 10px;
            background: #0f0f23;
            border-radius: 8px;
            font-size: 12px;
            color: #888;
            word-break: break-word;
        }
        .config-info {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-bottom: 20px;
        }
        .mode-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 9px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .mode-offline {
            background: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.3);
        }
        .mode-online {
            background: rgba(72, 219, 251, 0.2);
            color: #48dbfb;
            border: 1px solid rgba(72, 219, 251, 0.3);
        }
        .footer-info {
            text-align: center;
            color: #444;
            font-size: 10px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #222;
        }
        /* Info Section Styles */
        .info-section {
            background: #0f0f23;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            font-size: 14px;
            line-height: 1.5;
        }
        .info-section h3 {
            color: #48dbfb;
            margin-bottom: 15px;
            font-size: 16px;
        }
        .info-section p {
            margin-bottom: 10px;
            color: #ccc;
        }
        .info-section ul {
            margin: 10px 0;
            padding-left: 20px;
            color: #aaa;
        }
        .info-section li {
            margin-bottom: 5px;
        }
        .info-section em {
            color: #feca57;
            font-style: normal;
        }
        .info-section details {
            margin-top: 15px;
        }
        .info-section summary {
            color: #48dbfb;
            cursor: pointer;
            font-weight: 600;
        }
        .info-section pre {
            background: #1a1a2e;
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            color: #888;
            margin: 8px 0;
            white-space: pre-wrap;
            word-break: break-word;
        }
        /* Queue Styles */
        .queue-section {
            margin-top: 30px;
            padding: 20px;
            background: #0f0f23;
            border-radius: 15px;
            border: 1px solid #222;
        }
        .queue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .queue-title {
            font-size: 14px;
            color: #888;
            font-weight: 600;
        }
        .queue-stats {
            font-size: 11px;
            color: #666;
        }
        .queue-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .queue-item {
            display: flex;
            align-items: center;
            padding: 12px;
            margin-bottom: 8px;
            background: #1a1a2e;
            border-radius: 8px;
            border-left: 3px solid #333;
        }
        .queue-item.processing { border-left-color: #48dbfb; }
        .queue-item.completed { border-left-color: #2ecc71; }
        .queue-item.error { border-left-color: #ff6b6b; }
        .queue-item.queued { border-left-color: #feca57; }
        .queue-item.current { background: #16213e; border: 1px solid #48dbfb; }
        .queue-item-info {
            flex: 1;
            min-width: 0;
        }
        .queue-item-prompt {
            font-size: 12px;
            color: #ccc;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 4px;
        }
        .queue-item-status {
            font-size: 10px;
            color: #666;
        }
        .queue-item-progress {
            width: 80px;
            margin-left: 15px;
        }
        .queue-item-progress .progress-bar {
            height: 6px;
        }
        .queue-item-actions {
            margin-left: 10px;
        }
        .queue-item-actions a {
            font-size: 11px;
            color: #48dbfb;
            text-decoration: none;
        }
        .queue-item-actions a:hover {
            text-decoration: underline;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 10px;
            flex-shrink: 0;
        }
        .status-dot.processing { background: #48dbfb; animation: pulse 1s infinite; }
        .status-dot.completed { background: #2ecc71; }
        .status-dot.error { background: #ff6b6b; }
        .status-dot.queued { background: #feca57; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .empty-queue {
            text-align: center;
            color: #444;
            font-size: 12px;
            padding: 20px;
        }
        .capacity-info {
            font-size: 10px;
            color: #555;
            text-align: center;
            margin-top: 10px;
        }
        /* Toggle Switch */
        .toggle-group {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
            padding: 10px 15px;
            background: #0f0f23;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .toggle-label {
            font-size: 14px;
            color: #a0a0a0;
        }
        .toggle-label small {
            display: block;
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #333;
            transition: 0.3s;
            border-radius: 26px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: #666;
            transition: 0.3s;
            border-radius: 50%;
        }
        .toggle-switch input:checked + .toggle-slider {
            background: linear-gradient(90deg, #48dbfb, #2ecc71);
        }
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(24px);
            background-color: #fff;
        }
        /* Advanced Settings Styles */
        .advanced-settings {
            margin-bottom: 20px;
            background: #0f0f23;
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }
        .advanced-settings summary {
            padding: 12px 15px;
            cursor: pointer;
            color: #48dbfb;
            font-weight: 600;
            font-size: 14px;
            user-select: none;
            transition: background 0.2s;
        }
        .advanced-settings summary:hover {
            background: #1a1a2e;
        }
        .advanced-settings[open] summary {
            border-bottom: 1px solid #333;
        }
        .settings-panel {
            padding: 15px;
        }
        .setting-group {
            margin-bottom: 20px;
        }
        .setting-group label {
            display: block;
            color: #a0a0a0;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .setting-group label span {
            color: #48dbfb;
            font-weight: 600;
        }
        .setting-group input[type="range"] {
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: #333;
            border-radius: 3px;
            outline: none;
        }
        .setting-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, #48dbfb, #2ecc71);
            border-radius: 50%;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .setting-group input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        .setting-help {
            display: block;
            font-size: 11px;
            color: #666;
            margin-top: 6px;
            line-height: 1.4;
        }
        .setting-help strong {
            color: #feca57;
        }
        .settings-info {
            margin-top: 15px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 6px;
            font-size: 11px;
            color: #888;
        }
        .settings-info p {
            margin: 5px 0;
        }
        .settings-info strong {
            color: #48dbfb;
        }
        /* Character Library Styles */
        .character-section {
            margin-bottom: 25px;
            background: #0f0f23;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
        }
        .character-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .character-header h3 {
            font-size: 16px;
            color: #48dbfb;
            margin: 0;
            background: none;
            -webkit-text-fill-color: #48dbfb;
        }
        .btn-refresh {
            background: #1a1a2e;
            border: 1px solid #333;
            color: #888;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .btn-refresh:hover {
            background: #16213e;
            color: #48dbfb;
            border-color: #48dbfb;
        }
        .character-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            max-height: 300px;
            overflow-y: auto;
        }
        .character-card {
            background: #1a1a2e;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }
        .character-card:hover {
            border-color: #48dbfb;
            transform: translateY(-2px);
        }
        .character-card.selected {
            border-color: #2ecc71;
            background: #16213e;
        }
        .character-card-img {
            width: 100%;
            height: 120px;
            background: #0f0f23;
            border-radius: 6px;
            margin-bottom: 8px;
            object-fit: cover;
        }
        .character-card-name {
            font-size: 13px;
            font-weight: 600;
            color: #ccc;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .character-card-meta {
            font-size: 10px;
            color: #666;
        }
        .character-card-actions {
            position: absolute;
            top: 8px;
            right: 8px;
            display: flex;
            gap: 4px;
        }
        .btn-delete {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid rgba(255, 107, 107, 0.3);
            color: #ff6b6b;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }
        .btn-delete:hover {
            background: rgba(255, 107, 107, 0.4);
        }
        .empty-characters {
            grid-column: 1 / -1;
            text-align: center;
            padding: 40px 20px;
            color: #444;
        }
        .empty-characters p {
            font-size: 14px;
            margin-bottom: 8px;
        }
        .empty-characters small {
            font-size: 11px;
            color: #333;
        }
        .selected-character {
            margin-top: 15px;
            padding: 10px 15px;
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid rgba(46, 204, 113, 0.3);
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #2ecc71;
        }
        .selected-character strong {
            color: #2ecc71;
        }
        .btn-clear {
            background: transparent;
            border: none;
            color: #ff6b6b;
            cursor: pointer;
            font-size: 16px;
            padding: 0 8px;
            transition: transform 0.2s;
        }
        .btn-clear:hover {
            transform: scale(1.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 AI Image Generator<br> 
        <h2>Warzone Squad</h2>
        
        <!-- Info Section -->
        <div class="info-section">
            <h3>ℹ️ How It Works</h3>
            <p><strong>Workflow:</strong> Your prompt → <em>Ollama LLM optimizes it</em> → <em>RealVisXL generates image</em></p>
            <p><strong>Prompt Input Tips:</strong></p>
            <ul>
                <li>Keep it simple and clear - the AI will enhance it automatically</li>
                <li>Focus on subject, style, and mood (e.g., "cyberpunk city at night, neon lights")</li>
                <li>The optimizer creates detailed prompts optimized for RealVisXL's dual-prompt system</li>
                <li>Advanced: RealVisXL uses 2 prompts (subject + lighting/camera details) for best results</li>
            </ul>
            <details>
                <summary>Advanced Prompt Format (Optional)</summary>
                <p>If you disable optimization, you can manually provide dual prompts:</p>
                <pre>happy woman walking in a field of flowers
RAW photo, shot on Sony Alpha a7 III, 85mm, f/1.8, bokeh, sunny day, film grain</pre>
                <p>Line 1: Subject and composition | Line 2: Lighting, camera, and quality settings</p>
            </details>
        </div>
        
        <!-- Character Library Section -->
        <div class="character-section" id="characterSection">
            <div class="character-header">
                <h3>🎭 Character Library</h3>
                <button type="button" class="btn-refresh" onclick="loadCharacters()">🔄</button>
            </div>
            <div class="character-list" id="characterList">
                <div class="empty-characters">
                    <p>No saved characters yet</p>
                    <small>Generate an image and save it as a character to reuse with different poses/scenes</small>
                </div>
            </div>
            <div class="selected-character" id="selectedCharacterInfo" style="display: none;">
                <strong>Selected:</strong> <span id="selectedCharacterName"></span>
                <button type="button" class="btn-clear" onclick="clearCharacterSelection()">✕</button>
            </div>
        </div>
        
        <p class="config-info">
            RealVisXL V5.0 | {{ width }}x{{ height }} | {{ steps }} steps
            <span class="mode-badge {{ 'mode-offline' if is_offline else 'mode-online' }}">
                {{ '🔒 Offline' if is_offline else '🌐 Online' }}
            </span>
        </p>
        
        <form id="generateForm">
            <div class="form-group">
                <label for="prompt">Enter your prompt:</label>
                <textarea id="prompt" name="prompt" placeholder="e.g., beautiful sunset over mountains, professional photo"></textarea>
            </div>
            <div class="toggle-group">
                <div class="toggle-label">
                    🤖 AI Prompt Optimizer
                    <small>Enhance your prompt with Ollama for better results</small>
                </div>
                <label class="toggle-switch">
                    <input type="checkbox" id="optimizeToggle">
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="toggle-group">
                <div class="toggle-label">
                    🎭 Character Consistency
                    <small>Use fixed seed for consistent character generation</small>
                </div>
                <label class="toggle-switch">
                    <input type="checkbox" id="characterConsistencyToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            
            <!-- Advanced Settings Panel -->
            <details class="advanced-settings">
                <summary>⚙️ Advanced Settings</summary>
                <div class="settings-panel">
                    <div class="setting-group">
                        <label for="guidanceScale">
                            🎯 Guidance Scale: <span id="guidanceValue">{{ guidance_scale }}</span>
                        </label>
                        <input type="range" id="guidanceScale" min="1" max="20" step="0.5" value="{{ guidance_scale }}">
                        <small class="setting-help">
                            How closely the AI follows your prompt. Lower (1-5) = more creative/abstract. 
                            Higher (10-20) = stricter adherence but may look artificial. <strong>Recommended: 4-8</strong>
                        </small>
                    </div>
                    <div class="setting-group">
                        <label for="inferenceSteps">
                            🔄 Inference Steps: <span id="stepsValue">{{ steps }}</span>
                        </label>
                        <input type="range" id="inferenceSteps" min="10" max="150" step="5" value="{{ steps }}">
                        <small class="setting-help">
                            More steps = higher quality but slower. 20-30 = fast drafts. 
                            50-80 = good quality. 100+ = maximum detail. <strong>Default: {{ steps }}</strong>
                        </small>
                    </div>
                    <div class="settings-info">
                        <p>📊 <strong>Current Config:</strong> {{ scheduler }} scheduler, {{ 'Custom VAE' if vae_model else 'Default VAE' }}</p>
                        <p>💡 These settings only affect this generation. Config defaults are used for new sessions.</p>
                    </div>
                </div>
            </details>
            
            <button type="submit" id="submitBtn">✨ Generate Image</button>
        </form>
        
        <div class="status" id="status">
            <span class="spinner"></span>
            <span id="statusText">Generating image...</span>
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">0%</div>
                <div class="step-info" id="stepInfo">Step 0 / {{ steps }}</div>
            </div>
        </div>
        
        <div class="result" id="result">
            <img id="generatedImage" src="" alt="Generated Image">
            <br>
            <div class="button-group">
                <a id="downloadLink" href="" download class="download-btn">⬇️ Download Image</a>
                <button type="button" id="resetBtn" class="reset-btn">🔄 New Image</button>
            </div>
            <div class="prompt-display" id="optimizedPrompt"></div>
        </div>
        
        <!-- Job Queue Section -->
        <div class="queue-section" id="queueSection">
            <div class="queue-header">
                <span class="queue-title">📋 Job Queue</span>
                <span class="queue-stats" id="queueStats">0 active • 0 queued</span>
            </div>
            <div class="queue-list" id="queueList">
                <div class="empty-queue">No jobs in queue</div>
            </div>
            <div class="capacity-info">Max 5 concurrent jobs • Jobs process sequentially on GPU</div>
        </div>
        
        <div class="footer-info">
            {{ 'Local models • No internet required • Data stays on device' if is_offline else 'Using online models • Internet connection required' }}
        </div>
    </div>
    
    <script>
        const form = document.getElementById('generateForm');
        const submitBtn = document.getElementById('submitBtn');
        const status = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const result = document.getElementById('result');
        const generatedImage = document.getElementById('generatedImage');
        const downloadLink = document.getElementById('downloadLink');
        const optimizedPrompt = document.getElementById('optimizedPrompt');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const stepInfo = document.getElementById('stepInfo');
        const resetBtn = document.getElementById('resetBtn');
        const queueList = document.getElementById('queueList');
        const queueStats = document.getElementById('queueStats');
        const guidanceScale = document.getElementById('guidanceScale');
        const inferenceSteps = document.getElementById('inferenceSteps');
        const guidanceValue = document.getElementById('guidanceValue');
        const stepsValue = document.getElementById('stepsValue');
        
        let currentJobId = null;
        let queuePollInterval = null;
        let selectedCharacterId = null;
        
        // Character management functions
        async function loadCharacters() {
            try {
                const response = await fetch('/api/characters');
                const data = await response.json();
                
                if (data.success) {
                    renderCharacters(data.characters);
                }
            } catch (error) {
                console.error('Failed to load characters:', error);
            }
        }
        
        function renderCharacters(characters) {
            const characterList = document.getElementById('characterList');
            
            if (characters.length === 0) {
                characterList.innerHTML = `
                    <div class="empty-characters">
                        <p>No saved characters yet</p>
                        <small>Generate an image and save it as a character to reuse with different poses/scenes</small>
                    </div>
                `;
                return;
            }
            
            characterList.innerHTML = characters.map(char => `
                <div class="character-card ${selectedCharacterId === char.id ? 'selected' : ''}" 
                     onclick="selectCharacter('${char.id}', '${char.name}')">
                    ${char.reference_image ? 
                        `<img class="character-card-img" src="${char.reference_image}" alt="${char.name}">` :
                        `<div class="character-card-img"></div>`
                    }
                    <div class="character-card-name">${char.name}</div>
                    <div class="character-card-meta">Used ${char.times_used || 0} times</div>
                    <div class="character-card-actions">
                        <button class="btn-delete" onclick="deleteCharacter(event, '${char.id}')">🗑️</button>
                    </div>
                </div>
            `).join('');
        }
        
        function selectCharacter(characterId, characterName) {
            selectedCharacterId = characterId;
            
            // Update UI
            document.querySelectorAll('.character-card').forEach(card => {
                card.classList.remove('selected');
            });
            event.currentTarget.classList.add('selected');
            
            // Show selected character info
            document.getElementById('selectedCharacterName').textContent = characterName;
            document.getElementById('selectedCharacterInfo').style.display = 'flex';
            
            // Enable character consistency toggle automatically
            document.getElementById('characterConsistencyToggle').checked = true;
        }
        
        function clearCharacterSelection() {
            selectedCharacterId = null;
            document.querySelectorAll('.character-card').forEach(card => {
                card.classList.remove('selected');
            });
            document.getElementById('selectedCharacterInfo').style.display = 'none';
        }
        
        async function deleteCharacter(event, characterId) {
            event.stopPropagation(); // Prevent card selection
            
            if (!confirm('Delete this character? This cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch(`/api/characters/${characterId}`, {
                    method: 'DELETE'
                });
                const data = await response.json();
                
                if (data.success) {
                    if (selectedCharacterId === characterId) {
                        clearCharacterSelection();
                    }
                    loadCharacters(); // Reload list
                } else {
                    alert('Failed to delete character: ' + data.error);
                }
            } catch (error) {
                console.error('Failed to delete character:', error);
                alert('Failed to delete character');
            }
        }
        
        // Load characters on page load
        loadCharacters();
        
        // Update slider display values
        guidanceScale.addEventListener('input', () => {
            guidanceValue.textContent = guidanceScale.value;
        });
        inferenceSteps.addEventListener('input', () => {
            stepsValue.textContent = inferenceSteps.value;
        });
        
        function resetForm() {
            document.getElementById('prompt').value = '';
            result.className = 'result';
            status.className = 'status';
            progressContainer.className = 'progress-container';
            progressFill.style.width = '0%';
            progressText.textContent = '0%';
            stepInfo.textContent = 'Step 0 / {{ steps }}';
            generatedImage.src = '';
            downloadLink.href = '';
            optimizedPrompt.textContent = '';
            currentJobId = null;
            document.getElementById('prompt').focus();
        }
        
        resetBtn.addEventListener('click', resetForm);
        
        function updateProgress(step, total, stage) {
            const percent = Math.round((step / total) * 100);
            progressFill.style.width = percent + '%';
            progressText.textContent = percent + '%';
            stepInfo.textContent = stage + ' - Step ' + step + ' / ' + total;
        }
        
        function renderQueue(jobs) {
            const activeCount = jobs.filter(j => j.status === 'processing').length;
            const queuedCount = jobs.filter(j => j.status === 'queued').length;
            queueStats.textContent = `${activeCount} active • ${queuedCount} queued`;
            
            // Filter to show only recent jobs (last 10)
            const recentJobs = jobs.slice(-10).reverse();
            
            if (recentJobs.length === 0) {
                queueList.innerHTML = '<div class="empty-queue">No jobs in queue</div>';
                return;
            }
            
            queueList.innerHTML = recentJobs.map(job => {
                const isCurrent = job.id === currentJobId;
                const percent = job.total > 0 ? Math.round((job.step / job.total) * 100) : 0;
                const promptPreview = job.prompt.length > 40 ? job.prompt.substring(0, 40) + '...' : job.prompt;
                
                let statusText = '';
                if (job.status === 'queued') statusText = `Position ${job.queue_position + 1} in queue`;
                else if (job.status === 'processing') statusText = `${job.stage} - ${percent}%`;
                else if (job.status === 'completed') statusText = 'Completed';
                else if (job.status === 'error') statusText = 'Error: ' + (job.error || 'Unknown');
                
                let actions = '';
                if (job.status === 'completed' && job.filename) {
                    actions = `<a href="/download/${job.filename}" download>Download</a>`;
                }
                
                return `
                    <div class="queue-item ${job.status} ${isCurrent ? 'current' : ''}" data-job-id="${job.id}">
                        <div class="status-dot ${job.status}"></div>
                        <div class="queue-item-info">
                            <div class="queue-item-prompt">${promptPreview}</div>
                            <div class="queue-item-status">${statusText}</div>
                        </div>
                        ${job.status === 'processing' ? `
                            <div class="queue-item-progress">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${percent}%"></div>
                                </div>
                            </div>
                        ` : ''}
                        <div class="queue-item-actions">${actions}</div>
                    </div>
                `;
            }).join('');
            
            // Add click handlers for completed jobs
            queueList.querySelectorAll('.queue-item.completed').forEach(item => {
                item.style.cursor = 'pointer';
                item.addEventListener('click', (e) => {
                    if (e.target.tagName === 'A') return;
                    const jobId = item.dataset.jobId;
                    const job = jobs.find(j => j.id === jobId);
                    if (job && job.filename) {
                        showJobResult(job);
                    }
                });
            });
        }
        
        function showJobResult(job) {
            generatedImage.src = '/image/' + job.filename;
            downloadLink.href = '/download/' + job.filename;
            optimizedPrompt.textContent = 'Optimized prompt: ' + (job.optimized_prompt || job.prompt);
            result.className = 'result show';
            currentJobId = job.id;
        }
        
        async function pollQueue() {
            try {
                const res = await fetch('/queue');
                const data = await res.json();
                renderQueue(data.jobs);
                
                // Update current job progress if we have one
                if (currentJobId) {
                    const currentJob = data.jobs.find(j => j.id === currentJobId);
                    if (currentJob) {
                        if (currentJob.status === 'processing') {
                            updateProgress(currentJob.step, currentJob.total, currentJob.stage);
                        } else if (currentJob.status === 'completed') {
                            progressFill.style.width = '100%';
                            progressText.textContent = '100%';
                            stepInfo.textContent = 'Complete!';
                            status.className = 'status show success';
                            statusText.textContent = '✓ Image generated successfully!';
                            showJobResult(currentJob);
                        } else if (currentJob.status === 'error') {
                            status.className = 'status show error';
                            statusText.textContent = '✗ Error: ' + currentJob.error;
                            progressContainer.className = 'progress-container';
                        }
                    }
                }
            } catch (e) {
                console.error('Queue poll error:', e);
            }
        }
        
        // Start queue polling
        queuePollInterval = setInterval(pollQueue, 1000);
        pollQueue(); // Initial poll
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }
            
            // Reset progress
            progressFill.style.width = '0%';
            progressText.textContent = '0%';
            stepInfo.textContent = 'Submitting job...';
            
            // Show loading state
            submitBtn.disabled = true;
            status.className = 'status show loading';
            statusText.textContent = 'Adding to queue...';
            progressContainer.className = 'progress-container show';
            result.className = 'result';
            
            try {
                const optimizePrompt = document.getElementById('optimizeToggle').checked;
                const characterConsistency = document.getElementById('characterConsistencyToggle').checked;
                const guidanceScaleVal = parseFloat(document.getElementById('guidanceScale').value);
                const inferenceStepsVal = parseInt(document.getElementById('inferenceSteps').value);
                
                const requestBody = { 
                    prompt, 
                    optimize_prompt: optimizePrompt, 
                    character_consistency: characterConsistency,
                    guidance_scale: guidanceScaleVal,
                    inference_steps: inferenceStepsVal
                };
                
                // Include character_id if a character is selected
                if (selectedCharacterId) {
                    requestBody.character_id = selectedCharacterId;
                }
                
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Store current job ID and wait for completion via polling
                currentJobId = data.job_id;
                statusText.textContent = data.message;
                stepInfo.textContent = 'Waiting in queue...';
                
                // Clear prompt for next submission
                document.getElementById('prompt').value = '';
                
            } catch (error) {
                status.className = 'status show error';
                statusText.textContent = '✗ Error: ' + error.message;
                progressContainer.className = 'progress-container';
                currentJobId = null;
            } finally {
                submitBtn.disabled = false;
            }
        });
        
        // Navigation warning when user has pending/processing jobs or unsaved images
        let hasUnsavedWork = false;
        
        function updateUnsavedState() {
            // Check if there's a current job processing or a result showing
            const hasActiveJob = currentJobId !== null;
            const hasVisibleResult = result.classList.contains('show');
            hasUnsavedWork = hasActiveJob || hasVisibleResult;
        }
        
        // Update unsaved state periodically
        setInterval(updateUnsavedState, 1000);
        
        // Warn before leaving page
        window.addEventListener('beforeunload', (e) => {
            updateUnsavedState();
            if (hasUnsavedWork) {
                e.preventDefault();
                e.returnValue = 'You have generated images that may be lost. Are you sure you want to leave?';
                return e.returnValue;
            }
        });
    </script>
</body>
</html>
"""

# ============================================================================
# Load Pipeline
# ============================================================================

# Scheduler mapping: Different schedulers affect generation speed and quality
# - DPMSolverMultistepScheduler: Fast, high-quality (recommended for speed)
# - EulerDiscreteScheduler: Default, balanced quality
# - HeunDiscreteScheduler: Higher quality, slower
# - DDIMScheduler: Classic, deterministic results
SCHEDULER_MAP = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
    "DDIMScheduler": DDIMScheduler,
}

def load_pipeline():
    global pipe, device
    
    if pipe is not None:
        return pipe
    
    print("Loading RealVisXL V5.0 pipeline...")
    
    # Determine device
    if sys.platform == "darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    # Load from local models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    local_path = os.path.join(models_dir, 'RealVisXL_V5.0')
    
    # =========================================================================
    # VAE Model: Variational Auto-Encoder
    # =========================================================================
    # The VAE encodes/decodes images to/from latent space.
    # A better VAE = sharper details, better colors, less artifacts.
    # "stabilityai/sd-vae-ft-mse" is fine-tuned for better reconstruction.
    vae = None
    vae_loaded_from = None
    vae_model = REALVISXL_CONFIG.get('vae_model')
    if vae_model:
        # Check if VAE is already cached locally
        vae_cache_path = os.path.join(models_dir, f"models--{vae_model.replace('/', '--')}")
        vae_local_path = os.path.join(models_dir, vae_model.replace('/', '_'))
        
        if os.path.exists(vae_cache_path) or os.path.exists(vae_local_path):
            print(f"Loading custom VAE from local cache: {vae_model}")
            vae_loaded_from = "local"
        else:
            print(f"Downloading custom VAE: {vae_model} (~335MB)...")
            vae_loaded_from = "download"
        
        try:
            vae = AutoencoderKL.from_pretrained(
                vae_model,
                torch_dtype=dtype,
                cache_dir=models_dir
            )
            if vae_loaded_from == "download":
                print(f"✓ VAE downloaded and cached successfully")
            else:
                print(f"✓ VAE loaded from local cache")
        except Exception as e:
            print(f"Warning: Could not load custom VAE: {e}")
            print("Falling back to default VAE")
            vae = None
            vae_loaded_from = None
    
    global is_offline_mode
    if os.path.exists(local_path):
        print(f"Loading from local: {local_path}")
        is_offline_mode = True
        pipe = StableDiffusionXLPipeline.from_pretrained(
            local_path,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
            vae=vae  # Use custom VAE if loaded
        )
    else:
        print("Downloading RealVisXL V5.0...")
        is_offline_mode = False
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0",
            torch_dtype=dtype,
            cache_dir=models_dir,
            use_safetensors=True,
            vae=vae  # Use custom VAE if loaded
        )
        pipe.save_pretrained(local_path)
    
    # =========================================================================
    # Scheduler: Controls the denoising/sampling process
    # =========================================================================
    # Different schedulers trade off speed vs quality:
    # - DPMSolverMultistepScheduler: 20-30% faster, excellent quality
    # - EulerDiscreteScheduler: Default, balanced
    # - HeunDiscreteScheduler: Better quality, 2x slower
    # - DDIMScheduler: Classic, fully deterministic
    scheduler_name = REALVISXL_CONFIG.get('scheduler', 'EulerDiscreteScheduler')
    if scheduler_name in SCHEDULER_MAP:
        print(f"Using scheduler: {scheduler_name}")
        pipe.scheduler = SCHEDULER_MAP[scheduler_name].from_config(pipe.scheduler.config)
    else:
        print(f"Unknown scheduler '{scheduler_name}', using default")
    
    pipe = pipe.to(device)
    print(f"Pipeline loaded on {device}")
    
    # Log configuration summary
    print(f"  - Scheduler: {scheduler_name}")
    print(f"  - Custom VAE: {'Yes' if vae else 'No (using default)'}")
    print(f"  - Use Refiner: {REALVISXL_CONFIG.get('use_refiner', False)}")
    
    return pipe

# ============================================================================
# Routes
# ============================================================================
@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        width=REALVISXL_CONFIG.get('width', 832),
        height=REALVISXL_CONFIG.get('height', 1216),
        steps=REALVISXL_CONFIG.get('inference_steps', 45),
        guidance_scale=REALVISXL_CONFIG.get('guidance_scale', 6.0),
        scheduler=REALVISXL_CONFIG.get('scheduler', 'EulerDiscreteScheduler'),
        vae_model=REALVISXL_CONFIG.get('vae_model'),
        is_offline=is_offline_mode
    )

@app.route('/progress')
def get_progress():
    """Return current generation progress."""
    global progress_info
    return jsonify(progress_info)

@app.route('/queue')
def get_queue():
    """Return all jobs in the queue."""
    return jsonify({
        'jobs': get_all_jobs(),
        'active_count': sum(1 for j in get_all_jobs() if j['status'] == 'processing'),
        'queued_count': sum(1 for j in get_all_jobs() if j['status'] == 'queued'),
        'max_concurrent': MAX_CONCURRENT_JOBS
    })

@app.route('/job/<job_id>')
def get_job(job_id):
    """Return a specific job's info."""
    job = get_job_info(job_id)
    if job:
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404

@app.route('/generate', methods=['POST'])
def generate():
    """Add a new job to the queue."""
    global active_jobs_count
    
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '').strip()
        optimize_prompt = data.get('optimize_prompt', True)
        character_consistency = data.get('character_consistency', False)
        character_id = data.get('character_id')  # Optional character ID
        
        # If character_id is provided, load character and combine prompts
        if character_id:
            try:
                character = character_manager.get_character(character_id)
                if character:
                    # Use character's description as base, append user's scene description
                    user_prompt = character_manager.generate_character_prompt(
                        character_id, 
                        user_prompt
                    )
                    character_consistency = True  # Force character consistency on
                    print(f"[Character] Using character '{character['name']}' with seed {character['seed']}")
                else:
                    print(f"[Warning] Character {character_id} not found, proceeding without character")
            except Exception as char_err:
                print(f"[Error] Failed to load character: {char_err}")
        
        # Extract custom generation settings
        settings = {}
        if 'guidance_scale' in data:
            settings['guidance_scale'] = data['guidance_scale']
        if 'inference_steps' in data:
            settings['inference_steps'] = data['inference_steps']
        
        # If using a character, override settings with character's saved settings
        if character_id and character:
            char_settings = character.get('settings', {})
            if 'guidance_scale' in char_settings:
                settings['guidance_scale'] = char_settings['guidance_scale']
            if 'num_steps' in char_settings:
                settings['inference_steps'] = char_settings['num_steps']
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Create job with custom settings and character info
        job_id = create_job(
            user_prompt, 
            optimize_prompt=optimize_prompt,
            character_consistency=character_consistency,
            settings=settings if settings else None
        )
        
        # Store character info in job for seed override
        if character_id and character:
            with jobs_lock:
                jobs[job_id]['character'] = {
                    'id': character_id,
                    'name': character['name'],
                    'seed': character['seed']
                }
        
        print(f"[Job {job_id}] Created for prompt: {user_prompt[:50]}...")
        
        # Try to start the job immediately if capacity allows
        start_next_job()
        
        # Get queue position
        job = get_job_info(job_id)
        queue_pos = job.get('queue_position', 0) if job else 0
        
        message = 'Processing...' if job and job['status'] == 'processing' else f'Queued (position {queue_pos + 1})'
        
        return jsonify({
            'job_id': job_id,
            'message': message,
            'queue_position': queue_pos
        })
        
    except Exception as e:
        print(f"Error creating job: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve generated image from its folder."""
    output_dir = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/')
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # New structure: images are in individual folders
    # filename format: web_jobid_timestamp.png
    # folder name: web_jobid_timestamp (same as filename without extension)
    folder_name = filename.rsplit('.', 1)[0]  # Remove .png extension
    filepath = os.path.join(workspace_root, output_dir, folder_name, filename)
    
    if not os.path.exists(filepath):
        # Fallback: try old flat structure
        filepath = os.path.join(workspace_root, output_dir, filename)
    if not os.path.exists(filepath):
        # Try relative to script directory
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_images', 'realvisxl', filename)
    
    return send_file(filepath, mimetype='image/png')

@app.route('/download/<filename>')
def download_image(filename):
    """Download generated image from its folder."""
    output_dir = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/')
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # New structure: images are in individual folders
    folder_name = filename.rsplit('.', 1)[0]  # Remove .png extension
    filepath = os.path.join(workspace_root, output_dir, folder_name, filename)
    
    if not os.path.exists(filepath):
        # Fallback: try old flat structure
        filepath = os.path.join(workspace_root, output_dir, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_images', 'realvisxl', filename)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

# ============================================================================
# Character Management API
# ============================================================================

@app.route('/api/characters', methods=['GET'])
def get_characters():
    """Get list of all saved characters."""
    try:
        characters = character_manager.list_characters()
        return jsonify({
            'success': True,
            'characters': characters
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/characters/<character_id>', methods=['GET'])
def get_character(character_id):
    """Get specific character by ID."""
    try:
        character = character_manager.get_character(character_id)
        if character:
            return jsonify({
                'success': True,
                'character': character
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Character not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/characters', methods=['POST'])
def create_character():
    """Create a new character."""
    try:
        data = request.json
        character_id = character_manager.save_character(
            name=data['name'],
            description=data['description'],
            seed=data.get('seed', RANDOM_SEED),
            settings=data.get('settings', {}),
            reference_image=data.get('reference_image')
        )
        return jsonify({
            'success': True,
            'character_id': character_id,
            'message': f"Character '{data['name']}' created successfully"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/characters/<character_id>', methods=['DELETE'])
def delete_character(character_id):
    """Delete a character."""
    try:
        success = character_manager.delete_character(character_id)
        if success:
            return jsonify({
                'success': True,
                'message': 'Character deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Character not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("🎨 AI Image Generator Web App")
    print("=" * 50)
    print(f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}")
    print("Loading model on startup...")
    load_pipeline()
    print("")
    print("Starting Flask server...")
    print("Open browser to: http://localhost:3300")
    print("=" * 50)
    # Use threaded=True to handle multiple connections for queue polling
    app.run(host='0.0.0.0', port=3300, debug=False, threaded=True)