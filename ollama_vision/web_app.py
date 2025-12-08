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
from diffusers import StableDiffusionXLPipeline
import ollama
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT
from config import MODEL_NAME, RANDOM_SEED, REALVISXL_CONFIG

app = Flask(__name__)

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

def create_job(prompt, optimize_prompt=True):
    """Create a new job and return its ID."""
    global active_jobs_count
    job_id = str(uuid.uuid4())[:8]
    
    with jobs_lock:
        # Calculate queue position
        pending_jobs = sum(1 for j in jobs.values() if j['status'] in ['queued', 'processing'])
        
        job_info = {
            'id': job_id,
            'prompt': prompt,
            'optimize_prompt': optimize_prompt,
            'status': 'queued',  # queued, processing, completed, error
            'step': 0,
            'total': REALVISXL_CONFIG.get('inference_steps', 45),
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
        
        with pipeline_lock:
            seed = RANDOM_SEED + hash(job_id) % 10000  # Vary seed per job
            result = pipe(
                prompt=main_prompt,
                prompt_2=secondary_prompt if secondary_prompt else None,
                negative_prompt=REALVISXL_CONFIG.get('negative_prompt', ''),
                negative_prompt_2=REALVISXL_CONFIG.get('negative_prompt', ''),
                num_inference_steps=REALVISXL_CONFIG.get('inference_steps', 45),
                guidance_scale=REALVISXL_CONFIG.get('guidance_scale', 6.0),
                width=REALVISXL_CONFIG.get('width', 832),
                height=REALVISXL_CONFIG.get('height', 1216),
                generator=torch.Generator(device).manual_seed(seed),
                callback_on_step_end=job_progress_callback
            ).images[0]
        
        with jobs_lock:
            jobs[job_id]['stage'] = 'Saving'
        
        # Save image
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(workspace_root, REALVISXL_CONFIG.get('output_directory', 'ollama_vision/web_images/'))
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"web_{job_id}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        result.save(filepath)
        
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 AI Image Generator<br> 
        <h2>Warzone Squad</h2>
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
        
        let currentJobId = null;
        let queuePollInterval = null;
        
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
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, optimize_prompt: optimizePrompt })
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
    
    global is_offline_mode
    if os.path.exists(local_path):
        print(f"Loading from local: {local_path}")
        is_offline_mode = True
        pipe = StableDiffusionXLPipeline.from_pretrained(
            local_path,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True
        )
    else:
        print("Downloading RealVisXL V5.0...")
        is_offline_mode = False
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0",
            torch_dtype=dtype,
            cache_dir=models_dir,
            use_safetensors=True
        )
        pipe.save_pretrained(local_path)
    
    pipe = pipe.to(device)
    print(f"Pipeline loaded on {device}")
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
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Create job
        job_id = create_job(user_prompt, optimize_prompt=optimize_prompt)
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
    output_dir = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/web_generated/')
    # Get absolute path from workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(workspace_root, output_dir, filename)
    if not os.path.exists(filepath):
        # Try relative to script directory
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_images', 'realvisxl', filename)
    return send_file(filepath, mimetype='image/png')

@app.route('/download/<filename>')
def download_image(filename):
    output_dir = REALVISXL_CONFIG.get('output_directory', 'ollama_vision/generated_images/realvisxl/')
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(workspace_root, output_dir, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_images', 'realvisxl', filename)
    return send_file(filepath, as_attachment=True, download_name=filename)

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