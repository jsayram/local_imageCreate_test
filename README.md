# Ollama Vision + Stable Diffusion Image Generator

A local, offline-capable AI image generation pipeline that combines **Ollama's vision models** for intelligent prompt optimization with **Stable Diffusion** and **RealVisXL** for hyper-realistic image generation.

## ✨ Features

- 🖼️ **Text-to-Image** - Generate images from text descriptions
- 🔍 **Image-to-Image** - Analyze existing images and create enhanced versions
- 🎨 **Prompt Optimization** - Automatically enhances your prompts for better results
- 🍎 **Apple Silicon Optimized** - Full support for M1/M2/M3 Macs with MPS acceleration
- 🌐 **Offline Capable** - Run completely offline after initial model downloads
- ⚙️ **Configurable** - All parameters customizable via `config.json`

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/download) installed and running
- ~15GB disk space for models

### Installation

```bash
# Clone the repository
git clone https://github.com/username/local_imageCreate_test.git
cd local_imageCreate_test

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r ollama_vision/requirements.txt

# Copy config template
cp config.template.json config.json
```

### Download Models

```bash
# Download Ollama model
ollama pull wizard-vicuna-uncensored

# Download Stable Diffusion models
python ollama_vision/download_models.py
# Choose: 1 (SD v1.4), 2 (RealVisXL), or 3 (Both)
```

### Run the Application

```bash
source .venv/bin/activate
python ollama_vision/main.py
```

## 📋 Usage

1. **Choose Operating Mode:**
   - `1` - OFFLINE (uses locally cached models)
   - `2` - ONLINE (downloads models on-demand)

2. **Choose Image Model (Apple Silicon only):**
   - `1` - Default (SD v1.4) - Faster, smaller images
   - `2` - RealVisXL V4.0 - Hyper-realistic, larger images

3. **Enter image filename** (optional) - For image-to-image mode
4. **Enter your prompt** - Describe what you want to generate

Generated images are saved to `ollama_vision/generated_images/`

## ⚙️ Configuration

Edit `config.json` to customize all parameters:

```json
{
  "model_name": "your_local_favorite_model_diffuser",
  "random_seed": 136789,
  "output_directory": "ollama_vision/generated_images",
  
  "realvisxl": {
    "inference_steps": 40,
    "guidance_scale": 5.5,
    "width": 1024,
    "height": 1024,
    "negative_prompt": "..."
  },
  
  "sd_v14": {
    "inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512
  }
}
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `inference_steps` | Denoising iterations (more = better quality) | 30-50 |
| `guidance_scale` | Prompt adherence (lower = more natural) | 5.0-7.5 |
| `width` / `height` | Output resolution | 1024×1024 for SDXL |
| `random_seed` | For reproducible results | Any integer |
| `negative_prompt` | What to avoid in generation | See config |

### Character Consistency

To generate the **same person** across multiple images:

1. **Keep `random_seed` fixed** (e.g., `136789`)
2. **Use identical character description** in every prompt:
   ```
   a 28-year-old woman with shoulder-length auburn hair, 
   bright green eyes, oval face, light freckles
   ```

## 🍎 Apple Silicon Performance

Optimized for M1/M2/M3 Macs:

| Model | Resolution | Steps | Time |
|-------|------------|-------|------|
| SD v1.4 | 512×512 | 50 | ~20 sec |
| RealVisXL | 1024×1024 | 40 | ~45-60 sec |
| RealVisXL | 832×1216 | 40 | ~50-65 sec |

## 📁 Project Structure

```
local_imageCreate_test/
├── config.json              # Your configuration (gitignored)
├── config.template.json     # Configuration template
├── ollama_vision/
│   ├── main.py              # Main application
│   ├── config.py            # Configuration loader
│   ├── prompts.py           # System prompts
│   ├── download_models.py   # Model downloader
│   ├── requirements.txt     # Python dependencies
│   ├── models/              # Downloaded models (~15GB)
│   └── generated_images/    # Output images
└── README.md
```

## 🔧 Available Models

### Stable Diffusion v1.4
- **Best for:** Fast generation, smaller file sizes
- **Resolution:** 512×512
- **Size:** ~4GB

### RealVisXL V4.0 (Apple Silicon)
- **Best for:** Hyper-realistic portraits and photography
- **Resolution:** 1024×1024 (or 832×1216 for portraits)
- **Size:** ~6.5GB
- **Optimized CFG:** 5.0-6.0 for best realism

## 🛠️ Troubleshooting

### "Model not found locally"
Run the download script:
```bash
python ollama_vision/download_models.py
```

### Slow generation
- Reduce `inference_steps` to 20-30
- Use SD v1.4 instead of RealVisXL
- Close other GPU-intensive applications

### Out of memory
- Reduce image resolution in config
- Use SD v1.4 (smaller model)
- Restart the application to clear memory

### Ollama connection error
```bash
# Check if Ollama is running
ollama list

# Pull required model
ollama pull your_favorite_model_to_use
```

## 📄 License

- **Stable Diffusion v1.4:** CreativeML Open RAIL-M License
- **RealVisXL:** Check [model page](https://huggingface.co/SG161222/RealVisXL_V4.0) for licensing
