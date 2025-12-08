# Ollama Vision + Stable Diffusion - Complete Offline Setup

This application generates hyper-realistic images using:
- **Ollama's qwen3-vl** vision model for prompt optimization
- **Stable Diffusion v1.4** for image generation

## ✨ Features

- 🌐 **Fully offline capable** - runs without internet after initial setup
- 🖼️ **Image-to-image** - analyze existing images and create enhanced versions
- ✍️ **Text-to-image** - generate images from text descriptions
- 🎨 **Prompt optimization** - automatically creates detailed, hyper-realistic prompts
- ⚡ **Hardware acceleration** - supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU

## 📋 Prerequisites

1. **Python 3.8+** with virtual environment support
2. **Ollama** installed and running ([download here](https://ollama.ai/download))

## ⚙️ Configuration

Copy the template configuration file and customize it:

```bash
cp config.template.json config.json
```

Edit `config.json` to set your preferred model and settings:

```json
{
  "model_name": "your_ollama_model",
  "inference_steps": 20,
  "random_seed": 42,
  "output_directory": "ollama_vision/generated_images"
}
```

**Configuration Options:**
- `model_name`: Ollama model to use for prompt optimization
- `inference_steps`: Number of diffusion steps (higher = better quality, slower)
- `random_seed`: Random seed for reproducible results
- `output_directory`: Where generated images are saved

## 🚀 One-Time Setup (Requires Internet)

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r ollama_vision/requirements.txt
```

### Step 2: Download Ollama Model

```bash
# Pull the vision model (one-time download, ~2-4GB)
ollama pull qwen3-vl
```

### Step 3: Download Stable Diffusion Model

```bash
# Run the model download script (one-time download, ~4GB)
python ollama_vision/download_models.py
```

This script will:
- Download Stable Diffusion v1.4 to `ollama_vision/models/`
- Verify your Ollama setup
- Confirm the app is ready for offline use

## 🎯 Running the App

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Run the application
python ollama_vision/main.py
```

### Mode Selection

When you run the app, you'll be prompted to choose between:

- **[OFFLINE MODE]** 🟢 - Uses locally downloaded models (no internet required)
- **[ONLINE MODE]** 🔴 - Downloads models on-demand (requires internet)

**For offline operation:**
1. Run `python ollama_vision/download_models.py` first (one-time download)
2. Choose option 1 (OFFLINE) when running the app

**For online operation:**
1. Choose option 2 (ONLINE) when running the app
2. Models will be downloaded automatically if not cached locally

### Usage Examples

#### Text-to-Image
```
Enter image filename in images/ folder (or leave blank for text-only): [press Enter]
Enter your prompt: a majestic mountain landscape at sunset
```

#### Image-to-Image
```
Enter image filename in images/ folder (or leave blank for text-only): example.jpg
Enter your prompt: make it more dramatic with storm clouds
```

## 📁 Directory Structure

```
ollama_vision/
├── main.py                 # Main application
├── download_models.py      # Model download script
├── requirements.txt        # Python dependencies
├── models/                 # Downloaded models (created after setup)
│   └── stable-diffusion-v1-4/
├── generated_images/       # Output images (created automatically)
└── images/                 # Input images (create this folder if needed)
```

## ⚙️ Configuration

Edit `main.py` to customize:

- **Inference steps** (line 94): `num_steps = 20`
  - Lower (10-15): Faster, lower quality
  - Higher (30-50): Slower, better quality
  
- **Random seed** (line 97): Change `42` for different variations

- **Model** (line 14): Change `qwen3-vl:latest` to use different Ollama models

## 🔧 Hardware Acceleration

The app automatically detects and uses:
- **CUDA** (NVIDIA GPUs) - fastest
- **MPS** (Apple Silicon M1/M2/M3) - fast
- **CPU** - slower but works everywhere

## 📝 How It Works

1. **Prompt Enhancement**: Your input is sent to Ollama's vision model, which creates a detailed, optimized prompt
2. **Image Generation**: The enhanced prompt is used with Stable Diffusion to generate a photorealistic image
3. **Output**: Generated images are saved to `ollama_vision/generated_images/`

## 🌐 Offline Operation

After running the setup once with internet:

✅ **Works offline:**
- Stable Diffusion image generation
- All Python dependencies (if venv is copied)
- Generated images saved locally

⚠️ **Requires Ollama running:**
- Ollama service must be running (but doesn't need internet)
- Models are cached locally after first download

## 🛠️ Troubleshooting

### "Model not found locally"
Run the download script again:
```bash
python ollama_vision/download_models.py
```

### "Ollama connection error"
Make sure Ollama is running:
```bash
# Check if Ollama is running
ollama list

# If not, start it (it should auto-start on most systems)
```

### Out of memory errors
- Reduce `num_inference_steps` to 10-15
- Close other applications
- Use a smaller image resolution

### Slow generation
- Normal on CPU (2-5 minutes per image)
- GPU/MPS: 10-30 seconds per image
- Consider reducing inference steps for faster generation

## 📦 Model Sizes

- **qwen3-vl**: ~2-4GB (downloaded via Ollama)
- **Stable Diffusion v1.4**: ~4GB (downloaded to `models/`)
- **Total disk space needed**: ~8GB

## 🔐 Privacy

All processing happens locally:
- No data sent to external servers
- No internet required after setup
- Your images never leave your machine

## 📄 License

This project uses:
- Stable Diffusion v1.4 (CreativeML Open RAIL-M License)
- Ollama qwen3-vl (check Ollama's licensing)
- See respective licenses for commercial use

## 🤝 Contributing

Feel free to submit issues or pull requests!
