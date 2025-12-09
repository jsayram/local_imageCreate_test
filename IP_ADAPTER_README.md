# IP-Adapter Character Mode - Documentation

## Overview

The IP-Adapter Character Mode feature adds consistent character generation to your RealVisXL image pipeline. Using **IP-Adapter Plus for SDXL**, you can take a single reference portrait and generate the same person in completely different scenes, poses, and settings while maintaining facial identity.

## Key Differences from Existing Features

| Feature | Use Case | Identity Control | Scene Freedom |
|---------|----------|------------------|---------------|
| **IP-Adapter Mode** | Same face, different scenes | Strongest (locks facial features) | Complete freedom |
| **Img2Img Reference** | Modify existing image | Medium (preserves composition) | Limited (based on strength) |
| **Character Consistency (seed)** | Similar-looking person | Weakest (seed-based variation) | Complete freedom |

## Setup

### 1. Install Dependencies

All required packages are already in `requirements.txt`:
```bash
pip install --upgrade "diffusers[torch]" transformers accelerate safetensors
```

### 2. Prepare Reference Portrait

Place your reference portrait image in:
```
assets/characters/model_01.png
```

**Tips for best results:**
- Use a clear, well-lit portrait
- Face should be clearly visible and centered
- High resolution (1024x1024+ recommended)
- Neutral or desired expression
- Minimal occlusions (no sunglasses, hands covering face, etc.)

## Usage

### Quick Start Test

Run the standalone test script to verify everything works:

```bash
python test_character.py
```

This will generate 3 different scenes with the same character and save them to `outputs/`.

### Using in Main Application

1. **Start the application:**
   ```bash
   python ollama_vision/main.py
   ```

2. **Choose RealVisXL model** (option 2 or 3):
   - RealVisXL V5.0 (latest)
   - RealVisXL V4.0 (stable)

3. **Configure settings** (guidance, steps, etc.)

4. **Enable IP-Adapter Mode:**
   ```
   Enable IP-Adapter character mode? (y/N): y
   ```

5. **Select reference image:**
   - Choose from `assets/characters/`
   - Or enter custom path

6. **Set identity strength** (0.5-1.0):
   - **0.8 (default)**: Balanced - strong identity, good scene freedom
   - **0.9-1.0**: Maximum identity lock, minimal variation
   - **0.5-0.7**: More creative freedom, slightly weaker identity

7. **Enter your prompt:**
   - Describe only the SCENE, not the person
   - Example: "in a sunlit kitchen, casual outfit, natural light"
   - NOT: "woman with brown hair in a kitchen" (identity comes from reference)

## How It Works

### Technical Details

**IP-Adapter Plus** uses a CLIP image encoder to extract facial features from your reference portrait and injects them into the SDXL generation process. This provides much stronger identity preservation than seed-based methods while allowing complete scene freedom.

**Pipeline:**
1. Reference portrait → CLIP Image Encoder → Feature extraction
2. Your scene prompt → Text Encoders → Scene description
3. IP-Adapter → Combines identity features + scene description
4. SDXL UNet → Generates image with consistent face in new scene

### Identity Strength Tuning

The `ip_adapter_scale` parameter (0.5-1.0) controls how much the model prioritizes the reference identity:

```python
# In code:
pipe.set_ip_adapter_scale(0.8)  # Default
```

**Recommended ranges:**
- **0.5-0.6**: Loose identity, maximum creative variation
- **0.7-0.8**: Balanced (recommended starting point)
- **0.8-0.9**: Strong identity lock
- **0.9-1.0**: Maximum identity preservation, minimal variation

## Example Prompts

### Good Prompts (Scene-focused)
```
✅ "in a sunlit kitchen, casual outfit, natural light, half body shot"
✅ "wearing a black leather jacket in a rainy neon city street at night"
✅ "in a soft pastel studio with a blurred floral background, beauty portrait"
✅ "at a coffee shop, reading a book, warm afternoon lighting"
```

### Bad Prompts (Redundant character description)
```
❌ "woman with brown hair in a kitchen" - Identity already in reference!
❌ "25-year-old female model wearing..." - Don't describe the person!
```

## Integration with Existing Features

### Combined with Character Manager

1. Create initial portrait with IP-Adapter
2. Save as character with fixed seed
3. Future generations:
   - Use same reference image
   - Use same seed
   - = Maximum consistency across sessions

### Combined with Img2Img

- **IP-Adapter**: For character consistency (face)
- **Img2Img**: For composition/pose reference (body/scene)
- Can use both, but IP-Adapter typically gives better face consistency

## Troubleshooting

### Face doesn't match reference
- **Increase identity strength** → Try 0.9 instead of 0.8
- **Check reference image quality** → Clear, well-lit, unobstructed face
- **Remove character descriptions from prompt** → Describe scene only

### Too much variation
- **Lower guidance scale** → Try 3.5-4.5 for SDXL with IP-Adapter
- **Increase identity strength** → Try 0.9-1.0
- **Check negative prompt** → Avoid "different person, multiple faces"

### Scene doesn't change enough
- **Lower identity strength** → Try 0.6-0.7
- **More detailed scene prompts** → Be specific about setting/lighting
- **Increase guidance scale slightly** → Try 4.5-5.0

### Generation is slow
- IP-Adapter adds minimal overhead (~5% slower)
- Most time is SDXL itself
- Consider reducing steps (30 is usually sufficient)

## File Locations

```
ollama_vision/
  character_pipeline.py    # Core IP-Adapter implementation
  main.py                   # Integration with main app
  
assets/
  characters/               # Reference portraits go here
    model_01.png
    
outputs/                    # Test script outputs
  model01_scene_1.png
  model01_scene_2.png
  
test_character.py           # Standalone test script
```

## Advanced Usage

### Multiple Characters

Store different reference portraits:
```
assets/characters/
  jessica.png
  mike.png
  sarah.png
```

Select different references for different sessions.

### Adjusting Scale Mid-Generation

```python
from character_pipeline import adjust_ip_adapter_scale

# After building pipeline
adjust_ip_adapter_scale(pipe, 0.9)  # Increase identity lock
```

### Custom Model Versions

```python
# Use with RealVisXL V4.0
pipe = build_realvis_ip_adapter_pipeline(
    model_id="SG161222/RealVisXL_V4.0",
    ip_adapter_scale=0.8
)
```

## Performance

**M1 Max (64GB):**
- Build pipeline: ~30-45 seconds (one-time per session)
- Per image: ~45-60 seconds (similar to standard SDXL)
- Memory: ~8GB VRAM peak

**CUDA GPU:**
- Build pipeline: ~15-20 seconds
- Per image: ~10-20 seconds (depends on GPU)
- Memory: ~6-8GB VRAM

## Tips for Best Results

1. **Reference Selection**
   - Use professionally shot portraits when possible
   - Front-facing or 3/4 angle works best
   - Avoid extreme expressions (neutral/slight smile is versatile)

2. **Prompt Strategy**
   - Focus on environment, lighting, pose
   - Let IP-Adapter handle the face
   - Be specific about scene details

3. **Consistency Across Sessions**
   - Always use same reference image
   - Use same seed if combining with character manager
   - Keep identity strength consistent

4. **Experimentation**
   - Try different strengths for different use cases
   - Portrait shots: 0.8-0.9
   - Full body/distant shots: 0.6-0.7
   - Creative variations: 0.5-0.6

## Known Limitations

- **SDXL models only** (not compatible with SD v1.4)
- **Single person focus** (works best with one main character)
- **Face-forward bias** (extreme angles may have more variation)
- **First-time download** (IP-Adapter weights ~2.5GB, one-time)

## Support

If you encounter issues:

1. Check that reference image exists and is valid
2. Verify SDXL model is selected (not SD v1.4)
3. Try default settings first (0.8 strength, 30 steps, guidance 4.0)
4. Check console output for detailed error messages

For questions or improvements, please open an issue on GitHub.
