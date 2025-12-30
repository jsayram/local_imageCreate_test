# InstantID - Face Lock: Hard Mode 😈

## What is this?

InstantID is a **separate, more powerful identity preservation mode** that gives you MUCH stronger face/body consistency than your existing IP-Adapter Plus setup.

Think of it as:
- **IP-Adapter Plus** = Good balance, easy setup, flexible
- **InstantID** = Maximum face lock, requires more setup, but unbeatable consistency

## When to use InstantID vs IP-Adapter Plus

| Feature | IP-Adapter Plus (current) | InstantID (new) |
|---------|---------------------------|-----------------|
| Face consistency | Good (0.5-0.85 scale) | Excellent (near-perfect) |
| Setup complexity | Simple | Moderate (needs InsightFace) |
| Speed | Fast | Slightly slower |
| Dependencies | Light | +opencv, insightface |
| Use case | General character work | When face MUST be identical |

## Setup

### 1. Install dependencies

```bash
source .venv/bin/activate
pip install opencv-python insightface onnxruntime
```

### 2. Place your reference portrait

```bash
# Use the same reference you already have:
assets/characters/model_01.png
```

### 3. Run the test

```bash
python test_instantid_character.py
```

**First run will download:**
- InstantID ControlNet weights (~2.5GB)
- InstantID IP-Adapter weights
- InsightFace antelopev2 model

These are cached locally in `./checkpoints` and `./models` and won't re-download.

## How it works

1. **InsightFace** extracts a 512-dim face embedding from your reference
2. **ControlNet** uses facial keypoints to guide pose/structure
3. **IP-Adapter** injects the face embedding into SDXL's cross-attention
4. **Result**: Same face, different scenes, VERY strong consistency

## Usage

### Standalone test script

```bash
python test_instantid_character.py
```

Generates 3 scenes with locked identity in `outputs/instantid_test/`.

### Custom generation (Python)

```python
from instantid_pipeline import (
    build_instantid_pipeline,
    build_face_analyzer,
    generate_with_instantid,
)

# One-time setup
pipe = build_instantid_pipeline()
face_app = build_face_analyzer()

# Generate
image = generate_with_instantid(
    pipe=pipe,
    face_app=face_app,
    ref_image_path="assets/characters/model_01.png",
    scene_prompt="woman in a modern kitchen, casual outfit, natural light",
    seed=42,
    steps=30,
    guidance_scale=5.0,
    controlnet_conditioning_scale=0.8,  # Higher = stronger face lock
)

image.save("output.png")
```

## Tuning parameters

### `controlnet_conditioning_scale` (0.6-1.0)
- **0.6**: Allows more scene variation
- **0.8** (default): Balanced
- **1.0**: Maximum identity lock (may copy pose from reference)

### `guidance_scale` (4.0-7.0)
- **4.0-5.0**: More creative freedom
- **5.0** (default): Balanced
- **6.0-7.0**: Stricter prompt following

### `steps` (20-50)
- **20-30**: Faster, good quality
- **30** (default): Recommended
- **40-50**: Better quality, slower

## Comparison with IP-Adapter Plus

Run both tests and compare:

```bash
# IP-Adapter Plus (your existing setup)
python test_character.py
# → outputs/config_2_balanced_(recommended)/

# InstantID (new face-lock mode)
python test_instantid_character.py
# → outputs/instantid_test/
```

You should see **much stronger facial consistency** with InstantID, especially across extreme pose/lighting changes.

## File structure

```
instantid_pipeline.py          # Main wrapper module
test_instantid_character.py    # Standalone test script
third_party/instantid/         # Vendored official pipeline
checkpoints/                   # InstantID weights (auto-downloaded)
models/                        # InsightFace models (auto-downloaded)
```

## Integration with main.py (future)

This is **intentionally not integrated** into `main.py` yet. Once you test and like it, you can add:

```python
# In main.py, add a new mode option:
mode = input("Mode: [1] IP-Adapter Plus  [2] InstantID: ")

if mode == "2":
    from instantid_pipeline import build_instantid_pipeline, build_face_analyzer, generate_with_instantid
    # ... rest of integration
```

## Troubleshooting

### "No face detected"
- Ensure reference image has a clear, visible face
- Try a tighter crop showing face prominently
- Avoid extreme angles or occlusions

### "CUDA out of memory"
- InstantID uses more VRAM than IP-Adapter Plus
- Reduce image resolution or use CPU for InsightFace

### "InsightFace installation fails"
- Make sure you have `cmake` and build tools installed
- Try: `pip install insightface --no-cache-dir`
- On macOS: `brew install cmake`

## Notes

- **Does NOT modify** any of your existing code
- **Additive only** - your IP-Adapter Plus setup still works
- Think of this as a specialized "sniper rifle" for when you need perfect face consistency
- Use IP-Adapter Plus for general work, InstantID when face MUST match exactly

---

🎯 **Bottom line**: If the face is changing too much between scenes with IP-Adapter Plus, try InstantID for maximum lock.
