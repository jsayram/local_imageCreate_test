# InstantID Quick Start

## TL;DR

```bash
# 1. Install dependencies
source .venv/bin/activate
pip install opencv-python insightface onnxruntime

# 2. Run test
python test_instantid_character.py
```

First run downloads ~2.5GB of models (one-time only).

## What you get

**Much stronger face/body consistency** than IP-Adapter Plus. Same person across completely different:
- Scenes (kitchen → street → forest)
- Poses (sitting → standing → walking)
- Lighting (daylight → neon night → golden hour)
- Clothing (casual → leather jacket → fantasy dress)

## When to use

- **IP-Adapter Plus**: General character work, faster iteration, good balance
- **InstantID**: When face MUST be identical, magazine-quality consistency needed

See `INSTANTID_README.md` for full documentation.
