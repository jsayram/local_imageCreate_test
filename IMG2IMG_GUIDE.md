# Image-to-Image Reference Mode Guide

## Overview

The **Reference Image** feature uses SDXL's img2img pipeline to maintain facial features from a reference photo while generating new scenes, poses, or backgrounds.

## How It Works

1. **Select a saved character**
2. **Choose a reference image** from their generation folder
3. **Describe the new scene** (e.g., "at a coffee shop", "in the rain")
4. **Set strength** (how much to change vs preserve)
5. **Generate** - keeps facial features, changes everything else

## When to Use Reference Images

### ✅ Good Use Cases
- **Scene Changes**: Same person, different location/background
- **Pose Variations**: Same face, different body position
- **Lighting Changes**: Same person, different time of day
- **Outfit Changes**: Same face, different clothes
- **Expression Variations**: Same features, different emotion

### ❌ Not Recommended For
- First generation of a character (use text-to-image instead)
- Completely different people
- Drastically different art styles

## Strength Parameter Guide

The `strength` parameter controls how much the model changes from the reference:

### Conservative (0.3 - 0.5)
- **Use for**: Minor adjustments, slight expression changes
- **Preserves**: Almost everything - face, lighting, composition
- **Changes**: Very subtle scene elements
- **Example**: Same pose, slightly different background

### Balanced (0.5 - 0.7) ⭐ **RECOMMENDED**
- **Use for**: New scenes while keeping the person
- **Preserves**: Facial features, general proportions
- **Changes**: Background, lighting, pose
- **Example**: Office → Park, sitting → standing

### Aggressive (0.7 - 0.85)
- **Use for**: Major scene transformations
- **Preserves**: Core facial structure, identity
- **Changes**: Almost everything else
- **Example**: Indoor studio → Outdoor sunset

### Maximum (0.85 - 0.95)
- **Use for**: Creative reinterpretations
- **Preserves**: Basic facial resemblance only
- **Changes**: Nearly everything
- **Example**: Realistic → Artistic style

## Step-by-Step Example

### Scenario: Generate "Jessica" in different locations

**Step 1: Initial Generation (Text-to-Image)**
```
Prompt: "28-year-old woman with shoulder-length auburn hair, 
bright green eyes, oval face, light freckles, natural smile, 
professional portrait, studio lighting"

Save as: "Jessica"
Seed: 136789
```

**Step 2: Use Reference (Image-to-Image)**
```
1. Select character: Jessica
2. Choose reference: generated_20251208_143052.png (first image)
3. Strength: 0.75
4. New prompt: "standing in a misty forest at dawn, wearing a green jacket"

Result: Jessica's face from reference + forest scene
```

**Step 3: Another Scene**
```
1. Select character: Jessica
2. Choose reference: (same or different Jessica image)
3. Strength: 0.65
4. New prompt: "sitting at a Parisian cafe, drinking coffee, sunset lighting"

Result: Jessica's face + cafe scene
```

## Tips for Best Results

### 1. Choose Good Reference Images
- ✅ Clear, well-lit face
- ✅ Good resolution (1024×1024 recommended)
- ✅ Neutral or desired expression
- ❌ Blurry, dark, or distorted faces
- ❌ Extreme angles or occlusions

### 2. Match Image Dimensions
- Reference automatically resized to target resolution
- Best results when reference is same size as target
- SDXL works best at 1024×1024 or 832×1216

### 3. Prompt for the New Scene
- Focus on **what's different** (location, pose, clothing)
- Don't repeat facial features (model copies from reference)
- Include lighting/atmosphere for the new scene

**Good Prompt (with reference):**
```
"standing on a beach at sunset, wearing a blue sundress, 
wind in hair, warm golden lighting"
```

**Bad Prompt (redundant):**
```
"28-year-old woman with auburn hair and green eyes 
standing on a beach..."  ← facial details unnecessary
```

### 4. Adjust Strength Based on Reference Quality
- **High-quality reference**: Use 0.6-0.8 (can change more)
- **Average reference**: Use 0.5-0.7 (balanced)
- **Imperfect reference**: Use 0.4-0.6 (preserve more)

### 5. Combine with Character Consistency
- Use **same seed** (from saved character)
- Select **same character** each time
- Use **reference image** for precise facial control
- Result: Maximum consistency across generations

## Troubleshooting

### Face Doesn't Match Reference
- **Lower strength** (try 0.5 instead of 0.8)
- **Better reference image** (clearer face)
- **Check prompt** (conflicting facial descriptions)

### Scene Doesn't Change Enough
- **Increase strength** (try 0.8 instead of 0.6)
- **More detailed scene prompt**
- **Different lighting/atmosphere keywords**

### Getting Weird Results
- **Reference image quality** - use clear, well-lit images
- **Strength too high** (>0.9) - reduce to 0.7-0.8
- **Conflicting prompts** - ensure scene description makes sense

### Image Looks Blurry
- **Increase inference steps** (try 50 instead of 30)
- **Lower strength slightly** (preserve more detail)
- **Use higher resolution reference**

## Workflow Comparison

### Traditional Text-to-Image
```
Prompt: "woman with auburn hair, green eyes, in forest"
Problem: Each generation = different face
```

### Character Consistency (Fixed Seed)
```
Character: "Jessica" (seed 136789)
Prompt: "in forest"
Result: Similar face, but variations possible
```

### Character + Reference Image ⭐
```
Character: "Jessica" (seed 136789)
Reference: Best previous generation
Strength: 0.75
Prompt: "in forest"
Result: EXACT face from reference + forest scene
```

## Advanced Techniques

### 1. Progressive Refinement
Generate multiple variations, select best, use as reference for next batch:
```
Gen 1: Text-to-Image → 5 images
Select best → Save as reference
Gen 2: Img2Img (strength 0.6) → Refine pose
Gen 3: Img2Img (strength 0.7) → Different scene
```

### 2. Expression Control
Use different reference images for different expressions:
```
Reference A: Smiling Jessica → Happy scenes
Reference B: Serious Jessica → Dramatic scenes
```

### 3. Multi-Character Scenes
Generate each character separately with references, then composite:
```
Jessica (img2img) → Forest scene
Mike (img2img) → Same forest prompt
Composite → Both in scene (external tool)
```

## Quick Reference Table

| Goal | Strength | Prompt Focus |
|------|----------|--------------|
| Same pose, different background | 0.5-0.6 | Background details |
| Same face, different pose | 0.6-0.7 | Pose + scene |
| Different location entirely | 0.7-0.8 | New location + lighting |
| Different artistic style | 0.8-0.9 | Style keywords |
| Minor touch-ups | 0.3-0.5 | Specific adjustments |

## Conclusion

The Reference Image feature is the most powerful tool for **character consistency**:
- **Fixed Seed**: Similar faces across generations
- **Character Manager**: Reusable descriptions + settings
- **Reference Image**: EXACT facial features + new scenes

Combine all three for maximum control over your character generations!
