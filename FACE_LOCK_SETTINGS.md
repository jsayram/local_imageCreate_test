# InstantID Face Lock Settings - Maximum Identity Preservation

## Problem
The initial InstantID test was **not preserving the face** - it was changing between scenes despite using InstantID.

## Root Causes
1. **IP-Adapter scale too low** (was 0.8, needed 1.0)
2. **ControlNet conditioning too weak** (was 0.8, needed 1.0)
3. **Different seeds per scene** (was 42, 43, 44 - should be SAME seed)
4. **Weak negative prompts** (didn't explicitly prevent face changes)
5. **Prompts didn't emphasize identity** (missing "same person" prefix)

## Solution - Maximum Face Lock Configuration

### 1. IP-Adapter Scale: 1.0 (Maximum)
```python
# In build_instantid_pipeline():
pipe.set_ip_adapter_scale(1.0)  # Was 0.8, now 1.0 = MAXIMUM face lock
```

**Why:** IP-Adapter scale controls how strongly the face embedding influences the generation. 
- 0.5-0.7 = Soft influence (face can vary)
- 0.8-0.9 = Strong influence (mostly same face)
- **1.0 = MAXIMUM** (absolutely no face variation allowed)

### 2. ControlNet Conditioning Scale: 1.0 (Maximum)
```python
# In generate_with_instantid():
controlnet_conditioning_scale=1.0  # Was 0.8, now 1.0
```

**Why:** ControlNet uses facial keypoints (eyes, nose, mouth positions) to preserve facial structure.
- 0.6-0.8 = Moderate structure preservation
- **1.0 = MAXIMUM** structure lock (face geometry cannot change)

### 3. Same Seed Across All Scenes
```python
scenes = [
    {"prompt": "scene 1", "seed": 42},  # ALL use seed 42
    {"prompt": "scene 2", "seed": 42},  # Not 43!
    {"prompt": "scene 3", "seed": 42},  # Not 44!
]
```

**Why:** Different seeds introduce randomness that can alter facial features. Using the SAME seed across all scenes ensures maximum consistency.

### 4. Enhanced Negative Prompts
```python
negative_prompt = (
    "different person, changed face, wrong identity, altered features, "
    "different eyes, different nose, different mouth, "
    "morphed face, face swap, multiple people, clone, twin, lookalike, "
    "low quality, distorted, deformed, blurry, bad anatomy, extra limbs, out of frame"
)
```

**Why:** Explicitly tell the model what NOT to do:
- ❌ "different person" - prevents face swapping
- ❌ "changed face" - prevents facial alterations
- ❌ "wrong identity" - prevents identity drift
- ❌ "different eyes/nose/mouth" - prevents feature changes
- ❌ "face swap, clone, twin" - prevents similar-but-not-same faces

### 5. Identity-Focused Prompts
```python
full_prompt = f"photo of the same person, {scene_prompt}"
```

**Why:** Prepending "photo of the same person" to every prompt explicitly tells the model to preserve identity across scenes.

**Before:**
```
"sunlit living room, casual clothes..."
```

**After:**
```
"photo of the same person, sunlit living room, casual clothes..."
```

### 6. Identity Strength Ratio (Optional)
```python
identitynet_strength_ratio=0.8  # 0.8 = strong, 1.0 = maximum
```

**Why:** Some InstantID versions support this parameter to control face embedding strength (not all versions have this).

## Complete Settings Summary

| Parameter | Old Value | New Value | Purpose |
|-----------|-----------|-----------|---------|
| **IP-Adapter Scale** | 0.8 | **1.0** | Maximum face embedding influence |
| **ControlNet Scale** | 0.8 | **1.0** | Maximum facial structure lock |
| **Seed Strategy** | Different (42,43,44) | **Same (42,42,42)** | Eliminate randomness |
| **Negative Prompt** | Generic | **Face-specific** | Prevent identity drift |
| **Prompt Prefix** | None | **"same person"** | Emphasize identity |
| **Guidance Scale** | 5.0 | **5.0** | (unchanged) |
| **Steps** | 30 | **30** | (unchanged) |

## Testing Results

### Old Settings (Face Changing):
- IP-Adapter: 0.8
- ControlNet: 0.8
- Seeds: 42, 43, 44
- Result: ❌ **Face changed between scenes**

### New Settings (Maximum Lock):
- IP-Adapter: **1.0**
- ControlNet: **1.0**
- Seeds: **42, 42, 42**
- Enhanced prompts: ✅
- Result: ✅ **Face should be locked across scenes**

## When to Use What

### Soft Mode (IP-Adapter Plus)
- Scale: 0.5-0.7
- Use when: You want the same "type" of person but allow variation
- Best for: Character archetypes, general consistency

### Medium Mode (IP-Adapter Plus)
- Scale: 0.75-0.85
- Use when: You want strong consistency but some artistic freedom
- Best for: Styled portraits, artistic variations

### Hard Mode (InstantID - Maximum Lock)
- IP-Adapter: **1.0**
- ControlNet: **1.0**
- Same seed: **Yes**
- Use when: You need THE EXACT SAME PERSON across scenes
- Best for: Character continuity, storytelling, before/after shots

## Common Issues & Fixes

### Issue: Face still changing slightly
**Fix:** Increase steps to 40-50 for more refinement:
```python
steps=50  # More steps = more time for face to stabilize
```

### Issue: Face is locked but pose/composition is too similar
**Fix:** Add more variety to scene descriptions while keeping seed same:
```python
"photo of the same person, close-up portrait..."  # Scene 1
"photo of the same person, full body wide shot..."  # Scene 2
```

### Issue: Face is locked but image quality drops
**Fix:** This is the trade-off - maximum lock can reduce creativity. Try:
```python
controlnet_conditioning_scale=0.9  # Slightly reduce for quality
```

### Issue: Getting different facial expressions
**Fix:** This is NORMAL and GOOD - expressions should vary! Only the IDENTITY should be locked.

## Files Modified

1. **instantid_pipeline.py**
   - Set `pipe.set_ip_adapter_scale(1.0)` 
   - Added `identitynet_strength_ratio` parameter
   - Changed default `controlnet_conditioning_scale=1.0`
   - Enhanced negative prompts with face-specific terms
   - Added "same person" prefix to prompts

2. **test_instantid_character.py**
   - Changed all seeds to 42 (same seed)
   - Updated `controlnet_conditioning_scale=1.0`
   - Added `identitynet_strength_ratio=0.8`

## Next Steps

After testing with maximum lock settings:

1. **Check results**: Review `outputs/instantid_test/scene_{1-3}.png`
2. **Compare faces**: Ensure all 3 scenes have IDENTICAL facial features
3. **Fine-tune if needed**: If still changing, try:
   - Increase steps to 40-50
   - Use even more explicit negative prompts
   - Verify reference image has clear, frontal face

## Technical Notes

### Why InstantID is Stronger than IP-Adapter Plus

**IP-Adapter Plus:**
- Uses image embeddings only (CLIP-based)
- No facial keypoint guidance
- Relies purely on semantic similarity
- Face can drift over generations

**InstantID:**
- Uses InsightFace 512-dim face embeddings (facial recognition quality)
- Uses ControlNet with facial keypoints (eyes, nose, mouth positions)
- Two-stage guidance: structure (keypoints) + identity (embeddings)
- Face lock is much stronger

### Performance Impact

Maximum face lock settings:
- **Generation time**: ~90-120 seconds per image (30 steps × 3s/step)
- **VRAM usage**: ~8-10GB (SDXL + ControlNet + IP-Adapter)
- **Quality trade-off**: Stricter lock = less creative freedom
- **Best practice**: Use maximum lock only when identity preservation is critical

---

**Updated:** December 9, 2025
**Status:** ✅ Ready for testing with maximum face-lock configuration
