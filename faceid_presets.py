"""
FaceID / InstantID Preset Configurations

This module defines optimized preset configurations for both IP-Adapter (FaceID)
and InstantID pipelines. Each preset is tuned for specific use cases.

Presets:
- MAX_FACE_LOCK: Strongest identity preservation (for dataset creation, LoRA training)
- CINEMATIC_BALANCED: Good identity + creative flexibility (for everyday generation)
"""

# ============================================================================
# IP-ADAPTER (FaceID) PRESETS
# ============================================================================

IP_MAX_FACE_LOCK = {
    "steps": 32,
    "guidance_scale": 4.0,
    "ip_adapter_scale": 0.9,  # Very high identity strength
    "width": 1024,
    "height": 1024,
}
"""
IP-Adapter MAX_FACE_LOCK preset.

Use this for:
- Dataset generation for LoRA training
- Character sheets where face must be nearly identical
- Maximum identity consistency across hundreds of images

Characteristics:
- Very strong face lock (0.9 scale)
- Lower guidance for stability
- Square 1024x1024 output
- 32 steps for quality
"""

IP_CINEMATIC_BALANCED = {
    "steps": 28,
    "guidance_scale": 4.5,
    "ip_adapter_scale": 0.75,  # Balanced identity + flexibility
    "width": 896,
    "height": 1152,  # Portrait aspect ratio
}
"""
IP-Adapter CINEMATIC_BALANCED preset.

Use this for:
- Everyday character generation
- Storytelling with natural pose/lighting variation
- Artistic/creative projects where some flexibility is desired

Characteristics:
- Balanced identity preservation (0.75 scale)
- Portrait-oriented 896x1152
- 28 steps for speed/quality balance
- More natural variation in pose/expression
"""


# ============================================================================
# INSTANTID PRESETS
# ============================================================================

INSTANTID_MAX_FACE_LOCK = {
    "steps": 30,
    "guidance_scale": 5.0,
    "controlnet_conditioning_scale": 0.95,
    "ip_adapter_scale": 0.95,
    "width": 1024,
    "height": 1024,
}
"""
InstantID MAX_FACE_LOCK preset.

Use this for:
- LoRA training datasets (strongest possible face lock)
- Reference sheets requiring extreme consistency
- Face swap / identity transfer applications

Characteristics:
- Very strong ControlNet strength (0.95)
- Strong facial keypoint guidance
- Square 1024x1024 output
- 30 steps for quality
- Strong IP-Adapter scale (0.95)
"""

INSTANTID_CINEMATIC_BALANCED = {
    "steps": 30,
    "guidance_scale": 5.0,
    "controlnet_conditioning_scale": 0.8,
    "ip_adapter_scale": 0.8,
    "width": 896,
    "height": 1152,  # Portrait aspect ratio
}
"""
InstantID CINEMATIC_BALANCED preset.

Use this for:
- Natural character storytelling
- Portraits with varied lighting/angles
- General-purpose face-consistent generation

Characteristics:
- Official recommended ControlNet strength (0.8)
- Portrait-oriented output
- 30 steps for good balance
- Natural variation while maintaining identity
- Balanced IP-Adapter scale (0.8)
"""


# ============================================================================
# PRESET SELECTOR HELPERS
# ============================================================================

def get_ip_preset(name: str) -> dict:
    """
    Get IP-Adapter preset by name.
    
    Args:
        name: Either 'max' or 'balanced'
    
    Returns:
        Preset configuration dict
    """
    presets = {
        'max': IP_MAX_FACE_LOCK,
        'balanced': IP_CINEMATIC_BALANCED,
    }
    if name not in presets:
        raise ValueError(f"Unknown IP preset: {name}. Choose 'max' or 'balanced'")
    return presets[name]


def get_instantid_preset(name: str) -> dict:
    """
    Get InstantID preset by name.
    
    Args:
        name: Either 'max' or 'balanced'
    
    Returns:
        Preset configuration dict
    """
    presets = {
        'max': INSTANTID_MAX_FACE_LOCK,
        'balanced': INSTANTID_CINEMATIC_BALANCED,
    }
    if name not in presets:
        raise ValueError(f"Unknown InstantID preset: {name}. Choose 'max' or 'balanced'")
    return presets[name]
