"""
Test script for IP-Adapter Plus character consistency feature.
Verifies that the same person can be generated in different scenes.
"""

import sys
import os

# Add ollama_vision to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ollama_vision'))

from character_pipeline import build_realvis_ip_adapter_pipeline, generate_scene_with_reference

# Test configurations to compare
TEST_CONFIGS = {
    "1": {
        "name": "Maximum Scene Variation",
        "ip_scale": 0.5,
        "guidance": 6.0,
        "desc": "Same face, very different scenes/poses"
    },
    "2": {
        "name": "Balanced (Recommended)",
        "ip_scale": 0.65,
        "guidance": 6.5,
        "desc": "Good identity + scene variety"
    },
    "3": {
        "name": "Strong Identity",
        "ip_scale": 0.75,
        "guidance": 7.0,
        "desc": "Very consistent face, moderate variation"
    },
    "4": {
        "name": "Maximum Consistency",
        "ip_scale": 0.85,
        "guidance": 7.5,
        "desc": "Almost exact identity, less variety"
    }
}

if __name__ == "__main__":
    print("=" * 60)
    print("Character Consistency Test - Configuration Comparison")
    print("=" * 60)
    
    # Show available configurations
    print("\nAvailable test configurations:")
    print()
    for key, config in TEST_CONFIGS.items():
        print(f"  [{key}] {config['name']}")
        print(f"      IP-Adapter: {config['ip_scale']}, Guidance: {config['guidance']}")
        print(f"      → {config['desc']}")
        print()
    
    # Get user choice
    choice = input("Select configuration (1-4) or press Enter for option 2: ").strip()
    if not choice:
        choice = "2"
    
    if choice not in TEST_CONFIGS:
        print(f"Invalid choice. Using default (2 - Balanced)")
        choice = "2"
    
    selected_config = TEST_CONFIGS[choice]
    
    print(f"\n{'=' * 60}")
    print(f"Using: {selected_config['name']}")
    print(f"IP-Adapter Scale: {selected_config['ip_scale']}")
    print(f"Guidance Scale: {selected_config['guidance']}")
    print(f"{'=' * 60}")
    
    # Build the pipeline with selected identity strength
    print("\n[1/4] Building IP-Adapter pipeline...")
    pipe = build_realvis_ip_adapter_pipeline(ip_adapter_scale=selected_config['ip_scale'])
    
    # Reference image path
    ref = "assets/characters/model_01.png"
    
    # Check if reference exists
    if not os.path.exists(ref):
        print(f"\n❌ ERROR: Reference image not found at {ref}")
        print("Please place a portrait image at assets/characters/model_01.png")
        exit(1)
    
    print(f"\n[2/4] Using reference: {ref}")
    
    # Define test scenes with detailed descriptions
    scenes = [
        "modern white kitchen interior with marble countertops and stainless steel appliances, woman wearing casual denim jeans and white t-shirt, bright natural daylight from large windows, half body shot, standing relaxed pose, contemporary home setting",
        
        "dark rainy city street at night with neon signs and reflections, woman in black leather motorcycle jacket and combat boots walking, wet pavement reflections, blue and pink neon lights, cyberpunk urban atmosphere, full body shot, moody cinematic lighting",
        
        "professional photography studio with soft pink backdrop and out-of-focus pastel flowers, woman in elegant dress with glamorous makeup, beauty portrait headshot, studio softbox lighting, shallow depth of field bokeh, high fashion editorial style",
    ]
    
    # Create output directory for this config
    output_dir = f"outputs/config_{choice}_{selected_config['name'].replace(' ', '_').lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[3/4] Generating {len(scenes)} scenes with consistent character...")
    print()

    # Use a fixed seed across all scenes for stronger identity retention
    base_seed = 42
    
    # Generate each scene
    for i, scene in enumerate(scenes, start=1):
        print(f"\n--- Scene {i}/{len(scenes)} ---")
        print(f"Prompt: {scene[:80]}...")
        
        img = generate_scene_with_reference(
            pipe,
            ref_image_path=ref,
            scene_prompt=scene,
            seed=base_seed,
            steps=40,
            guidance_scale=selected_config['guidance'],
        )
        
        output_path = f"{output_dir}/scene_{i}.png"
        img.save(output_path)
        print(f"✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("[4/4] ✓ All scenes generated successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}/")
    print("The SAME FACE should appear in 3 DIFFERENT scenes/poses.")
    print("\nTip: Try different configurations to compare results!")
