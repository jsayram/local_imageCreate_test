"""
InstantID Character Consistency Test
Tests the InstantID "Face Lock: Hard Mode" with stronger identity preservation.
"""

import os
from instantid_pipeline import (
    build_instantid_pipeline,
    build_face_analyzer,
    generate_with_instantid,
)

if __name__ == "__main__":
    print("=" * 70)
    print("InstantID Character Test - Face Lock: Hard Mode")
    print("=" * 70)
    
    print("\n[INFO] InstantID provides MUCH stronger face consistency than")
    print("       standard IP-Adapter Plus, at the cost of extra setup.")
    print("       Perfect for when you need the SAME person across many scenes.\n")
    
    # Reference image path
    ref = "assets/characters/model_01.png"
    
    # Check if reference exists
    if not os.path.exists(ref):
        print(f"\n❌ ERROR: Reference image not found at {ref}")
        print("Please place a portrait image at assets/characters/model_01.png")
        exit(1)
    
    print(f"[1/4] Using reference: {ref}\n")
    
    # Build the pipeline (this will download models on first run)
    print("[2/4] Building InstantID pipeline...")
    print("       (First run will download ~2.5GB of checkpoints)")
    pipe = build_instantid_pipeline()
    
    # Build face analyzer
    print("\n[3/4] Building face analyzer...")
    print("       (First run will download InsightFace models)")
    face_app = build_face_analyzer()
    
    # Define test scenes
    print("\n[4/4] Generating scenes with LOCKED identity...")
    print()
    
    scenes = [
        {
            "prompt": "portrait, looking at camera, natural smile, soft lighting",
            "seed": 42,  # SAME SEED for all scenes = maximum identity lock
        },
        {
            "prompt": "wearing elegant black dress, city background, evening light, professional headshot",
            "seed": 42,  # SAME SEED for maximum face consistency
        },
        {
            "prompt": "casual white shirt, outdoor natural setting, golden hour, relaxed pose",
            "seed": 42,  # SAME SEED for maximum face consistency
        },
    ]
    
    # Create output directory
    output_dir = "outputs/instantid_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each scene
    for i, scene_config in enumerate(scenes, start=1):
        print(f"\n{'─' * 70}")
        print(f"Scene {i}/{len(scenes)}")
        print(f"{'─' * 70}")
        print(f"Prompt: {scene_config['prompt'][:65]}...")
        
        img = generate_with_instantid(
            pipe=pipe,
            face_app=face_app,
            ref_image_path=ref,
            scene_prompt=scene_config['prompt'],
            seed=scene_config['seed'],
            steps=30,
            guidance_scale=5.0,
            controlnet_conditioning_scale=0.8,
            identitynet_strength_ratio=0.8,
        )
        
        output_path = f"{output_dir}/scene_{i}.png"
        img.save(output_path)
        print(f"✓ Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ All scenes generated successfully!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nWith InstantID, the face should be VERY consistent across all")
    print("3 different scenes - much stronger identity lock than IP-Adapter!")
    print("\nCompare these results with your standard IP-Adapter outputs to")
    print("see the difference in face/body consistency.\n")
