"""
FaceID / InstantID Preset Verification Test (Enhanced)

This script verifies that InstantID pipelines work correctly with the preset configurations.
It uses RealVisXL for better photorealism and tests multiple use cases.
"""

from pathlib import Path
import os

from faceid_presets import (
    INSTANTID_MAX_FACE_LOCK,
    INSTANTID_CINEMATIC_BALANCED,
)

from instantid_pipeline import (
    build_instantid_pipeline,
    build_face_analyzer,
    generate_with_instantid,
)

if __name__ == "__main__":
    print("=" * 70)
    print("FaceID / InstantID Preset Verification Test (Enhanced)")
    print("=" * 70)
    
    # Setup
    out_dir = Path("outputs/faceid_tests_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = "assets/characters/model_01.png"
    
    # Use RealVisXL if available, otherwise fallback to base SDXL
    realvis_path = "ollama_vision/models/RealVisXL_V4.0"
    if os.path.exists(realvis_path):
        model_path = realvis_path
        print(f"Using RealVisXL model at: {model_path}")
    else:
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"RealVisXL not found, using base SDXL: {model_path}")
    
    print(f"\nReference image: {ref_path}")
    print(f"Output directory: {out_dir}\n")
    
    # ========================================================================
    # INSTANTID TESTS
    # ========================================================================
    
    print("─" * 70)
    print("INSTANTID TESTS")
    print("─" * 70)
    
    print("\n[1/2] Building InstantID pipeline...")
    instantid_pipe = build_instantid_pipeline(base_model_path=model_path)
    
    print("\n[2/2] Building face analyzer...")
    face_app = build_face_analyzer()
    
    # Define test cases
    # Using user's specific settings from generated_20251208_213537.txt
    user_seed = 136789
    user_steps = 45
    user_width = 832
    user_height = 1216
    user_neg_prompt = "worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, open mouth, tooth, ugly, tiling, poorly drawn hands, porly drawn eyes, poorly drawn face, extra limbs, disfigured, deformed"

    # Override preset with user dimensions/steps
    USER_MAX_LOCK = INSTANTID_MAX_FACE_LOCK.copy()
    USER_MAX_LOCK.update({
        "steps": user_steps,
        "width": user_width,
        "height": user_height,
        # Keep the strong lock we just tuned
        "controlnet_conditioning_scale": 0.95,
        "ip_adapter_scale": 0.95,
    })

    test_cases = [
        {
            "name": "01_portrait_max_lock_user_settings",
            "prompt": "professional portrait, natural expression, soft lighting, 8k, photorealistic",
            "preset": USER_MAX_LOCK,
            "debug": True
        },
        {
            "name": "02_full_body_action_user_settings",
            "prompt": "full body shot, running in a park, athletic wear, sunny day, dynamic pose, detailed background",
            "preset": USER_MAX_LOCK,
            "debug": False
        },
    ]
    
    print(f"\nRunning {len(test_cases)} test cases with User Seed {user_seed}...")
    
    for case in test_cases:
        print(f"\n--- Running Case: {case['name']} ---")
        print(f"Prompt: {case['prompt']}")
        print(f"Preset: {case['preset']}")
        
        result = generate_with_instantid(
            pipe=instantid_pipe,
            face_app=face_app,
            ref_image_path=ref_path,
            scene_prompt=case['prompt'],
            seed=user_seed,  # Using the specific seed 136789
            steps=case['preset']["steps"],
            guidance_scale=case['preset']["guidance_scale"],
            controlnet_conditioning_scale=case['preset']["controlnet_conditioning_scale"],
            ip_adapter_scale=case['preset']["ip_adapter_scale"],
            return_debug=case['debug']
        )
        
        if case['debug']:
            img, kps = result
            kps_path = out_dir / f"{case['name']}_kps.png"
            kps.save(kps_path)
            print(f"  ✓ Saved Debug Keypoints: {kps_path}")
        else:
            img = result
            
        out_path = out_dir / f"{case['name']}.png"
        img.save(out_path)
        print(f"  ✓ Saved: {out_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("✓ Enhanced Verification Complete")
    print("=" * 70)
    print(f"Check {out_dir} for results.")
    print("Inspect '01_portrait_max_lock_kps.png' to verify face detection accuracy.")
