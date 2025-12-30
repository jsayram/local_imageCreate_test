"""
FaceID / InstantID Preset Verification Test

This script verifies that both IP-Adapter and InstantID pipelines work correctly
with the preset configurations defined in faceid_presets.py.

Generates 4 test images:
- IP-Adapter with MAX_FACE_LOCK preset
- IP-Adapter with CINEMATIC_BALANCED preset
- InstantID with MAX_FACE_LOCK preset
- InstantID with CINEMATIC_BALANCED preset

Does NOT modify main.py or any existing code - purely for verification.
"""

from pathlib import Path

from faceid_presets import (
    IP_MAX_FACE_LOCK,
    IP_CINEMATIC_BALANCED,
    INSTANTID_MAX_FACE_LOCK,
    INSTANTID_CINEMATIC_BALANCED,
)

from instantid_pipeline import (
    build_instantid_pipeline,
    build_face_analyzer,
    generate_with_instantid,  # Use the original working function
)

# Removed - using generate_with_instantid directly instead
# from instantid_helpers import generate_instantid_with_config

# Note: IP-Adapter test is optional - uncomment if you have the pipeline
# from character_pipeline import build_realvis_ip_adapter_pipeline
# from ip_adapter_helpers import generate_with_ip_and_config


if __name__ == "__main__":
    print("=" * 70)
    print("FaceID / InstantID Preset Verification Test")
    print("=" * 70)
    
    # Setup
    out_dir = Path("outputs/faceid_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = "assets/characters/model_01.png"
    
    print(f"\nReference image: {ref_path}")
    print(f"Output directory: {out_dir}\n")
    
    # ========================================================================
    # OPTIONAL: IP-Adapter Tests (uncomment if available)
    # ========================================================================
    
    # print("─" * 70)
    # print("IP-ADAPTER TESTS")
    # print("─" * 70)
    # 
    # print("\n[1/4] Building IP-Adapter pipeline...")
    # ip_pipe = build_realvis_ip_adapter_pipeline()
    # 
    # scene_ip = "in a cozy apartment living room, soft warm lighting, casual outfit"
    # 
    # print("\n[2/4] Testing IP_MAX_FACE_LOCK preset...")
    # print(f"  Settings: {IP_MAX_FACE_LOCK}")
    # img_ip_max = generate_with_ip_and_config(
    #     pipe=ip_pipe,
    #     ref_image_path=ref_path,
    #     scene_prompt=scene_ip,
    #     cfg=IP_MAX_FACE_LOCK,
    #     seed=42,
    # )
    # out_path = out_dir / "ip_max_face_lock.png"
    # img_ip_max.save(out_path)
    # print(f"  ✓ Saved: {out_path}")
    # 
    # print("\n[3/4] Testing IP_CINEMATIC_BALANCED preset...")
    # print(f"  Settings: {IP_CINEMATIC_BALANCED}")
    # img_ip_cine = generate_with_ip_and_config(
    #     pipe=ip_pipe,
    #     ref_image_path=ref_path,
    #     scene_prompt=scene_ip,
    #     cfg=IP_CINEMATIC_BALANCED,
    #     seed=43,
    # )
    # out_path = out_dir / "ip_cinematic_balanced.png"
    # img_ip_cine.save(out_path)
    # print(f"  ✓ Saved: {out_path}")
    
    # ========================================================================
    # INSTANTID TESTS
    # ========================================================================
    
    print("─" * 70)
    print("INSTANTID TESTS")
    print("─" * 70)
    
    print("\n[1/4] Building InstantID pipeline...")
    instantid_pipe = build_instantid_pipeline()
    
    print("\n[2/4] Building face analyzer...")
    face_app = build_face_analyzer()
    
    scene_instantid = "professional portrait, natural expression, soft lighting"
    
    print("\n[3/4] Testing INSTANTID_MAX_FACE_LOCK preset...")
    print(f"  Settings: {INSTANTID_MAX_FACE_LOCK}")
    img_inst_max = generate_with_instantid(
        pipe=instantid_pipe,
        face_app=face_app,
        ref_image_path=ref_path,
        scene_prompt=scene_instantid,
        seed=42,
        steps=INSTANTID_MAX_FACE_LOCK["steps"],
        guidance_scale=INSTANTID_MAX_FACE_LOCK["guidance_scale"],
        controlnet_conditioning_scale=INSTANTID_MAX_FACE_LOCK["controlnet_conditioning_scale"],
        ip_adapter_scale=INSTANTID_MAX_FACE_LOCK["ip_adapter_scale"],
    )
    out_path = out_dir / "instantid_max_face_lock.png"
    img_inst_max.save(out_path)
    print(f"  ✓ Saved: {out_path}")
    
    print("\n[4/4] Testing INSTANTID_CINEMATIC_BALANCED preset...")
    print(f"  Settings: {INSTANTID_CINEMATIC_BALANCED}")
    
    scene_2 = "elegant evening dress, city lights background"
    
    img_inst_cine = generate_with_instantid(
        pipe=instantid_pipe,
        face_app=face_app,
        ref_image_path=ref_path,
        scene_prompt=scene_2,
        seed=42,
        steps=INSTANTID_CINEMATIC_BALANCED["steps"],
        guidance_scale=INSTANTID_CINEMATIC_BALANCED["guidance_scale"],
        controlnet_conditioning_scale=INSTANTID_CINEMATIC_BALANCED["controlnet_conditioning_scale"],
        ip_adapter_scale=INSTANTID_CINEMATIC_BALANCED["ip_adapter_scale"],
    )
    out_path = out_dir / "instantid_cinematic_balanced.png"
    img_inst_cine.save(out_path)
    print(f"  ✓ Saved: {out_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("✓ Preset Verification Complete")
    print("=" * 70)
    
    print(f"\nGenerated files in {out_dir}/:")
    # print("  • ip_max_face_lock.png")
    # print("  • ip_cinematic_balanced.png")
    print("  • instantid_max_face_lock.png")
    print("  • instantid_cinematic_balanced.png")
    
    print("\nExpected results:")
    print("  MAX_FACE_LOCK → Nearly identical face, minimal variation")
    print("  CINEMATIC_BALANCED → Same person, more natural pose/lighting variety")
    
    print("\nVisually compare the outputs to verify preset behavior!")
    print("If all looks good, these presets are ready to use.\n")
