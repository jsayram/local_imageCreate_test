"""
End-to-end verification: InstantID → FLUX refinement workflow.

This script tests the two-stage pipeline:
1. RealVisXL + InstantID → identity-locked base image
2. FLUX img2img → photorealistic refinement

Do NOT modify existing behavior or main.py.
"""

from pathlib import Path
from PIL import Image

from instantid_pipeline import (
    build_instantid_pipeline,
    build_face_analyzer,
    generate_with_instantid,
)

from flux_refiner import (
    build_flux_img2img_pipeline,
    refine_photo_with_flux,
)


if __name__ == "__main__":
    print("=" * 70)
    print("InstantID → FLUX.2 Two-Stage Verification Test")
    print("=" * 70)
    
    # Ensure output directory exists
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Reference image
    ref = "assets/characters/model_01.png"
    
    # ========================================================================
    # STAGE 1: InstantID - Generate identity-locked base
    # ========================================================================
    
    print("\n" + "─" * 70)
    print("STAGE 1: InstantID - Identity-Locked Base Generation")
    print("─" * 70)
    
    print("\n[1/5] Building InstantID pipeline...")
    instantid_pipe = build_instantid_pipeline()
    
    print("\n[2/5] Building face analyzer...")
    face_app = build_face_analyzer()
    
    print(f"\n[3/5] Generating identity-locked base image...")
    print(f"       Reference: {ref}")
    
    base_prompt = (
        "ultra realistic portrait of the same woman in a softly lit studio, "
        "shot on full-frame camera, 85mm lens, f/1.4, shallow depth of field"
    )
    
    base_img = generate_with_instantid(
        pipe=instantid_pipe,
        face_app=face_app,
        ref_image_path=ref,
        scene_prompt=base_prompt,
        seed=123,
        steps=30,
    )
    
    stage1_path = out_dir / "stage1_identity_flux2_test.png"
    base_img.save(stage1_path)
    print(f"\n✓ Saved identity-locked base image to: {stage1_path}")
    
    # ========================================================================
    # STAGE 2: FLUX - Photorealistic Refinement
    # ========================================================================
    
    print("\n" + "─" * 70)
    print("STAGE 2: FLUX - Photorealistic Refinement")
    print("─" * 70)
    
    print("\n[4/5] Building FLUX img2img pipeline...")
    print("       (First run will download ~20GB FLUX.1-dev model)")
    flux_pipe = build_flux_img2img_pipeline()
    
    print("\n[5/5] Refining base image with FLUX...")
    
    # Reload base image
    base_img = Image.open(stage1_path).convert("RGB")
    
    # Define photographic refinement prompts
    prompts = [
        "ultra realistic photo of the same woman in a dim jazz bar, warm tungsten light, 50mm lens, candid shot",
        "fashion editorial photo of the same woman on a rooftop at golden hour, soft backlight, 85mm lens",
    ]
    
    for i, p in enumerate(prompts, start=1):
        print(f"\n   Refinement {i}/{len(prompts)}")
        print(f"   Prompt: {p[:60]}...")
        
        refined = refine_photo_with_flux(
            pipe=flux_pipe,
            base_image=base_img,
            prompt=p,
            seed=200 + i,
            steps=28,  # FLUX works well with 20-30 steps
            strength=0.35,  # Low strength preserves identity
            guidance_scale=3.5,  # FLUX guidance sweet spot
        )
        
        out_path = out_dir / f"flux2_refined_test_{i}.png"
        refined.save(out_path)
        print(f"   ✓ Saved: {out_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("✓ Two-Stage Verification Complete")
    print("=" * 70)
    print(f"\nGenerated files in {out_dir}/:")
    print(f"  1. {stage1_path.name} - InstantID base (identity-locked)")
    print(f"  2. flux2_refined_test_1.png - FLUX refined (jazz bar)")
    print(f"  3. flux2_refined_test_2.png - FLUX refined (rooftop golden hour)")
    
    print("\nVerification checklist:")
    print("  ☐ Stage 1 image has the same face as reference")
    print("  ☐ Stage 2 images preserve the identity from Stage 1")
    print("  ☐ Stage 2 images show photorealistic refinement quality")
    print("  ☐ Different scenes/lighting work correctly")
    
    print("\nIf all checks pass, the two-stage workflow is functional!")
    print("You can now integrate this into main.py as needed.\n")
