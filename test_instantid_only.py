"""
InstantID-only verification test.
Tests that the InstantID pipeline works independently before adding FLUX.2 refinement.
"""

from pathlib import Path
from instantid_pipeline import (
    build_instantid_pipeline,
    build_face_analyzer,
    generate_with_instantid,
)

if __name__ == "__main__":
    print("=" * 70)
    print("InstantID Verification Test")
    print("=" * 70)
    
    # Ensure output directory exists
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Reference image
    ref = "assets/characters/model_01.png"
    
    print(f"\n[1/3] Using reference: {ref}")
    
    # Build InstantID pipeline
    print("\n[2/3] Building InstantID pipeline...")
    instantid_pipe = build_instantid_pipeline()
    
    print("\n[2/3] Building face analyzer...")
    face_app = build_face_analyzer()
    
    # Generate identity-locked image
    print("\n[3/3] Generating identity-locked portrait...")
    
    prompt = (
        "professional portrait photo in a softly lit studio, "
        "natural expression, looking at camera"
    )
    
    result_img = generate_with_instantid(
        pipe=instantid_pipe,
        face_app=face_app,
        ref_image_path=ref,
        scene_prompt=prompt,
        seed=42,
        steps=30,
    )
    
    # Save result
    output_path = out_dir / "stage1_identity_test.png"
    result_img.save(output_path)
    
    print("\n" + "=" * 70)
    print("✓ InstantID Verification Complete")
    print("=" * 70)
    print(f"\nSaved: {output_path}")
    print("\nVerify that the face matches the reference image.")
    print("If successful, you can proceed to test FLUX.2 refinement.\n")
