"""
Validate reference image for InstantID face detection.
This script checks if a reference image has a detectable face suitable for InstantID.
"""

import sys
from PIL import Image
from instantid_pipeline import build_face_analyzer

def validate_reference(image_path: str):
    """Check if reference image has a clear, detectable face."""
    
    print("=" * 70)
    print("InstantID Reference Image Validator")
    print("=" * 70)
    print(f"\nChecking: {image_path}\n")
    
    # Check if file exists
    try:
        img = Image.open(image_path)
        print(f"✓ Image loaded: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"❌ ERROR: Cannot load image: {e}")
        return False
    
    # Check resolution
    if img.size[0] < 512 or img.size[1] < 512:
        print(f"⚠️  WARNING: Image is small ({img.size[0]}x{img.size[1]}). Recommended: at least 512x512")
    
    # Build face analyzer
    print("\n[1/2] Building InsightFace analyzer...")
    try:
        face_app = build_face_analyzer()
    except Exception as e:
        print(f"❌ ERROR: Cannot build face analyzer: {e}")
        return False
    
    # Detect faces
    print("[2/2] Detecting faces...")
    try:
        img_array = img.convert("RGB")
        import numpy as np
        img_array = np.array(img_array)
        
        faces = face_app.get(img_array)
        
        if len(faces) == 0:
            print("\n" + "=" * 70)
            print("❌ FACE DETECTION FAILED")
            print("=" * 70)
            print("\nNo face detected in this image!")
            print("\nREQUIREMENTS for InstantID reference image:")
            print("  • Clear, frontal face view (not profile)")
            print("  • Face fully visible (no hands, hair, or objects covering it)")
            print("  • Both eyes clearly visible")
            print("  • Good lighting on the face")
            print("  • Face should be at least 200x200 pixels in the image")
            print("\nCURRENT IMAGE ISSUES:")
            print("  ❌ Face is covered/obscured")
            print("  ❌ Face angle is not frontal enough")
            print("  ❌ Lighting is too dark")
            print("  ❌ Face is too small in the image")
            print("\n" + "=" * 70)
            return False
        
        elif len(faces) > 1:
            print(f"\n⚠️  WARNING: {len(faces)} faces detected!")
            print("InstantID will use the largest/most prominent face.")
            print("For best results, use an image with only ONE face.\n")
        
        # Analyze the main face
        face = faces[0]
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        print("\n" + "=" * 70)
        print("✅ FACE DETECTED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nFace bounding box: {bbox}")
        print(f"Face size: {face_width}x{face_height} pixels")
        print(f"Number of faces detected: {len(faces)}")
        
        # Quality checks
        print("\nQUALITY CHECKS:")
        
        if face_width < 200 or face_height < 200:
            print(f"  ⚠️  Face is small ({face_width}x{face_height}). Larger is better.")
        else:
            print(f"  ✓ Face size is good ({face_width}x{face_height})")
        
        # Check if face is centered
        img_center_x = img.size[0] / 2
        face_center_x = (bbox[0] + bbox[2]) / 2
        if abs(face_center_x - img_center_x) > img.size[0] * 0.3:
            print("  ⚠️  Face is off-center. Centered faces work better.")
        else:
            print("  ✓ Face is reasonably centered")
        
        # Check face confidence (if available)
        if hasattr(face, 'det_score'):
            confidence = face.det_score
            if confidence < 0.5:
                print(f"  ⚠️  Low detection confidence ({confidence:.2f}). Face may be unclear.")
            else:
                print(f"  ✓ Detection confidence: {confidence:.2f}")
        
        print("\n" + "=" * 70)
        print("✅ THIS IMAGE IS SUITABLE FOR INSTANTID")
        print("=" * 70)
        print("\nYou can use this image as a reference for InstantID generation.")
        print("The face will be locked across all generated scenes.\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during face detection: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_reference.py <path_to_image>")
        print("Example: python validate_reference.py assets/characters/model_01.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = validate_reference(image_path)
    sys.exit(0 if success else 1)
