# preprocess_images.py
from PIL import Image
import os
import sys # Import sys for installation check

# --- Installation Check ---
try:
    from PIL import Image
except ImportError:
    print("Pillow library not found. Please install it using:")
    print("pip install Pillow")
    sys.exit(1) # Exit if Pillow is not installed
# --------------------------

def fix_png_profiles(image_dir):
    """Opens PNG images, removes ICC profile, and saves in place."""
    print(f"[*] Processing directory: {image_dir}")
    count = 0
    errors = 0
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith('.png'):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    # Check if image has an ICC profile before saving
                    if 'icc_profile' in img.info:
                        print(f"  Removing profile from: {file}")
                        # Save without the profile
                        img.save(img_path, format='PNG', icc_profile=None)
                        count += 1
                    # Else: No profile to remove, skip saving
                except Exception as e:
                    print(f"  ERROR processing {img_path}: {e}")
                    errors += 1
    print(f"[*] Finished processing {image_dir}. Profiles removed from {count} images. Errors: {errors}.")

# --- Script Execution ---
if __name__ == "__main__":
    # !! IMPORTANT: Back up your data before running this script !!
    # This script modifies your image files in place.

    # Define the directories containing your PNG images
    # These paths assume the script is run from the project root
    low_dir = os.path.join('test', 'low')
    high_dir = os.path.join('test', 'high')

    print("--- Starting PNG Profile Removal ---")

    if os.path.isdir(low_dir):
        fix_png_profiles(low_dir)
    else:
        print(f"Warning: Directory not found - {low_dir}")

    if os.path.isdir(high_dir):
        fix_png_profiles(high_dir)
    else:
        print(f"Warning: Directory not found - {high_dir}")

    print("--- PNG Profile Removal Finished ---")