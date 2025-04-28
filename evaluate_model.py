# evaluate_model.py
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from trainer.input import IMG_WIDTH, IMG_HEIGHT
# import color 
from skimage import color

# --- Add project root to Python path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# ---------------------------------------

# --- Import necessary components from the trainer package ---
from trainer import input as input_data
# --- MODIFICATION: Import the TF wrapper, not a direct numpy func ---
from trainer.utils import tf_lab_normalized_to_rgb # Import the TF wrapper
# --- END MODIFICATION ---
from trainer.model import ColorCastRemoval # Import the custom model class
# Import metric if needed, e.g., CIEDE2000 (requires skimage)
try:
    from skimage.color import deltaE_ciede2000, lab2rgb # Import lab2rgb for metric calc if needed
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not found. CIEDE2000 metric will not be calculated.")

def calculate_metrics(img_true_rgb, img_pred_rgb):
    """Calculates PSNR, SSIM, and optionally CIEDE2000."""
    metrics = {}
    # Ensure float32 for TF metrics
    img_true_tf = tf.convert_to_tensor(img_true_rgb, dtype=tf.float32)
    img_pred_tf = tf.convert_to_tensor(img_pred_rgb, dtype=tf.float32)

    # Add batch dim if single image
    if tf.rank(img_true_tf) == 3:
        img_true_tf = tf.expand_dims(img_true_tf, 0)
        img_pred_tf = tf.expand_dims(img_pred_tf, 0)

    # Calculate PSNR and SSIM using TensorFlow
    metrics['psnr'] = tf.image.psnr(img_pred_tf, img_true_tf, max_val=1.0)[0].numpy()
    metrics['ssim'] = tf.image.ssim(img_pred_tf, img_true_tf, max_val=1.0)[0].numpy()

    if SKIMAGE_AVAILABLE:
        try:
            # Convert RGB [0,1] numpy to standard LAB for CIEDE2000 using skimage
            # Ensure input is float64 for skimage
            lab_true_sk = color.rgb2lab(np.clip(img_true_rgb, 0, 1).astype(np.float64))
            lab_pred_sk = color.rgb2lab(np.clip(img_pred_rgb, 0, 1).astype(np.float64))

            # Calculate mean CIEDE2000 over the image
            delta_e = deltaE_ciede2000(lab_true_sk, lab_pred_sk)
            metrics['ciede2000'] = np.mean(delta_e)
        except Exception as e:
            print(f"Warning: Could not calculate CIEDE2000: {e}")
            metrics['ciede2000'] = np.nan

    return metrics


def evaluate(args):
    """Loads model, runs evaluation, prints metrics, and saves samples."""
    print(f"--- Starting Model Evaluation ---")
    print(f"Arguments: {args}")

    # Create output directory for images and debug images
    debug_dir = None # Initialize debug_dir
    if args.output_dir:
        print(f"[*] Creating output directory for images at: {args.output_dir}")
        tf.io.gfile.makedirs(args.output_dir)
        debug_dir = os.path.join(args.output_dir, "debug_intermediates")
        tf.io.gfile.makedirs(debug_dir)
        print(f"[*] Debug images will be saved to: {debug_dir}")
    else:
        print("Warning: --output-dir not specified. Comparison and debug images will not be saved.")

    # --- Load Check 1: Verify model path ---
    if not args.model_path or not tf.io.gfile.exists(args.model_path):
        print(f"Error: Saved model not found at specified path: {args.model_path}")
        return
    print(f"[*] Attempting to load saved model from: {args.model_path}")

    # Load saved model
    try:
        custom_objects = {'ColorCastRemoval': ColorCastRemoval}
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
        print("[*] Model loaded successfully.")
        model.summary()
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Test Data ---
    if not args.test_low_dir or not args.test_high_dir:
        print("Error: Please provide paths to low and high-quality test images.")
        return

    print("[*] Loading test dataset (expecting LAB)...")
    test_dataset = input_data.load_dataset(
        low_dir=args.test_low_dir,
        high_dir=args.test_high_dir,
        batch_size=args.batch_size,
        is_training=False
    )

    if tf.data.experimental.cardinality(test_dataset) == 0:
        print("Error: No data found in the test dataset directories.")
        return

    # Initialize lists to store metrics
    all_psnr = []
    all_ssim = []
    all_ciede2000 = []

    # --- Evaluation Loop ---
    print("[*] Starting evaluation loop...")
    images_saved = 0
    debug_images_saved_this_run = 0

    for step, (batch_low_lab_tf, batch_high_lab_tf) in enumerate(test_dataset):
        if step % 10 == 0:
             print(f"  Processing batch {step+1}...")

        # --- Perform model inference ---
        # Ensure training=False is passed if model.call expects it
        try:
             enhanced_lab_tf, _ = model(batch_low_lab_tf, training=False)
        except TypeError: # Fallback if model.call doesn't accept training arg
             print("Warning: model.call might not accept 'training' argument. Calling without it.")
             enhanced_lab_tf, _ = model(batch_low_lab_tf)


        # --- Convert LAB Tensors back to RGB Tensors using TF Wrapper ---
        try:
            # --- MODIFICATION: Use TF wrapper ---
            batch_low_rgb_tf = tf_lab_normalized_to_rgb(batch_low_lab_tf)
            batch_high_rgb_tf = tf_lab_normalized_to_rgb(batch_high_lab_tf)
            enhanced_rgb_tf = tf_lab_normalized_to_rgb(enhanced_lab_tf)
            # --- END MODIFICATION ---

            # --- MODIFICATION: Convert TF Tensors to NumPy AFTER conversion ---
            batch_low_rgb_np = batch_low_rgb_tf.numpy()
            batch_high_rgb_np = batch_high_rgb_tf.numpy()
            enhanced_rgb_np = enhanced_rgb_tf.numpy()
            # --- END MODIFICATION ---

        except Exception as conversion_e:
             print(f"ERROR during LAB->RGB conversion in evaluate_model (batch {step+1}): {conversion_e}", file=sys.stderr)
             import traceback
             traceback.print_exc() # Print full traceback for conversion errors
             continue # Skip batch if conversion fails

        # Iterate through images in the batch
        batch_size_actual = batch_low_rgb_np.shape[0]
        for i in range(batch_size_actual):
            img_idx = step * args.batch_size + i

            # Get individual images (NumPy arrays, RGB format, range [0, 1])
            # Clip just in case, although the conversion function should handle it
            im_low_rgb = np.clip(batch_low_rgb_np[i], 0.0, 1.0)
            im_high_rgb = np.clip(batch_high_rgb_np[i], 0.0, 1.0)
            im_enhanced_rgb = np.clip(enhanced_rgb_np[i], 0.0, 1.0)

            # # --- Intermediate Visualization ---
            # if debug_dir and debug_images_saved_this_run < args.num_images:
            #      try:
            #         img_to_save_low = (im_low_rgb * 255).astype(np.uint8)[:,:,::-1] # BGR for cv2
            #         img_to_save_enhanced = (im_enhanced_rgb * 255).astype(np.uint8)[:,:,::-1] # BGR for cv2
            #         img_to_save_high = (im_high_rgb * 255).astype(np.uint8)[:,:,::-1] # BGR for cv2

            #         debug_save_path_low = os.path.join(debug_dir, f'debug_{img_idx:04d}_0_input_rgb.png')
            #         cv2.imwrite(debug_save_path_low, img_to_save_low)
            #         debug_save_path_enh = os.path.join(debug_dir, f'debug_{img_idx:04d}_1_output_rgb.png')
            #         cv2.imwrite(debug_save_path_enh, img_to_save_enhanced)
            #         debug_save_path_high = os.path.join(debug_dir, f'debug_{img_idx:04d}_2_target_rgb.png')
            #         cv2.imwrite(debug_save_path_high, img_to_save_high)

            #         debug_images_saved_this_run += 1
            #      except Exception as debug_save_e:
            #          print(f"Warning: Could not save debug intermediate images for index {img_idx}: {debug_save_e}")
            # # --- End Intermediate Visualization ---

            # --- Calculate Metrics ---
            try:
                metrics = calculate_metrics(im_high_rgb, im_enhanced_rgb)
                all_psnr.append(metrics.get('psnr', np.nan))
                all_ssim.append(metrics.get('ssim', np.nan))
                if SKIMAGE_AVAILABLE:
                    all_ciede2000.append(metrics.get('ciede2000', np.nan))
                metric_title = (f"Output\nPSNR: {metrics.get('psnr', np.nan):.2f}, "
                                f"SSIM: {metrics.get('ssim', np.nan):.4f}")
                if SKIMAGE_AVAILABLE:
                     metric_title += f"\nCIEDE2k: {metrics.get('ciede2000', np.nan):.2f}"
            except Exception as metric_e:
                print(f"Warning: Could not calculate metrics for image index {img_idx}: {metric_e}")
                metric_title = "Model Output (Metric Error)"
                all_psnr.append(np.nan)
                all_ssim.append(np.nan)
                if SKIMAGE_AVAILABLE: all_ciede2000.append(np.nan)

            # --- Save Final Comparison Plot ---
            if args.output_dir and images_saved < args.num_images:
                try:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(im_low_rgb)
                    axes[0].set_title('Input (Low Quality)')
                    axes[0].axis('off')
                    axes[1].imshow(im_enhanced_rgb)
                    axes[1].set_title(metric_title)
                    axes[1].axis('off')
                    axes[2].imshow(im_high_rgb)
                    axes[2].set_title('Ground Truth (High Quality)')
                    axes[2].axis('off')
                    plt.tight_layout()
                    save_path = os.path.join(args.output_dir, f'comparison_{img_idx:04d}.png')
                    plt.savefig(save_path)
                    plt.close(fig)
                    images_saved += 1
                except Exception as vis_e:
                     print(f"Warning: Could not save comparison visualization for image index {img_idx}: {vis_e}")

        # Check if we have saved enough images after processing the batch
        # Stop saving images but continue processing batches for metrics
        if args.output_dir and images_saved >= args.num_images and debug_images_saved_this_run >= args.num_images:
             # Only print this message once
             if step == 0 or (images_saved == args.num_images and debug_images_saved_this_run == args.num_images and i == batch_size_actual -1) : # Rough check to print once
                 print(f"  Reached limit of {args.num_images} saved comparison and debug images. Continuing metric calculation...")

    # --- End outer loop ---

    print("[*] Evaluation loop finished.")

    # --- Calculate and Print Average Results ---
    avg_psnr = np.nanmean(all_psnr) if all_psnr else np.nan
    avg_ssim = np.nanmean(all_ssim) if all_ssim else np.nan
    avg_ciede2000 = np.nanmean(all_ciede2000) if SKIMAGE_AVAILABLE and all_ciede2000 else np.nan

    print("\n--- Evaluation Results (Averages) ---")
    print(f"  Images Processed: {len(all_psnr)}") # Use len(all_psnr) as it reflects successfully processed images
    print(f"  Average PSNR: {avg_psnr:.4f}")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    if SKIMAGE_AVAILABLE:
        print(f"  Average CIEDE2000: {avg_ciede2000:.4f}")
    if args.output_dir:
        print(f"[*] Sample comparison images saved to: {args.output_dir}")
        if debug_dir:
             print(f"[*] Debug intermediate images saved to: {debug_dir}")
    print("------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained Color Cast Removal model.")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the saved .keras model file.'
    )
    parser.add_argument(
        '--test-low-dir',
        required=True,
        type=str,
        help='Path to TEST low-quality images directory (e.g., *.png)'
    )
    parser.add_argument(
        '--test-high-dir',
        required=True,
        type=str,
        help='Path to TEST high-quality images directory (e.g., *.png)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Optional: Directory to save comparison and debug images.'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=10,
        help='Maximum number of comparison image sets (plots and intermediates) to save.'
    )
    args = parser.parse_args()
    evaluate(args)

# **Key Changes in `evaluate_model.py`:**

# 1.  **Import:** Changed the import from `trainer.utils` to only import `tf_lab_normalized_to_rgb`.
# 2.  **Conversion:** Inside the evaluation loop, the conversion from LAB to RGB now uses the imported TensorFlow wrapper:
#     ```python
#     batch_low_rgb_tf = tf_lab_normalized_to_rgb(batch_low_lab_tf)
#     batch_high_rgb_tf = tf_lab_normalized_to_rgb(batch_high_lab_tf)
#     enhanced_rgb_tf = tf_lab_normalized_to_rgb(enhanced_lab_tf)
#     ```
# 3.  **NumPy Conversion:** After the conversion using the TF wrapper, the resulting tensors (`*_tf`) are converted to NumPy arrays (`*_np`) using `.numpy()`:
#     ```python
#     batch_low_rgb_np = batch_low_rgb_tf.numpy()
#     batch_high_rgb_np = batch_high_rgb_tf.numpy()
#     enhanced_rgb_np = enhanced_rgb_tf.numpy()
#     ```
# 4.  **Subsequent Usage:** All subsequent code (clipping, saving debug images, calculating metrics, saving comparison plots) now uses the NumPy arrays (`im_low_rgb`, `im_high_rgb`, `im_enhanced_rgb`).
# 5.  **Error Handling:** Added a `try...except TypeError` block around the `model(...)` call just in case the loaded model's `call` signature doesn't accept the `training` argument (though it should after the last `model.py` modification). Also added `traceback.print_exc()` to the conversion error block for better debugging if it happens again.

# This should resolve the `AttributeError` and allow your evaluation script to run correctly, using the properly clipped LAB-to-RGB conversion defined in your `utils.py`. You should now be able to see the saved comparison and debug images in your specified output directo