import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define if ximgproc is available (optional, for advanced dehazing)
try:
    import cv2.ximgproc
    CV2_XIMGPROC_AVAILABLE = True
    print("[*] cv2.ximgproc module found. Advanced dehazing option available.")
except ImportError:
    CV2_XIMGPROC_AVAILABLE = False
    print("[!] cv2.ximgproc module not found. Advanced dehazing (darkChannelDehazing) will be skipped.")

from skimage import exposure, color as skimage_color, filters, restoration

# --- Add project root to Python path if your trainer package is not installed ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------------------------------------

from trainer.model import ColorCastRemoval
from trainer.utils import tf_rgb_to_lab_normalized, tf_lab_normalized_to_rgb
from trainer.input import IMG_WIDTH, IMG_HEIGHT

# --- Pre-processing Function ---
def preprocess_image_hybrid(rgb_image_np_0_1):
    if rgb_image_np_0_1.ndim == 2: # Grayscale
        print("Warning: Input image is grayscale. Converting to RGB.")
        rgb_image_np_0_1 = skimage_color.gray2rgb(rgb_image_np_0_1)
    elif rgb_image_np_0_1.shape[-1] == 4: # RGBA
        print("Warning: Input image is RGBA. Converting to RGB.")
        rgb_image_np_0_1 = skimage_color.rgba2rgb(rgb_image_np_0_1)

    image_to_process = np.clip(rgb_image_np_0_1, 0.0, 1.0).astype(np.float32)
    img_uint8 = (image_to_process * 255).astype(np.uint8) # For OpenCV

    # --- Optional Denoising (apply first if input is very noisy) ---
    # print("    Applying Denoising (Bilateral Filter)...")
    # denoised_uint8 = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=50, sigmaSpace=50) # Milder parameters
    # img_uint8 = denoised_uint8 # if denoising is applied

    # 1. CLAHE
    # print("    Applying CLAHE...")
    lab_cv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_cv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Tune clipLimit
    cl = clahe.apply(l_channel)
    merged_lab_cv = cv2.merge((cl, a_channel, b_channel))
    processed_rgb_uint8 = cv2.cvtColor(merged_lab_cv, cv2.COLOR_LAB2RGB)
    processed_rgb_np = (processed_rgb_uint8 / 255.0).astype(np.float32)

    # 2. Gamma Correction (for brightness/haze perception)
    # print("    Applying Gamma Correction...")
    processed_rgb_np = exposure.adjust_gamma(processed_rgb_np, gamma=0.9) # Tune gamma (0.7-1.0)

    # --- More Advanced Dehazing (Optional - requires opencv-contrib) ---
    if CV2_XIMGPROC_AVAILABLE:
        try:
            # print("    Applying Advanced Dehazing (Dark Channel Prior)...")
            img_to_dehaze_uint8 = (processed_rgb_np * 255).astype(np.uint8)
            # Note: darkChannelDehazing expects BGR input
            img_bgr_to_dehaze = cv2.cvtColor(img_to_dehaze_uint8, cv2.COLOR_RGB2BGR)
            # Parameters for darkChannelDehazing might need tuning
            dehazed_bgr = cv2.ximgproc.darkChannelDehazing(img_bgr_to_dehaze, patchSize=15, omega=0.95, t0=0.1, ksize=-1)
            processed_rgb_uint8_dehazed = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb_np = (processed_rgb_uint8_dehazed / 255.0).astype(np.float32)
        except Exception as e_dehaze:
            print(f"Warning: Advanced dehazing failed: {e_dehaze}. Skipping.")


    return np.clip(processed_rgb_np, 0.0, 1.0).astype(np.float32)

# --- Post-processing Function ---
def postprocess_image_hybrid(rgb_image_np_0_1):
    image_to_process = np.clip(rgb_image_np_0_1, 0.0, 1.0).astype(np.float32)
    # img_uint8_for_cv = (image_to_process * 255).astype(np.uint8) # For OpenCV functions

    # --- Optional Denoising of CNN output (if it's noisy) ---
    # print("    Applying Denoising to CNN output...")
    # Milder denoising for CNN output
    # denoised_cnn_output_np = restoration.denoise_bilateral(image_to_process, sigma_color=0.05, sigma_spatial=5, channel_axis=-1)
    # image_to_process = np.clip(denoised_cnn_output_np, 0.0, 1.0)

    # 1. Sharpening
    # print("    Applying Sharpening...")
    sharpened_rgb_np = filters.unsharp_mask(image_to_process, radius=1.5, amount=1.2, channel_axis=-1, preserve_range=False)
    sharpened_rgb_np = np.clip(sharpened_rgb_np, 0.0, 1.0)

    # 2. Global Contrast Stretch (per-channel)
    # print("    Applying Contrast Stretch...")
    contrast_stretched_channels = []
    if sharpened_rgb_np.ndim == 3 and sharpened_rgb_np.shape[-1] == 3: # RGB image
        for i_ch in range(sharpened_rgb_np.shape[-1]):
            channel = sharpened_rgb_np[..., i_ch]
            p_low, p_high = np.percentile(channel, (1, 99))
            
            if p_low >= p_high:
                ch_min_val = channel.min()
                ch_max_val = channel.max()
                if ch_min_val < ch_max_val: p_low, p_high = ch_min_val, ch_max_val
                else: p_low, p_high = 0.0, 1.0 # Fallback for truly flat channel
            
            try:
                stretched_channel = exposure.rescale_intensity(channel, in_range=(p_low, p_high))
                contrast_stretched_channels.append(stretched_channel)
            except ValueError as e:
                print(f"Warning: rescale_intensity failed for channel {i_ch} ({e}). Using channel as is.")
                contrast_stretched_channels.append(channel)
        
        if len(contrast_stretched_channels) == 3:
            final_rgb_np = np.stack(contrast_stretched_channels, axis=-1)
        else:
            print("Warning: Channel processing for contrast stretch failed. Using sharpened image.")
            final_rgb_np = sharpened_rgb_np
    elif sharpened_rgb_np.ndim == 2: # Grayscale
        p_low, p_high = np.percentile(sharpened_rgb_np, (1, 99))
        if p_low >= p_high:
            img_min_val, img_max_val = sharpened_rgb_np.min(), sharpened_rgb_np.max()
            if img_min_val < img_max_val: p_low, p_high = img_min_val, img_max_val
            else: p_low, p_high = 0.0, 1.0
        try:
            final_rgb_np = exposure.rescale_intensity(sharpened_rgb_np, in_range=(p_low, p_high))
        except ValueError as e:
            print(f"Warning: rescale_intensity failed for grayscale image ({e}). Using sharpened image.")
            final_rgb_np = sharpened_rgb_np
    else: # Unexpected shape
        print("Warning: Image is not RGB or Grayscale after sharpening. Skipping contrast stretch.")
        final_rgb_np = sharpened_rgb_np

    # 3. Optional: Saturation Boost (after contrast and sharpening)
    print("    Applying Saturation Boost...")
    hsv_img = skimage_color.rgb2hsv(final_rgb_np)
    hsv_img[..., 1] = np.clip(hsv_img[..., 1] * 1.1, 0, 1.0) # Tune factor
    final_rgb_np = skimage_color.hsv2rgb(hsv_img)

    return np.clip(final_rgb_np, 0.0, 1.0).astype(np.float32)


# --- Metric Calculation ---
def calculate_metrics_inference(img_true_rgb, img_pred_rgb):
    metrics_dict = {}
    img_true_tf = tf.convert_to_tensor(img_true_rgb, dtype=tf.float32)
    img_pred_tf = tf.convert_to_tensor(img_pred_rgb, dtype=tf.float32)

    if tf.rank(img_true_tf) == 3:
        img_true_tf = tf.expand_dims(img_true_tf, 0)
        img_pred_tf = tf.expand_dims(img_pred_tf, 0)

    try:
        metrics_dict['psnr'] = tf.image.psnr(img_pred_tf, img_true_tf, max_val=1.0)[0].numpy()
    except Exception: metrics_dict['psnr'] = np.nan
    try:
        metrics_dict['ssim'] = tf.image.ssim(img_pred_tf, img_true_tf, max_val=1.0)[0].numpy()
    except Exception: metrics_dict['ssim'] = np.nan
    try:
        lab_true_sk = skimage_color.rgb2lab(np.clip(img_true_rgb, 0, 1).astype(np.float64))
        lab_pred_sk = skimage_color.rgb2lab(np.clip(img_pred_rgb, 0, 1).astype(np.float64))
        delta_e_array = skimage_color.deltaE_ciede2000(lab_true_sk, lab_pred_sk, channel_axis=-1)
        metrics_dict['ciede2000'] = np.mean(delta_e_array)
    except Exception: metrics_dict['ciede2000'] = np.nan
    return metrics_dict

# --- Main Inference Function ---
def run_hybrid_inference(args):
    print(f"--- Starting Hybrid Inference ---")
    print(f"Arguments: {args}")

    if not tf.io.gfile.exists(args.model_path):
        print(f"Error: Saved model not found at: {args.model_path}"); return

    try:
        custom_objects = {'ColorCastRemoval': ColorCastRemoval}
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
        print("[*] Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}"); import traceback; traceback.print_exc(); return

    if not os.path.exists(args.input_path):
        print(f"Error: Input path not found: {args.input_path}"); return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[*] Output images and plots will be saved to: {args.output_dir}")
    else:
        print("Warning: --output-dir not specified. Plots will not be saved.")

    image_paths = []
    if os.path.isdir(args.input_path):
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'):
            image_paths.extend(tf.io.gfile.glob(os.path.join(args.input_path, ext)))
    elif os.path.isfile(args.input_path): image_paths.append(args.input_path)

    if not image_paths: print(f"No images found at: {args.input_path}"); return

    print(f"[*] Found {len(image_paths)} images to process.")
    all_metrics_summary = []

    for i, img_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {img_path}")
        try:
            img_bgr_uint8 = cv2.imread(img_path)
            if img_bgr_uint8 is None: print(f"Warning: Could not read image {img_path}. Skipping."); continue
            
            img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
            raw_input_rgb_np = (img_rgb_uint8 / 255.0).astype(np.float32)
            resized_input_rgb_np = cv2.resize(raw_input_rgb_np, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            print("  Applying pre-processing...")
            preprocessed_rgb_np = preprocess_image_hybrid(resized_input_rgb_np)
            
            preprocessed_rgb_tensor_batched = tf.convert_to_tensor(np.expand_dims(preprocessed_rgb_np, axis=0), dtype=tf.float32)
            input_lab_tf_batched = tf_rgb_to_lab_normalized(preprocessed_rgb_tensor_batched)

            print("  Running CNN inference...")
            enhanced_lab_tf_batched, _ = model(input_lab_tf_batched, training=False)
            enhanced_rgb_tf_batched = tf_lab_normalized_to_rgb(enhanced_lab_tf_batched)
            cnn_output_rgb_np = enhanced_rgb_tf_batched.numpy()[0]
            cnn_output_rgb_np = np.clip(cnn_output_rgb_np, 0.0, 1.0)

            print("  Applying post-processing...")
            final_output_rgb_np = postprocess_image_hybrid(cnn_output_rgb_np)

            base_filename = os.path.basename(img_path)
            name, ext = os.path.splitext(base_filename)

            gt_rgb_np_resized = None; metrics_cnn = None; metrics_hybrid = None
            if args.ground_truth_dir:
                gt_path = os.path.join(args.ground_truth_dir, base_filename)
                if os.path.exists(gt_path):
                    gt_bgr_uint8_gt = cv2.imread(gt_path)
                    if gt_bgr_uint8_gt is not None:
                        gt_rgb_uint8_gt = cv2.cvtColor(gt_bgr_uint8_gt, cv2.COLOR_BGR2RGB)
                        gt_rgb_np_gt = (gt_rgb_uint8_gt / 255.0).astype(np.float32)
                        gt_rgb_np_resized = cv2.resize(gt_rgb_np_gt, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                        
                        metrics_cnn = calculate_metrics_inference(gt_rgb_np_resized, cnn_output_rgb_np)
                        metrics_hybrid = calculate_metrics_inference(gt_rgb_np_resized, final_output_rgb_np)
                        all_metrics_summary.append({'image': base_filename, 'cnn': metrics_cnn, 'hybrid': metrics_hybrid})
                        
                        print(f"    Metrics (CNN vs GT): PSNR: {metrics_cnn.get('psnr', np.nan):.2f}, SSIM: {metrics_cnn.get('ssim', np.nan):.4f}, CIEDE2k: {metrics_cnn.get('ciede2000', np.nan):.2f}")
                        print(f"    Metrics (Hybrid vs GT): PSNR: {metrics_hybrid.get('psnr', np.nan):.2f}, SSIM: {metrics_hybrid.get('ssim', np.nan):.4f}, CIEDE2k: {metrics_hybrid.get('ciede2000', np.nan):.2f}")
                    else: print(f"Warning: Could not read ground truth image {gt_path}")
                else: print(f"Warning: Ground truth not found for {base_filename} at {gt_path}")

            if args.output_dir:
                # cv2.imwrite(os.path.join(args.output_dir, f"{name}_hybrid_output{ext}"),
                #             cv2.cvtColor((final_output_rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                fig.suptitle(f"Image: {base_filename}", fontsize=14, y=0.98)

                axes[0, 0].imshow(resized_input_rgb_np); axes[0, 0].set_title('Input (Resized)'); axes[0, 0].axis('off')
                axes[0, 1].imshow(cnn_output_rgb_np); axes[0, 1].set_title('CNN Output'); axes[0, 1].axis('off')
                if metrics_cnn:
                    cnn_metric_text = (f"PSNR: {metrics_cnn.get('psnr', np.nan):.2f}\n"
                                       f"SSIM: {metrics_cnn.get('ssim', np.nan):.4f}\n"
                                       f"CIEDE2k: {metrics_cnn.get('ciede2000', np.nan):.2f}")
                    axes[0, 1].text(0.03, -0.12, cnn_metric_text, transform=axes[0, 1].transAxes,
                                    fontsize=8, verticalalignment='top', linespacing=1.5,
                                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

                axes[1, 0].imshow(final_output_rgb_np); axes[1, 0].set_title('Hybrid Output'); axes[1, 0].axis('off')
                if metrics_hybrid:
                    hybrid_metric_text = (f"PSNR: {metrics_hybrid.get('psnr', np.nan):.2f}\n"
                                          f"SSIM: {metrics_hybrid.get('ssim', np.nan):.4f}\n"
                                          f"CIEDE2k: {metrics_hybrid.get('ciede2000', np.nan):.2f}")
                    axes[1, 0].text(0.03, -0.12, hybrid_metric_text, transform=axes[1, 0].transAxes,
                                    fontsize=8, verticalalignment='top', linespacing=1.5,
                                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

                if gt_rgb_np_resized is not None:
                    axes[1, 1].imshow(gt_rgb_np_resized); axes[1, 1].set_title('Ground Truth')
                else: axes[1, 1].set_title('No Ground Truth')
                axes[1, 1].axis('off')
                
                plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjusted rect for suptitle and text
                plot_save_path = os.path.join(args.output_dir, f"{name}_summary_2x2.png")
                plt.savefig(plot_save_path); plt.close(fig)
                print(f"  Saved 2x2 summary plot to {plot_save_path}")
        
        except Exception as e:
            print(f"Error processing image {img_path}: {e}"); import traceback; traceback.print_exc()

    if all_metrics_summary:
        avg_cnn_psnr = np.nanmean([m['cnn']['psnr'] for m in all_metrics_summary if m['cnn'] and 'psnr' in m['cnn'] and not np.isnan(m['cnn']['psnr'])])
        avg_cnn_ssim = np.nanmean([m['cnn']['ssim'] for m in all_metrics_summary if m['cnn'] and 'ssim' in m['cnn'] and not np.isnan(m['cnn']['ssim'])])
        avg_cnn_ciede = np.nanmean([m['cnn']['ciede2000'] for m in all_metrics_summary if m['cnn'] and 'ciede2000' in m['cnn'] and not np.isnan(m['cnn']['ciede2000'])])
        avg_hybrid_psnr = np.nanmean([m['hybrid']['psnr'] for m in all_metrics_summary if m['hybrid'] and 'psnr' in m['hybrid'] and not np.isnan(m['hybrid']['psnr'])])
        avg_hybrid_ssim = np.nanmean([m['hybrid']['ssim'] for m in all_metrics_summary if m['hybrid'] and 'ssim' in m['hybrid'] and not np.isnan(m['hybrid']['ssim'])])
        avg_hybrid_ciede = np.nanmean([m['hybrid']['ciede2000'] for m in all_metrics_summary if m['hybrid'] and 'ciede2000' in m['hybrid'] and not np.isnan(m['hybrid']['ciede2000'])])

        print("\n--- Average Metrics Summary ---")
        if not any(all_metrics_summary): print("No metrics were successfully calculated for averaging.")
        else:
            print(f"Processed {len(all_metrics_summary)} image(s) with ground truth for metrics.")
            print(f"CNN Output Avg: PSNR: {avg_cnn_psnr:.2f}, SSIM: {avg_cnn_ssim:.4f}, CIEDE2k: {avg_cnn_ciede:.2f}")
            print(f"Hybrid Output Avg: PSNR: {avg_hybrid_psnr:.2f}, SSIM: {avg_hybrid_ssim:.4f}, CIEDE2k: {avg_hybrid_ciede:.2f}")
    elif args.ground_truth_dir :
        print("\n--- Average Metrics Summary ---")
        print("No images with corresponding ground truth were successfully processed for metrics calculation.")


    print("--- Hybrid Inference Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid Inference Pipeline for Color Cast Removal.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained .keras model file.')
    parser.add_argument('--input-path', type=str, required=True, help='Path to an input image or a directory of images.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save processed images and plots. (Optional)')
    parser.add_argument('--ground-truth-dir', type=str, default=None, help='Directory containing ground truth images for metric calculation. (Optional, filenames must match inputs)')
    cli_args = parser.parse_args()
    run_hybrid_inference(cli_args)